"""
Analytical placer — ePlace/DREAMPlace style global placement on GPU.

Pipeline:
  1. Global placement: Nesterov-accelerated gradient descent on
        f(x) = W_lse(x) + λ * N_e(x)
     where
        W_lse = log-sum-exp smoothed HPWL over nets (pin-level)
        N_e   = electrostatic potential energy from a 2D Poisson solve
                over the density overflow (bell-shaped rasterization).
     λ is annealed adaptively from a small value as density overflow falls.
  2. Legalization: size-sorted spiral search (anchor = post-GP position)
     guarantees zero overlaps for hard macros.
  3. Soft macros keep their post-GP positions (they naturally overlap).

Usage:
    uv run evaluate submissions/analytical/placer.py -b ibm01
    uv run evaluate submissions/analytical/placer.py --all
"""

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark


def _pick_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(0)
            if cap >= (7, 0):
                device = torch.device("cuda")
        except Exception:
            pass
    return device


# ─── Net / pin model ─────────────────────────────────────────────────────────


def _build_net_tensors(benchmark: Benchmark, device):
    """
    Flatten the net hypergraph into pin-level tensors suitable for GPU scatter ops.

    Returns:
        pin_node:   [P] int — node index for each pin
                               [0, n_all):        macro (hard/soft) index
                               [n_all, n_all+nP): port index
        pin_offset: [P, 2] float — offset from macro center (0 for soft / ports)
        pin_net:    [P] int — which net each pin belongs to
        net_count:  int — number of nets (with ≥2 pins)
        net_weight: [n_nets] float — weight per net (inverse clique weight)
    """
    n_all = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    n_ports = benchmark.port_positions.shape[0]

    # For each (macro, pin) pair, how to map:
    #   hard macros: have pin offsets in benchmark.macro_pin_offsets[i]
    #   soft macros: one "pin" at center (offset 0)
    #   ports:       fixed node, offset 0
    # The benchmark net_nodes list gives unique nodes per net — we treat each
    # node as a single pin at its canonical location. That matches how the
    # loader built net_nodes (sorted set of nodes) and is what PlacementCost
    # ultimately uses for the bounding-box wirelength.

    pin_node_list = []
    pin_offset_list = []
    pin_net_list = []
    net_weight_list = []

    for net_idx, nodes in enumerate(benchmark.net_nodes):
        if nodes.numel() < 2:
            continue
        new_net_id = len(net_weight_list)
        # Clique-net model weight: 2 / k (common for HPWL proxies with many-pin nets)
        k = nodes.numel()
        w = 2.0 / max(k, 2)
        net_weight_list.append(w)
        for node in nodes.tolist():
            pin_node_list.append(node)
            pin_offset_list.append((0.0, 0.0))
            pin_net_list.append(new_net_id)

    if not pin_node_list:
        return (
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, 2, dtype=torch.float32, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            0,
            torch.zeros(0, dtype=torch.float32, device=device),
        )

    pin_node = torch.tensor(pin_node_list, dtype=torch.long, device=device)
    pin_offset = torch.tensor(pin_offset_list, dtype=torch.float32, device=device)
    pin_net = torch.tensor(pin_net_list, dtype=torch.long, device=device)
    n_nets = len(net_weight_list)
    net_weight = torch.tensor(net_weight_list, dtype=torch.float32, device=device)

    return pin_node, pin_offset, pin_net, n_nets, net_weight


def _gather_pin_positions(pos, ports, pin_node, pin_offset, n_all):
    """
    Return [P, 2] pin coordinates, pulling from movable/fixed macro pos or
    from the fixed port tensor (indices ≥ n_all).
    """
    is_port = pin_node >= n_all
    safe_macro_idx = torch.where(is_port, torch.zeros_like(pin_node), pin_node)
    safe_port_idx = torch.where(is_port, pin_node - n_all, torch.zeros_like(pin_node))
    macro_coords = pos[safe_macro_idx] + pin_offset
    port_coords = ports[safe_port_idx] if ports.shape[0] > 0 else torch.zeros_like(macro_coords)
    return torch.where(is_port.unsqueeze(-1), port_coords, macro_coords)


# ─── Log-sum-exp wirelength ──────────────────────────────────────────────────


def _lse_wirelength(pin_pos, pin_net, n_nets, net_weight, gamma):
    """
    Weighted LSE HPWL with per-net max-trick for numerical stability.

    For each net n, contribution is:
        γ·(log Σ_i exp(x_i / γ) - log Σ_i exp(-x_i / γ)) + same for y
    """
    if pin_pos.numel() == 0 or n_nets == 0:
        return pin_pos.sum() * 0.0

    device = pin_pos.device
    inv_g = 1.0 / gamma
    x = pin_pos[:, 0] * inv_g
    y = pin_pos[:, 1] * inv_g

    # Per-net max / min stabilisation — detached (they act as offsets only)
    with torch.no_grad():
        neg_inf = torch.full((n_nets,), -1e30, device=device)
        x_max = neg_inf.clone().scatter_reduce(0, pin_net, x.detach(), reduce="amax", include_self=True)
        x_min = neg_inf.clone().scatter_reduce(0, pin_net, (-x).detach(), reduce="amax", include_self=True)
        y_max = neg_inf.clone().scatter_reduce(0, pin_net, y.detach(), reduce="amax", include_self=True)
        y_min = neg_inf.clone().scatter_reduce(0, pin_net, (-y).detach(), reduce="amax", include_self=True)

    ex_pos = torch.exp(x - x_max[pin_net])
    ex_neg = torch.exp(-x - x_min[pin_net])
    ey_pos = torch.exp(y - y_max[pin_net])
    ey_neg = torch.exp(-y - y_min[pin_net])

    sx_pos = torch.zeros(n_nets, device=device).scatter_add(0, pin_net, ex_pos)
    sx_neg = torch.zeros(n_nets, device=device).scatter_add(0, pin_net, ex_neg)
    sy_pos = torch.zeros(n_nets, device=device).scatter_add(0, pin_net, ey_pos)
    sy_neg = torch.zeros(n_nets, device=device).scatter_add(0, pin_net, ey_neg)

    wl_x = gamma * (torch.log(sx_pos + 1e-20) + x_max + torch.log(sx_neg + 1e-20) + x_min)
    wl_y = gamma * (torch.log(sy_pos + 1e-20) + y_max + torch.log(sy_neg + 1e-20) + y_min)

    return (net_weight * (wl_x + wl_y)).sum()


# ─── Electrostatic density ───────────────────────────────────────────────────


def _rasterize_density(pos, sizes, cw, ch, nx, ny):
    """
    Rasterize macros as filled rectangles onto an [ny, nx] grid.
    Uses exact area overlap (differentiable, piecewise-linear in position).
    Returns density map [ny, nx] in units of (macro area covered / bin area).
    """
    device = pos.device
    N = pos.shape[0]
    bw = cw / nx
    bh = ch / ny

    # Bin edges: centers spacing = bw
    x_lo = torch.arange(nx, device=device, dtype=pos.dtype) * bw  # [nx]
    y_lo = torch.arange(ny, device=device, dtype=pos.dtype) * bh  # [ny]

    # Macro extents
    lx = pos[:, 0:1] - sizes[:, 0:1] / 2  # [N, 1]
    ux = pos[:, 0:1] + sizes[:, 0:1] / 2
    ly = pos[:, 1:2] - sizes[:, 1:2] / 2
    uy = pos[:, 1:2] + sizes[:, 1:2] / 2

    # Overlap with each bin along x: [N, nx]
    ov_x = torch.clamp(torch.minimum(ux, x_lo + bw) - torch.maximum(lx, x_lo), min=0.0)
    ov_y = torch.clamp(torch.minimum(uy, y_lo + bh) - torch.maximum(ly, y_lo), min=0.0)

    # Density: sum over macros of outer product. Use einsum for efficiency.
    # density[i,j] = sum_m ov_y[m,i] * ov_x[m,j]
    density = torch.einsum("mi,mj->ij", ov_y, ov_x) / (bw * bh)
    return density


def _poisson_fft(rho, bw, bh):
    """
    Solve ∇²ψ = -ρ on a periodic torus (approximation of Neumann BC) via FFT.
    rho must be zero-mean for a well-defined solution; we zero the DC coefficient.
    Returns ψ [ny, nx].
    """
    ny, nx = rho.shape
    device = rho.device
    dtype = rho.dtype

    rho_hat = torch.fft.rfft2(rho)

    ky = torch.fft.fftfreq(ny, d=bh, dtype=dtype, device=device) * (2.0 * math.pi)
    kx = torch.fft.rfftfreq(nx, d=bw, dtype=dtype, device=device) * (2.0 * math.pi)
    k2 = ky.unsqueeze(1) ** 2 + kx.unsqueeze(0) ** 2

    # Avoid div-by-zero at DC; set that component of ψ to 0 after the divide.
    k2_safe = torch.where(k2 == 0, torch.ones_like(k2), k2)
    psi_hat = rho_hat / k2_safe
    psi_hat[0, 0] = 0.0  # enforce zero-mean potential

    psi = torch.fft.irfft2(psi_hat, s=(ny, nx))
    return psi


def _density_energy(pos, sizes, cw, ch, nx, ny, target_density):
    """
    Electrostatic density energy.  We use rho_overflow = rho - target
    (can be negative), solve for ψ, and return 0.5 * <rho_overflow, ψ>.

    The *target* is the uniform background charge required to make the
    system net-neutral.  Only overflow cells produce net force.
    """
    rho = _rasterize_density(pos, sizes, cw, ch, nx, ny)
    rho_overflow = rho - target_density
    # Ensure exactly zero mean (FFT requires this):
    rho_overflow = rho_overflow - rho_overflow.mean()

    bw = cw / nx
    bh = ch / ny
    psi = _poisson_fft(rho_overflow, bw, bh)
    # Energy scaled by bin area so it has physical units of "overflow × potential"
    energy = 0.5 * (rho_overflow * psi).sum() * (bw * bh)
    return energy, rho


# ─── Detailed placement ─────────────────────────────────────────────────────


def _detailed_place_hard(
    legal_hw, movable_np, sizes_np, hw, hh, cw, ch,
    pin_node_np, pin_net_np, net_weight_np, port_pos_np, soft_pos_np,
    n_hard, n_all, n_nets, max_passes=4, seed=42,
):
    """
    Greedy local HPWL shifts for hard macros only, with hard-overlap rejection.

    Uses bounding-box HPWL (exact TILOS cost) rather than LSE — this is
    post-legalization, so the smoothness of LSE isn't needed.

    Cache per-net HPWL, update only nets touching the moved macro each step.
    """
    import random as _rand
    _rand.seed(seed)

    # Per-net node list & per-hard-macro net list
    net_nodes = [[] for _ in range(n_nets)]
    hard_nets = [set() for _ in range(n_hard)]
    P = len(pin_node_np)
    for p in range(P):
        node = int(pin_node_np[p])
        net = int(pin_net_np[p])
        net_nodes[net].append(node)
        if node < n_hard:
            hard_nets[node].add(net)
    net_nodes = [list(set(nl)) for nl in net_nodes]
    hard_nets = [sorted(s) for s in hard_nets]

    # Flat xy arrays indexed by node
    n_soft = n_all - n_hard
    n_ports = port_pos_np.shape[0]
    pos_x = np.empty(n_all + n_ports, dtype=np.float64)
    pos_y = np.empty(n_all + n_ports, dtype=np.float64)
    pos_x[:n_hard] = legal_hw[:, 0]
    pos_y[:n_hard] = legal_hw[:, 1]
    if n_soft > 0:
        pos_x[n_hard:n_all] = soft_pos_np[:, 0]
        pos_y[n_hard:n_all] = soft_pos_np[:, 1]
    if n_ports > 0:
        pos_x[n_all:] = port_pos_np[:, 0]
        pos_y[n_all:] = port_pos_np[:, 1]

    def _net_hpwl(nodes):
        k = len(nodes)
        if k < 2:
            return 0.0
        xmin = xmax = pos_x[nodes[0]]
        ymin = ymax = pos_y[nodes[0]]
        for idx in range(1, k):
            n = nodes[idx]
            x = pos_x[n]
            y = pos_y[n]
            if x < xmin: xmin = x
            elif x > xmax: xmax = x
            if y < ymin: ymin = y
            elif y > ymax: ymax = y
        return (xmax - xmin) + (ymax - ymin)

    net_hpwls = [_net_hpwl(net_nodes[n]) for n in range(n_nets)]

    # AABB arrays for fast overlap test
    lx = np.empty(n_hard); ly = np.empty(n_hard)
    ux = np.empty(n_hard); uy = np.empty(n_hard)
    for j in range(n_hard):
        lx[j] = legal_hw[j, 0] - sizes_np[j, 0] / 2
        ly[j] = legal_hw[j, 1] - sizes_np[j, 1] / 2
        ux[j] = legal_hw[j, 0] + sizes_np[j, 0] / 2
        uy[j] = legal_hw[j, 1] + sizes_np[j, 1] / 2

    gap = 0.01
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for pass_i in range(max_passes):
        improved = False
        order = list(range(n_hard))
        _rand.shuffle(order)
        for i in order:
            if not movable_np[i]:
                continue
            my_nets = hard_nets[i]
            if not my_nets:
                continue
            w = sizes_np[i, 0]
            h = sizes_np[i, 1]
            old_x = pos_x[i]
            old_y = pos_y[i]
            best_x = old_x
            best_y = old_y
            best_delta = -1e-9
            old_sum = 0.0
            for n in my_nets:
                old_sum += net_weight_np[n] * net_hpwls[n]

            for s_frac in (0.5, 0.25, 0.1):
                sx = w * s_frac
                sy = h * s_frac
                for dx, dy in dirs:
                    nx_ = old_x + dx * sx
                    ny_ = old_y + dy * sy
                    if nx_ < hw[i]: nx_ = hw[i]
                    elif nx_ > cw - hw[i]: nx_ = cw - hw[i]
                    if ny_ < hh[i]: ny_ = hh[i]
                    elif ny_ > ch - hh[i]: ny_ = ch - hh[i]
                    if abs(nx_ - old_x) < 1e-9 and abs(ny_ - old_y) < 1e-9:
                        continue
                    # Overlap test: new AABB of i vs all other hard AABBs.
                    # Clash if separation on *every* axis is less than gap.
                    nlx = nx_ - w / 2; nly = ny_ - h / 2
                    nux = nx_ + w / 2; nuy = ny_ + h / 2
                    clash = False
                    for j in range(n_hard):
                        if j == i:
                            continue
                        if nux > lx[j] - gap and nlx < ux[j] + gap and nuy > ly[j] - gap and nly < uy[j] + gap:
                            clash = True
                            break
                    if clash:
                        continue
                    # Compute new HPWL of affected nets
                    pos_x[i] = nx_
                    pos_y[i] = ny_
                    new_sum = 0.0
                    for n in my_nets:
                        new_sum += net_weight_np[n] * _net_hpwl(net_nodes[n])
                    delta = new_sum - old_sum
                    if delta < best_delta:
                        best_delta = delta
                        best_x = nx_
                        best_y = ny_
                    # Restore
                    pos_x[i] = old_x
                    pos_y[i] = old_y

            if best_delta < -1e-9:
                pos_x[i] = best_x
                pos_y[i] = best_y
                legal_hw[i, 0] = best_x
                legal_hw[i, 1] = best_y
                lx[i] = best_x - w / 2
                ly[i] = best_y - h / 2
                ux[i] = best_x + w / 2
                uy[i] = best_y + h / 2
                for n in my_nets:
                    net_hpwls[n] = _net_hpwl(net_nodes[n])
                improved = True

        if not improved:
            break

    return legal_hw


# ─── Legalization ────────────────────────────────────────────────────────────


def _legalize(pos, anchor, movable, sizes, hw, hh, cw, ch, n):
    """
    Size-sorted spiral legalization starting from given positions.
    Largest first, each placed macro is frozen; movement quality is measured
    against *anchor* (ideally the post-GP position for min displacement).
    """
    gap = 0.01
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            c = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap) & placed
            c[idx] = False
            if not c.any():
                placed[idx] = True
                continue

        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.2
        best_p = legal[idx].copy()
        best_d = float("inf")
        for r in range(1, 300):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    ncx = np.clip(legal[idx, 0] + dxm * step, hw[idx], cw - hw[idx])
                    ncy = np.clip(legal[idx, 1] + dym * step, hh[idx], ch - hh[idx])
                    if placed.any():
                        dx = np.abs(ncx - legal[:, 0])
                        dy = np.abs(ncy - legal[:, 1])
                        c = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap) & placed
                        c[idx] = False
                        if c.any():
                            continue
                    d = (ncx - anchor[idx, 0]) ** 2 + (ncy - anchor[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([ncx, ncy])
                        found = True
            if found:
                break
        legal[idx] = best_p
        placed[idx] = True

    return legal


# ─── Placer ──────────────────────────────────────────────────────────────────


class AnalyticalPlacer:
    def __init__(
        self,
        seed: int = 42,
        gp_iters: int = 800,
        n_grid: int = 128,
        target_overflow: float = 0.08,
        lambda0: float = 0.01,
        lambda_growth: float = 1.02,
        lambda_max: float = 1000.0,
        gamma_scale: float = 4.0,
        step_size: float = 0.04,
        anchor_w0: float = 0.15,
        anchor_decay: float = 0.995,
        n_trials: int = 8,
        trial_time_budget_s: float = 2400.0,
        use_autodmp: bool = True,
        autodmp_trials: int = 40,
        autodmp_startup: int = 6,
    ):
        self.seed = seed
        self.gp_iters = gp_iters
        self.n_grid = n_grid
        self.target_overflow = target_overflow
        self.lambda0 = lambda0
        self.lambda_growth = lambda_growth
        self.lambda_max = lambda_max
        self.gamma_scale = gamma_scale  # γ = gamma_scale * bin_width
        self.step_size = step_size
        self.anchor_w0 = anchor_w0
        self.anchor_decay = anchor_decay
        # Multi-start: run up to n_trials diverse trials, keep the best by the
        # real TILOS proxy (compute_proxy_cost).  trial_time_budget_s caps the
        # wall-clock — if exceeded, we stop launching new trials and return
        # the best so far.  Set n_trials=1 for single-run behavior.
        self.n_trials = n_trials
        self.trial_time_budget_s = trial_time_budget_s
        # AutoDMP-style per-benchmark hyperparameter search via Optuna TPE.
        # When enabled, supersedes the hand-coded multi-start configs.
        # autodmp_trials is an upper bound — trial_time_budget_s still caps wall time.
        # autodmp_startup is the number of random-sampled warmup trials before TPE kicks in.
        self.use_autodmp = use_autodmp
        self.autodmp_trials = autodmp_trials
        self.autodmp_startup = autodmp_startup
        self._plc_cache = {}

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Multi-start orchestrator: runs diverse trials, returns the best by TILOS proxy."""
        if self.n_trials <= 1:
            return self._place_single(benchmark)

        plc = self._load_scoring_plc(benchmark)
        if plc is None:
            # No scorer available → nothing to pick between; run baseline only.
            return self._place_single(benchmark)

        if self.use_autodmp:
            result = self._autodmp_search(benchmark, plc)
            if result is not None:
                return result
            # Fall through to legacy multi-start if AutoDMP can't run
            # (e.g., Optuna missing).

        configs = self._build_trial_configs()

        best_placement = None
        best_cost = float("inf")
        t_start = time.time()

        for i, cfg in enumerate(configs[: self.n_trials]):
            elapsed = time.time() - t_start
            # Always run the first trial; afterwards, respect the budget
            if i > 0 and elapsed >= self.trial_time_budget_s:
                break
            try:
                placement = self._place_single(benchmark, **cfg)
            except Exception:
                continue
            cost = self._score(placement, benchmark, plc)
            if cost < best_cost:
                best_cost = cost
                best_placement = placement
            # If a single trial already blew the budget, don't start another
            if (time.time() - t_start) >= self.trial_time_budget_s:
                break

        return best_placement if best_placement is not None else self._place_single(benchmark)

    # ── AutoDMP-style search ────────────────────────────────────────────────

    def _autodmp_search(self, benchmark: Benchmark, plc) -> "torch.Tensor | None":
        """
        AutoDMP-style per-benchmark hyperparameter search via Optuna TPE.

        Follows the Agnesina et al. ISPD 2023 recipe: treat DREAMPlace-class
        hyperparameters as a black-box search space, use TPE (tree-structured
        Parzen estimator) as the surrogate, proxy cost as the objective, and
        a wall-clock budget as the stop criterion. A deterministic warmup
        trial seeds the study with the current hand-tuned defaults so we
        never regress below the baseline even if TPE needs time to converge.
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            return None

        # Silence per-trial spam; we track progress ourselves if needed.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        t_start = time.time()

        best_placement = {"pos": None, "cost": float("inf")}

        def objective(trial: "optuna.Trial") -> float:
            # Respect wall-clock before starting a new trial
            if (time.time() - t_start) >= self.trial_time_budget_s:
                trial.study.stop()
                raise optuna.TrialPruned()

            cfg = {
                "lambda0": trial.suggest_float("lambda0", 1e-3, 1e-1, log=True),
                "lambda_growth": trial.suggest_float("lambda_growth", 1.005, 1.05),
                "gamma_scale": trial.suggest_float("gamma_scale", 1.0, 8.0, log=True),
                "step_size": trial.suggest_float("step_size", 0.01, 0.1, log=True),
                "anchor_w0": trial.suggest_float("anchor_w0", 0.0, 0.3),
                "anchor_decay": trial.suggest_float("anchor_decay", 0.985, 0.999),
                "target_overflow": trial.suggest_float("target_overflow", 0.05, 0.15),
                "perturb_sigma": trial.suggest_float("perturb_sigma", 0.0, 0.08),
                "n_grid": trial.suggest_categorical("n_grid", [96, 128, 192, 256]),
                "seed": trial.suggest_int("seed", 0, 10_000),
            }

            try:
                placement = self._place_single(benchmark, **cfg)
            except Exception:
                return float("inf")

            cost = self._score(placement, benchmark, plc)
            improved = cost < best_placement["cost"]
            if improved:
                best_placement["cost"] = cost
                best_placement["pos"] = placement

            if os.environ.get("AUTODMP_LOG", "0") == "1":
                mark = "*" if improved else " "
                print(
                    f"[autodmp] {mark} trial {trial.number:3d} "
                    f"proxy={cost:.4f} best={best_placement['cost']:.4f} "
                    f"t={time.time() - t_start:6.1f}s",
                    flush=True,
                )

            # Stop as soon as budget is exhausted mid-trial
            if (time.time() - t_start) >= self.trial_time_budget_s:
                trial.study.stop()
            return cost

        sampler = TPESampler(
            seed=self.seed,
            n_startup_trials=self.autodmp_startup,
            multivariate=True,
            group=True,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # Warm-start: enqueue the hand-tuned multi-start configs so TPE starts
        # with ≥ N_good observations. Ensures AutoDMP can never regress below
        # the legacy multi-start in the worst case.
        base_grid = self.n_grid if self.n_grid in (96, 128, 192, 256) else 128
        warm_configs = [
            # Baseline defaults
            dict(perturb_sigma=0.0, seed=self.seed),
            # Mirrors _build_trial_configs overrides, mapped onto the search space
            dict(perturb_sigma=0.02, seed=123),
            dict(perturb_sigma=0.05, seed=456),
            dict(perturb_sigma=0.03, anchor_w0=0.0, seed=789),
            dict(perturb_sigma=0.05, lambda0=0.005, seed=1001),
            dict(perturb_sigma=0.05, lambda0=0.02, seed=1337),
            dict(perturb_sigma=0.03, step_size=0.02, seed=2023),
            dict(perturb_sigma=0.03, gamma_scale=2.0, seed=4242),
        ]
        for overrides in warm_configs:
            params = {
                "lambda0": self.lambda0,
                "lambda_growth": self.lambda_growth,
                "gamma_scale": self.gamma_scale,
                "step_size": self.step_size,
                "anchor_w0": self.anchor_w0,
                "anchor_decay": self.anchor_decay,
                "target_overflow": self.target_overflow,
                "perturb_sigma": 0.0,
                "n_grid": base_grid,
                "seed": self.seed,
            }
            params.update(overrides)
            study.enqueue_trial(params)

        try:
            study.optimize(
                objective,
                n_trials=self.autodmp_trials,
                timeout=self.trial_time_budget_s,
                catch=(Exception,),
                show_progress_bar=False,
            )
        except Exception:
            pass

        return best_placement["pos"]

    # ── Scoring / plc loading ───────────────────────────────────────────────

    def _load_scoring_plc(self, benchmark: Benchmark):
        """Load a PlacementCost for TILOS proxy scoring. Cached by benchmark name.
        Returns None if no known path matches (falls back to no scoring / first trial wins)."""
        if benchmark.name in self._plc_cache:
            return self._plc_cache[benchmark.name]
        candidates = [
            f"external/MacroPlacement/Testcases/ICCAD04/{benchmark.name}/netlist.pb.txt",
            f"external/MacroPlacement/Flows/NanGate45/{benchmark.name}/netlist/output_CT_Grouping/netlist.pb.txt",
        ]
        plc = None
        for netlist in candidates:
            if os.path.exists(netlist):
                try:
                    from macro_place._plc import PlacementCost
                    plc = PlacementCost(netlist)
                    initial = netlist.replace("netlist.pb.txt", "initial.plc")
                    if os.path.exists(initial):
                        plc.restore_placement(initial, ifInital=True, ifReadComment=True)
                    break
                except Exception:
                    plc = None
        self._plc_cache[benchmark.name] = plc
        return plc

    def _score(self, placement: torch.Tensor, benchmark: Benchmark, plc) -> float:
        """Score via compute_proxy_cost. Returns +inf on any overlap or failure."""
        try:
            from macro_place.objective import compute_proxy_cost
            r = compute_proxy_cost(placement, benchmark, plc)
            if r.get("overlap_count", 0) > 0:
                return float("inf")
            return float(r["proxy_cost"])
        except Exception:
            return float("inf")

    def _build_trial_configs(self) -> list:
        """Diverse configs for multi-start. Always includes a baseline trial first."""
        return [
            {},  # baseline (no perturb, current hyperparams)
            {"perturb_sigma": 0.02, "seed": 123},
            {"perturb_sigma": 0.05, "seed": 456},
            {"perturb_sigma": 0.03, "anchor_w0": 0.0, "seed": 789},
            {"perturb_sigma": 0.05, "lambda0": 0.005, "seed": 1001},
            {"perturb_sigma": 0.05, "lambda0": 0.02, "seed": 1337},
            {"perturb_sigma": 0.03, "step_size": 0.02, "seed": 2023},
            {"perturb_sigma": 0.03, "gamma_scale": 2.0, "seed": 4242},
        ]

    def _place_single(self, benchmark: Benchmark, perturb_sigma: float = 0.0, **overrides) -> torch.Tensor:
        """Single GP + legalization run. Attribute overrides are applied for this trial only."""
        # Temporarily apply overrides
        saved = {}
        for k, v in overrides.items():
            if hasattr(self, k):
                saved[k] = getattr(self, k)
                setattr(self, k, v)
        try:
            return self._place_core(benchmark, perturb_sigma=perturb_sigma)
        finally:
            for k, v in saved.items():
                setattr(self, k, v)

    def _place_core(self, benchmark: Benchmark, perturb_sigma: float = 0.0) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        device = _pick_device()

        n_all = benchmark.num_macros
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        sizes = benchmark.macro_sizes.float().to(device)  # [n_all, 2]
        anchor0 = benchmark.macro_positions.float().to(device)  # initial placement
        fixed = benchmark.macro_fixed.to(device)  # [n_all] bool
        movable = (~fixed).float().unsqueeze(1)  # [n_all, 1]
        ports = benchmark.port_positions.to(device) if benchmark.port_positions.numel() > 0 else torch.zeros(0, 2, device=device)

        # Net tensors
        pin_node, pin_offset, pin_net, n_nets, net_weight = _build_net_tensors(benchmark, device)

        # Grid setup
        nx = self._pick_grid(cw, ch, self.n_grid, axis="x")
        ny = self._pick_grid(cw, ch, self.n_grid, axis="y")
        bw = cw / nx
        bh = ch / ny

        # Target density: pretend all macro area spreads uniformly
        total_area = float((sizes[:n_hard, 0] * sizes[:n_hard, 1]).sum().item())
        # Include soft macros in the density field too so hard macros respect their footprint
        soft_area = float((sizes[n_hard:, 0] * sizes[n_hard:, 1]).sum().item()) if n_all > n_hard else 0.0
        canvas_area = cw * ch
        target_density = (total_area + soft_area) / canvas_area
        # Safety: cap at the achievable fill
        target_density = max(min(target_density, 1.0), 0.05)

        # LSE temperature: start large (smoothing), decrease over time
        # Scale with canvas / bin size so it's dimensionally consistent
        diag = math.sqrt(cw * cw + ch * ch)
        gamma_start = diag * 0.02 * self.gamma_scale
        gamma_end = diag * 0.002

        # Global placement
        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        lb = torch.stack([hw, hh], dim=1)
        ub = torch.stack([cw - hw, ch - hh], dim=1)

        pos = anchor0.clone()
        if perturb_sigma > 0:
            # Seeded-random initial perturbation (torch.manual_seed already called above).
            # Scale by canvas diag so sigma is unitless fraction of chip extent.
            noise = torch.randn_like(anchor0)
            pos = pos + perturb_sigma * diag * noise * movable
            pos = torch.min(torch.max(pos, lb), ub)
        y = pos.clone()
        lam = self.lambda0

        # ── Estimate initial gradient norms → balance λ so density pull ≈ WL pull ──
        y_tmp = y.detach().clone().requires_grad_(True)
        pp = _gather_pin_positions(y_tmp, ports, pin_node, pin_offset, n_all)
        wl_only = _lse_wirelength(pp, pin_net, n_nets, net_weight, gamma_start)
        (wl_g,) = torch.autograd.grad(wl_only, y_tmp)

        y_tmp = y.detach().clone().requires_grad_(True)
        ed_only, _ = _density_energy(y_tmp, sizes, cw, ch, nx, ny, target_density)
        (ed_g,) = torch.autograd.grad(ed_only, y_tmp)

        mov_mask = movable.squeeze(1) > 0
        wl_norm = wl_g[mov_mask].norm().item() + 1e-12
        ed_norm = ed_g[mov_mask].norm().item() + 1e-12
        lam = self.lambda0 * (wl_norm / ed_norm)

        # Keep best by (overflow-small + wirelength-small)
        best_pos = pos.clone()
        best_score = float("inf")
        best_overflow = float("inf")

        # Anchor-weight normalization: pick α0 so anchor gradient ~ wl gradient initially
        anchor_scale = wl_norm  # gradient magnitude of WL
        anchor_w = self.anchor_w0 * anchor_scale

        a = 1.0  # Nesterov momentum coefficient
        prev_ov = float("inf")
        for it in range(self.gp_iters):
            frac = it / max(1, self.gp_iters - 1)
            gamma = gamma_start * ((gamma_end / gamma_start) ** frac)

            y_req = y.detach().clone().requires_grad_(True)
            pin_pos = _gather_pin_positions(y_req, ports, pin_node, pin_offset, n_all)
            wl = _lse_wirelength(pin_pos, pin_net, n_nets, net_weight, gamma)
            ed, rho = _density_energy(y_req, sizes, cw, ch, nx, ny, target_density)

            # Anchor regularization: pull toward initial position
            disp = y_req - anchor0
            anchor_loss = (disp * disp * movable).sum()

            loss = wl + lam * ed + anchor_w * anchor_loss

            (grad,) = torch.autograd.grad(loss, y_req)
            grad = grad * movable  # freeze fixed macros

            # Adaptive step
            gnorm = grad.norm().item() + 1e-12
            step = self.step_size * diag / gnorm

            pos_new = y - step * grad
            pos_new = torch.min(torch.max(pos_new, lb), ub)

            # Nesterov momentum
            a_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * a * a))
            beta = (a - 1.0) / a_new
            y = pos_new + beta * (pos_new - pos)
            y = torch.min(torch.max(y, lb), ub)
            pos = pos_new
            a = a_new

            with torch.no_grad():
                overflow_mass = torch.clamp(rho - target_density, min=0.0).sum() * (bw * bh)
                ov = float(overflow_mass.item() / max(total_area, 1e-9))

            # Anneal λ and decay anchor
            if it > 5:
                lam = min(lam * self.lambda_growth, self.lambda_max)
                anchor_w = anchor_w * self.anchor_decay

            # Track best candidate: minimize WL at acceptable overflow
            score = float(wl.item()) * (1.0 + max(ov - self.target_overflow, 0.0) * 5.0)
            if score < best_score:
                best_score = score
                best_overflow = ov
                best_pos = pos.detach().clone()

            if ov < self.target_overflow and it > 100:
                break
            prev_ov = ov

        pos = best_pos

        # Legalization (hard macros only)
        pos_np = pos.detach().cpu().numpy().astype(np.float64)
        anchor_np = pos_np[:n_hard].copy()  # min-displacement from GP result
        sizes_np = sizes.detach().cpu().numpy().astype(np.float64)
        mov_np = (~fixed).cpu().numpy()
        hw_np = sizes_np[:, 0] / 2
        hh_np = sizes_np[:, 1] / 2

        legal = _legalize(
            pos_np[:n_hard], anchor_np, mov_np[:n_hard], sizes_np[:n_hard],
            hw_np[:n_hard], hh_np[:n_hard], cw, ch, n_hard,
        )

        # NOTE: a pure-WL detailed-placement pass was tried here; it regressed
        # density (0.68 → 0.80) because local HPWL-minimizing shifts have no
        # density awareness and macros cluster across passes. Keeping the
        # legalized positions directly.

        # Build final position tensor on device, with hard macros pinned to legal
        final_pos = pos.detach().clone()
        final_pos[:n_hard] = torch.tensor(legal, dtype=torch.float32, device=device)

        # ── Soft-macro-only refinement: hard macros are frozen, soft follow WL + density.
        if n_all > n_hard and n_nets > 0:
            final_pos = self._refine_soft(
                final_pos, anchor0, sizes, ports, pin_node, pin_offset, pin_net,
                n_nets, net_weight, movable, n_all, n_hard, lb, ub, cw, ch,
                nx, ny, target_density, diag, gamma_end, lam,
            )

        result = benchmark.macro_positions.clone()
        result[:] = final_pos.cpu()
        # Ensure hard macros are exactly at the legalized integer-safe positions
        result[:n_hard] = torch.tensor(legal, dtype=torch.float32)
        return result

    def _refine_soft(self, pos, anchor0, sizes, ports, pin_node, pin_offset, pin_net,
                     n_nets, net_weight, movable, n_all, n_hard, lb, ub, cw, ch,
                     nx, ny, target_density, diag, gamma, lam_gp):
        """
        Refine soft macros only, hard macros are pinned.
        Keep density penalty so soft macros don't collapse, but track best by WL
        subject to the overflow staying near its initial level.
        """
        iters = 120
        soft_mov = movable.clone()
        soft_mov[:n_hard] = 0.0

        pos = pos.detach().clone()
        # Rebalance λ for soft refinement: don't use the inflated end-of-GP λ
        # (it would over-weight density and freeze improvement).  Instead use
        # the initial-balance λ scaled by a small factor.
        y_req = pos.clone().requires_grad_(True)
        pp = _gather_pin_positions(y_req, ports, pin_node, pin_offset, n_all)
        wl_only = _lse_wirelength(pp, pin_net, n_nets, net_weight, gamma)
        (wl_g,) = torch.autograd.grad(wl_only, y_req)

        y_req = pos.clone().requires_grad_(True)
        ed_only, rho0 = _density_energy(y_req, sizes, cw, ch, nx, ny, target_density)
        (ed_g,) = torch.autograd.grad(ed_only, y_req)
        mov_mask = soft_mov.squeeze(1) > 0
        if mov_mask.sum().item() == 0:
            return pos
        wl_norm = wl_g[mov_mask].norm().item() + 1e-12
        ed_norm = ed_g[mov_mask].norm().item() + 1e-12
        lam = 0.2 * (wl_norm / ed_norm)  # softer than GP

        base_overflow = float((torch.clamp(rho0 - target_density, min=0.0).sum() * (cw/nx) * (ch/ny)).item())

        y = pos.clone()
        a = 1.0
        best = pos.clone()
        best_wl = float("inf")
        best_overflow = float("inf")
        anchor_w = 0.001

        for it in range(iters):
            y_req = y.detach().clone().requires_grad_(True)
            pin_pos = _gather_pin_positions(y_req, ports, pin_node, pin_offset, n_all)
            wl = _lse_wirelength(pin_pos, pin_net, n_nets, net_weight, gamma)
            ed, rho = _density_energy(y_req, sizes, cw, ch, nx, ny, target_density)
            disp = y_req - anchor0
            aloss = (disp * disp * soft_mov).sum() * anchor_w
            loss = wl + lam * ed + aloss
            (grad,) = torch.autograd.grad(loss, y_req)
            grad = grad * soft_mov
            gnorm = grad.norm().item() + 1e-12
            step = 0.015 * diag / gnorm
            pos_new = y - step * grad
            pos_new = torch.min(torch.max(pos_new, lb), ub)
            a_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * a * a))
            beta = (a - 1.0) / a_new
            y = pos_new + beta * (pos_new - pos)
            y = torch.min(torch.max(y, lb), ub)
            pos = pos_new
            a = a_new

            with torch.no_grad():
                ov_mass = float((torch.clamp(rho - target_density, min=0.0).sum() * (cw/nx) * (ch/ny)).item())

            # Track best: prefer lower WL provided overflow doesn't grow significantly
            cur_wl = float(wl.item())
            if ov_mass <= base_overflow * 1.15 and cur_wl < best_wl:
                best_wl = cur_wl
                best_overflow = ov_mass
                best = pos.detach().clone()
        return best

    @staticmethod
    def _pick_grid(cw, ch, n_target, axis):
        """Pick grid size so bins are roughly square and total count ≈ n_target^2."""
        if axis == "x":
            return max(32, int(round(n_target * math.sqrt(cw / max(ch, 1e-9)))))
        else:
            return max(32, int(round(n_target * math.sqrt(ch / max(cw, 1e-9)))))
