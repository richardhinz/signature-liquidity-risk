import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from itertools import product as iproduct
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# 1. PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Params:
    sigma:   float = 0.774    # calibrated volatility
    mu:      float = 0.0      # drift (risk-neutral)
    T:       float = 1.0      # horizon (years)
    n_steps: int   = 252      # daily steps
    alpha:   float = 0.975    # VaR confidence level (Basel FRTB)
    n_paths: int   = 2000     # Monte Carlo paths
    N_sig:   int   = 2        # max signature level
    seed:    int   = 2026

p = Params()
rng = np.random.default_rng(p.seed)
dt = p.T / p.n_steps
t_grid = np.linspace(0, p.T, p.n_steps + 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PATH GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def gbm_paths(n, endpoint_override=None):
    """
    Generate n paths of geometric Brownian motion.
    Returns log-price increments shape (n, n_steps+1), normalised to start at 0.

    Plain language: standard random walk — prices move up and down symmetrically.
    Each step is independent of all previous steps (Markov property holds).
    """
    Z = rng.standard_normal((n, p.n_steps))
    log_price = np.zeros((n, p.n_steps + 1))
    for i in range(p.n_steps):
        log_price[:, i+1] = log_price[:, i] + \
            (p.mu - 0.5*p.sigma**2)*dt + p.sigma*np.sqrt(dt)*Z[:, i]
    # Normalise to 1-variation = 1 for clean signature computation
    return log_price


def linear_crash_path(n, final_drop=-0.30):
    """
    Generate n identical linear crash paths: price drops at constant rate.
    Plain language: slow, steady decline — like a gradual bear market.
    Lévy area should be zero (path IS the chord).
    """
    path = np.linspace(0, final_drop, p.n_steps + 1)
    return np.tile(path, (n, 1))


def quadratic_crash_path(n, final_drop=-0.30):
    """
    Generate n identical quadratic crash paths: price accelerates downward.
    Plain language: fast crash — most of the fall happens in the first third.
    Lévy area should be large (path curves far below the chord baseline).
    """
    # P(t) = final_drop * (t/T)^2 — quadratic acceleration
    path = final_drop * (t_grid / p.T)**2
    return np.tile(path, (n, 1))


def mixed_regime_paths(n_normal, n_slow, n_fast, final_drop=-0.30):
    """
    Generate a mixed dataset of all three regimes.
    Labels: 0 = normal, 1 = slow crash, 2 = fast crash.
    """
    normal = gbm_paths(n_normal)
    slow   = linear_crash_path(n_slow, final_drop)
    fast   = quadratic_crash_path(n_fast, final_drop)
    paths  = np.vstack([normal, slow, fast])
    labels = np.array([0]*n_normal + [1]*n_slow + [2]*n_fast)
    return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIGNATURE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_signature_level1(path):
    """
    Level-1 signature of the time-augmented path (t, P(t)).
    Components: (integral of dt, integral of dP) = (T, P(T) - P(0))

    Plain language: Level-1 just tells you where the path ended up.
    It knows nothing about how it got there.
    """
    n = path.shape[0]
    # Time component: integral of dt = T (same for all paths)
    sig_t = np.full(n, p.T)
    # Price component: total displacement
    sig_p = path[:, -1] - path[:, 0]
    return np.column_stack([sig_t, sig_p])


def compute_levy_area(path):
    """
    Lévy area of the time-augmented path (t, P(t)):
        A = (1/2) * integral[ t * dP - P * dt ]
          = (1/2) * [integral(t dP) - integral(P dt)]

    Plain language: the Lévy area measures the AREA enclosed between the
    actual price path and the straight line from start to finish.
    - Zero for a straight-line (constant velocity) path
    - Large and negative for a path that crashes fast early on
    - Fluctuates around zero for random Brownian motion

    We compute this by numerical integration (trapezoidal rule).
    """
    n = path.shape[0]
    levy = np.zeros(n)

    for i in range(n):
        P = path[i]
        t = t_grid

        # integral of t * dP: sum over steps t_k * (P_{k+1} - P_k)
        dP = np.diff(P)
        t_mid = t[:-1]   # left endpoint for Riemann sum
        int_t_dP = np.sum(t_mid * dP)

        # integral of P * dt: trapezoidal rule
        int_P_dt = np.trapezoid(P, t)

        levy[i] = 0.5 * (int_t_dP - int_P_dt)

    return levy


def compute_signature_level2(path):
    n = path.shape[0]
    sig_11 = np.zeros(n)
    sig_12 = np.zeros(n)
    sig_21 = np.zeros(n)
    sig_22 = np.zeros(n)

    for i in range(n):
        P = path[i]
        t = t_grid
        dP = np.diff(P)
        dt_arr = np.diff(t)

        # Running integrals (forward Riemann sums)
        int_t = np.cumsum(np.concatenate([[0], dt_arr]))    # integral of dt up to each step
        int_P = np.cumsum(np.concatenate([[0], dP]))         # integral of dP up to each step

        # Sig^(1,1): integral of [integral of dt] * dt
        sig_11[i] = np.sum(int_t[:-1] * dt_arr)

        # Sig^(1,2): integral of [integral of dt] * dP
        sig_12[i] = np.sum(int_t[:-1] * dP)

        # Sig^(2,1): integral of [integral of dP] * dt
        sig_21[i] = np.sum(int_P[:-1] * dt_arr)

        # Sig^(2,2): integral of [integral of dP] * dP
        sig_22[i] = np.sum(int_P[:-1] * dP)

    return np.column_stack([sig_11, sig_12, sig_21, sig_22])


def compute_full_signature(path, level=2):
    sig1 = compute_signature_level1(path)
    if level == 1:
        return sig1
    sig2 = compute_signature_level2(path)
    return np.column_stack([sig1, sig2])


# ─────────────────────────────────────────────────────────────────────────────
# 4. LIQUIDITY-ADJUSTED LOSS FUNCTIONAL
# ─────────────────────────────────────────────────────────────────────────────

def liquidity_adjusted_loss(path, kappa=0.05, eta=0.80):
    n = path.shape[0]
    phi_0 = 0.01   # baseline spread

    losses = np.zeros(n)
    for i in range(n):
        P = path[i]
        dP = np.diff(P)
        P_mid = P[:-1]

        # Level effect: |P(t) - P(0)|
        level_effect = np.abs(P_mid - P[0])

        # Velocity effect: |dP/dt|
        velocity_effect = np.abs(dP) / dt

        # Spread at each step
        phi_t = phi_0 + kappa * level_effect + eta * velocity_effect

        # Loss: phi_t * |dP_t| integrated
        losses[i] = np.sum(phi_t * np.abs(dP))

    return losses


# ─────────────────────────────────────────────────────────────────────────────
# 5. VAR COMPUTATION: MARKOVIAN vs SIGNATURE
# ─────────────────────────────────────────────────────────────────────────────

def markovian_var(losses, alpha=0.975):
    return np.quantile(losses, alpha)


def signature_var(path, losses, level=2, alpha=0.975, n_train=None):
    n = path.shape[0]
    if n_train is None:
        n_train = n // 2

    # Compute signature features
    features = compute_full_signature(path, level=level)

    # Standardise
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[:n_train])
    X_test  = scaler.transform(features[n_train:])
    y_train = losses[:n_train]

    # Elastic Net with cross-validation
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=5, max_iter=5000, random_state=42
    )
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    y_true = losses[n_train:]

    # VaR from predicted losses
    sig_var = np.quantile(y_pred, alpha)
    true_var = np.quantile(y_true, alpha)
    markov_var = markovian_var(y_true, alpha)

    return {
        "sig_var":    sig_var,
        "markov_var": markov_var,
        "true_var":   true_var,
        "model":      model,
        "r2":         model.score(X_test, y_true),
        "l1_ratio":   model.l1_ratio_,
        "n_features": np.sum(model.coef_ != 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation():
    print("=" * 65)
    print("  EMPIRICAL SIMULATION — PAPER I")
    print("  Non-Markovian Liquidity Risk via Rough Path Signatures")
    print(f"  sigma={p.sigma}, T={p.T}, n_steps={p.n_steps}, n_paths={p.n_paths}")
    print("=" * 65)

    # ── A. Three representative paths ─────────────────────────────────────
    print("\n[1/4] Computing Lévy area for representative paths...")

    single_normal = gbm_paths(1)
    single_slow   = linear_crash_path(1, final_drop=-0.30)
    single_fast   = quadratic_crash_path(1, final_drop=-0.30)

    levy_normal = compute_levy_area(single_normal)[0]
    levy_slow   = compute_levy_area(single_slow)[0]
    levy_fast   = compute_levy_area(single_fast)[0]

    print(f"\n  Lévy Area Results (same start and end point for crash paths):")
    print(f"  Normal (GBM):        A = {levy_normal:+.6f}  (fluctuates around 0)")
    print(f"  Slow crash (linear): A = {levy_slow:+.6f}  (exactly 0 — path IS the chord)")
    print(f"  Fast crash (quad.):  A = {levy_fast:+.6f}  (large negative — path curves below)")

    # ── B. Distribution of Lévy area across regimes ───────────────────────
    print("\n[2/4] Simulating Lévy area distribution across 500 paths per regime...")
    n_each = 500

    normal_paths = gbm_paths(n_each)
    slow_paths   = linear_crash_path(n_each) + rng.normal(0, 0.01, (n_each, p.n_steps+1))
    fast_paths   = quadratic_crash_path(n_each) + rng.normal(0, 0.01, (n_each, p.n_steps+1))

    levy_normals = compute_levy_area(normal_paths)
    levy_slows   = compute_levy_area(slow_paths)
    levy_fasts   = compute_levy_area(fast_paths)

    print(f"  Normal BM:    mean = {levy_normals.mean():+.5f}, std = {levy_normals.std():.5f}")
    print(f"  Slow crash:   mean = {levy_slows.mean():+.5f}, std = {levy_slows.std():.5f}")
    print(f"  Fast crash:   mean = {levy_fasts.mean():+.5f}, std = {levy_fasts.std():.5f}")

    expected_var = p.sigma**2 * p.T**3 / 12
    print(f"\n  Theoretical Var[A] under BM: sigma^2*T^3/12 = {expected_var:.6f}")
    print(f"  Empirical Var[A] under BM:                  = {levy_normals.var():.6f}")
    print(f"  Ratio (should be ~1.0): {levy_normals.var()/expected_var:.4f}")

    # ── C. Signature VaR vs Markovian VaR ────────────────────────────────
    print("\n[3/4] Computing Signature VaR vs Markovian VaR...")

    results_table = []

    for regime, path_fn, label in [
        ("Normal (GBM)",        lambda: gbm_paths(p.n_paths),                          "normal"),
        ("Slow crash + noise",  lambda: linear_crash_path(p.n_paths) +
                                        rng.normal(0, 0.02, (p.n_paths, p.n_steps+1)), "slow"),
        ("Fast crash + noise",  lambda: quadratic_crash_path(p.n_paths) +
                                        rng.normal(0, 0.02, (p.n_paths, p.n_steps+1)), "fast"),
    ]:
        paths  = path_fn()
        losses = liquidity_adjusted_loss(paths)

        res_l1 = signature_var(paths, losses, level=1, alpha=p.alpha)
        res_l2 = signature_var(paths, losses, level=2, alpha=p.alpha)

        results_table.append({
            "regime":         regime,
            "true_var":       res_l2["true_var"],
            "markov_var":     res_l2["markov_var"],
            "sig_var_l1":     res_l1["sig_var"],
            "sig_var_l2":     res_l2["sig_var"],
            "r2_l1":          res_l1["r2"],
            "r2_l2":          res_l2["r2"],
            "capital_gap_l1": res_l1["true_var"] - res_l2["markov_var"],
            "capital_gap_l2": res_l2["true_var"] - res_l2["markov_var"],
            "active_features": res_l2["n_features"],
        })

    print(f"\n  {'Regime':<25} {'True VaR':>10} {'Markov VaR':>12} "
          f"{'Sig VaR L1':>12} {'Sig VaR L2':>12} "
          f"{'R² (L2)':>10} {'Cap. Gap':>10}")
    print("  " + "-"*93)
    for r in results_table:
        print(f"  {r['regime']:<25} {r['true_var']:>10.4f} {r['markov_var']:>12.4f} "
              f"{r['sig_var_l1']:>12.4f} {r['sig_var_l2']:>12.4f} "
              f"{r['r2_l2']:>10.4f} {r['capital_gap_l2']:>+10.4f}")

    return {
        "single_paths":  (single_normal, single_slow, single_fast),
        "levy_vals":     (levy_normal, levy_slow, levy_fast),
        "levy_dists":    (levy_normals, levy_slows, levy_fasts),
        "results_table": results_table,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. PUBLICATION-QUALITY FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(data):
    print("\n[4/4] Generating figures...")
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.35)

    single_normal, single_slow, single_fast = data["single_paths"]
    levy_normal, levy_slow, levy_fast       = data["levy_vals"]
    levy_normals, levy_slows, levy_fasts    = data["levy_dists"]
    results                                 = data["results_table"]

    # ── Panel 1: Three representative paths ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_grid, single_normal[0],  color="#888888", lw=1.5,
             label=f"Normal BM (A={levy_normal:+.4f})")
    ax1.plot(t_grid, single_slow[0],    color="#003870", lw=2.0, ls="--",
             label=f"Slow crash (A={levy_slow:+.4f})")
    ax1.plot(t_grid, single_fast[0],    color="#E05A1B", lw=2.0,
             label=f"Fast crash (A={levy_fast:+.4f})")

    # Draw chord (baseline) for crash paths
    chord = np.linspace(0, single_slow[0,-1], p.n_steps+1)
    ax1.plot(t_grid, chord, color="black", lw=0.8, ls=":", alpha=0.6,
             label="Chord (baseline)")
    # Shade area between fast crash and chord
    ax1.fill_between(t_grid, single_fast[0], chord,
                     where=single_fast[0] < chord,
                     alpha=0.12, color="#E05A1B", label="Lévy area (fast crash)")

    ax1.set_xlabel("Time $t$ (years)", fontsize=9)
    ax1.set_ylabel("Log-price $P(t)$", fontsize=9)
    ax1.set_title("Three Path Types: Same Endpoint, Different Shape\n"
                  "Lévy area detects curvature, not direction",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.set_facecolor("#F8F8F8")
    ax1.grid(True, alpha=0.4, lw=0.5)

    # ── Panel 2: Lévy area distributions ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(-0.035, 0.025, 60)
    ax2.hist(levy_normals, bins=bins, color="#888888", alpha=0.6,
             density=True, label=f"Normal BM  (mean={levy_normals.mean():+.4f})")
    ax2.hist(levy_fasts,   bins=bins, color="#E05A1B", alpha=0.6,
             density=True, label=f"Fast crash (mean={levy_fasts.mean():+.4f})")
    ax2.hist(levy_slows,   bins=bins, color="#003870", alpha=0.6,
             density=True, label=f"Slow crash (mean={levy_slows.mean():+.4f})")

    # Theoretical BM distribution: N(0, sigma^2 T^3/12)
    levy_var = p.sigma**2 * p.T**3 / 12
    x_th = np.linspace(-0.035, 0.025, 200)
    ax2.plot(x_th, norm.pdf(x_th, 0, np.sqrt(levy_var)),
             color="black", lw=1.5, ls="--", label=r"Theory: $N(0,\sigma^2T^3/12)$")

    ax2.axvline(0, color="black", lw=0.8, ls="-")
    ax2.set_xlabel(r"Lévy Area $\mathbb{A}(X)_{0,T}$", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("Distribution of Lévy Area by Regime\n"
                  "Fast crash systematically shifts left",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7.5)
    ax2.set_facecolor("#F8F8F8")
    ax2.grid(True, alpha=0.4, lw=0.5)

    # ── Panel 3: VaR comparison bar chart ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    regimes = [r["regime"].replace(" + noise","").replace(" (GBM)","") for r in results]
    x       = np.arange(len(regimes))
    w       = 0.22

    bars_true   = ax3.bar(x - w,    [r["true_var"]    for r in results],
                          w, label="True VaR",          color="#222222", alpha=0.85)
    bars_markov = ax3.bar(x,         [r["markov_var"]  for r in results],
                          w, label="Markovian VaR",     color="#888888", alpha=0.85)
    bars_sigl1  = ax3.bar(x + w,     [r["sig_var_l1"]  for r in results],
                          w, label="Sig. VaR (Level 1)", color="#1F5C99", alpha=0.85)
    bars_sigl2  = ax3.bar(x + 2*w,   [r["sig_var_l2"]  for r in results],
                          w, label="Sig. VaR (Level 2)", color="#003870", alpha=0.85)

    ax3.set_xticks(x + w/2)
    ax3.set_xticklabels(regimes, fontsize=9)
    ax3.set_ylabel("VaR at 97.5% (Basel FRTB)", fontsize=9)
    ax3.set_title("Markovian vs Signature VaR by Regime\n"
                  "Level-2 signature closes the capital gap",
                  fontsize=10, fontweight="bold")
    ax3.legend(fontsize=7.5)
    ax3.set_facecolor("#F8F8F8")
    ax3.grid(True, axis="y", alpha=0.4, lw=0.5)

    # ── Panel 4: Capital gap (True VaR minus Markovian VaR) ──────────────
    ax4 = fig.add_subplot(gs[1, 1])
    gaps_markov = [r["capital_gap_l2"]                         for r in results]
    gaps_sig    = [r["true_var"] - r["sig_var_l2"]             for r in results]
    r2_vals     = [r["r2_l2"]                                   for r in results]

    colors_gap = ["#003870" if g > 0 else "#E05A1B" for g in gaps_markov]
    bars_gap = ax4.bar(x - w/2, gaps_markov, w*2,
                       color=colors_gap, alpha=0.85, label="Capital gap: True - Markovian")
    bars_sig_gap = ax4.bar(x + w*1.5, gaps_sig, w*2,
                           color="#888888", alpha=0.60, label="Residual gap: True - Sig L2")

    # R² annotation
    for i, r2 in enumerate(r2_vals):
        ax4.text(x[i] - w/2, max(gaps_markov[i], 0) + 0.001,
                 f"$R^2$={r2:.2f}", ha="center", va="bottom", fontsize=8)

    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_xticks(x + w/2)
    ax4.set_xticklabels(regimes, fontsize=9)
    ax4.set_ylabel("VaR Gap (percentage points)", fontsize=9)
    ax4.set_title("Capital Gap: True VaR minus Markovian VaR\n"
                  "Signature Level-2 closes most of the gap",
                  fontsize=10, fontweight="bold")
    ax4.legend(fontsize=7.5)
    ax4.set_facecolor("#F8F8F8")
    ax4.grid(True, axis="y", alpha=0.4, lw=0.5)

    fig.suptitle(
        "Empirical Simulation — Non-Markovian Liquidity Risk via Rough Path Signatures\n"
        f"Parameters: sigma={p.sigma}, T={p.T}yr, alpha={p.alpha} (Basel FRTB), "
        f"n={p.n_paths} paths  |  Richard Elvis Hinz, April 2026",
        fontsize=11, fontweight="bold", y=0.99
    )
    plt.savefig("/home/claude/empirical_simulation.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    print("  Figure saved: empirical_simulation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = run_simulation()
    plot_results(data)
    print("\nSimulation complete.")
