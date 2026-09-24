"""Stage 5: journal figures.

Greyscale-readable, 300 dpi, confidence bands shown, axis units labelled.
Outputs PNG (for Word) and PDF (for typesetting) into results/figures/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

FIG = os.path.join(C.RESULTS, "figures")
R = C.RESULTS

plt.rcParams.update({
    # Empirical Economics requires sans-serif figure lettering (Helvetica or
    # Arial) at roughly 8-12 pt, sized consistently across the artwork.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "figure.dpi": 300,
})

GREY = "0.45"
LGREY = "0.80"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def fig1_irf():
    d = pd.read_csv(os.path.join(R, "table2_baseline_irf.csv")).sort_values("h")
    
    # Added constrained_layout=True to automatically handle padding and prevent cut-offs
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True, constrained_layout=True)
    
    panels = [("food", "food_se", "(a) Food CPI"),
              ("rel", "rel_se", "(b) Relative food price"),
              ("head", "head_se", "(c) Headline CPI")]
              
    for ax, (b, s, title) in zip(axes, panels):
        lo90, hi90 = d[b] - 1.645 * d[s], d[b] + 1.645 * d[s]
        lo95, hi95 = d[b] - 1.960 * d[s], d[b] + 1.960 * d[s]
        ax.fill_between(d["h"], lo95, hi95, color=LGREY, alpha=0.65, lw=0)
        ax.fill_between(d["h"], lo90, hi90, color=GREY, alpha=0.45, lw=0)
        ax.plot(d["h"], d[b], color="black", lw=1.3)
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.axvline(0, color="black", lw=0.6, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([-12, -6, 0, 6, 12, 18, 24])
        # Removed individual ax.set_xlabel() from the loop
        
    axes[0].set_ylabel("cumulative response (pp)")
    
    # Set a single, centered x-axis label for the entire figure
    fig.supxlabel("months since the growing-season anomaly", fontsize=10)
    
    save(fig, "fig1_baseline_irf")


def fig2_shape():
    b = pd.read_csv(os.path.join(R, "table3c_bin_response.csv"))
    b = b[b["h"] == 6].copy()
    order = ["S <= -1.5  (severe cold)", "-1.5 < S <= -0.5  (mild cold)",
             "0.5 < S <= 1.5  (mild heat)", "S > 1.5  (severe heat)"]
    b["ord"] = b["bin"].apply(lambda x: order.index(x) if x in order else 99)
    b = b.sort_values("ord")
    xs = [-2.0, -1.0, 1.0, 2.0]
    lab = ["severe\ncold", "mild\ncold", "mild\nheat", "severe\nheat"]

    q = pd.read_csv(os.path.join(R, "table3a_linear_vs_quadratic.csv"))
    q6 = q[q["h"] == 6].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))

    ax = axes[0]
    ax.errorbar(xs, b["b"], yerr=1.96 * b["se"], fmt="s", color="black",
                ms=4.5, lw=1.0, capsize=3)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(lab + [""][:0] if False else ["severe\ncold", "mild\ncold",
                                                     "normal\n(reference)", "mild\nheat",
                                                     "severe\nheat"])
    ax.plot([0], [0], "o", color="white", mec="black", ms=5, zorder=5)
    ax.set_ylabel("effect vs a normal season (pp)")
    ax.set_title("(a) Nonparametric bin response, h = 6", fontsize=9)

    ax = axes[1]
    s = np.linspace(-3, 3, 200)
    fit = q6["lin"] * s + q6["quad"] * s ** 2
    se_band = 1.96 * (q6["lin_se"] * np.abs(s) + q6["quad_se"] * s ** 2)
    ax.fill_between(s, fit - se_band, fit + se_band, color=LGREY, alpha=0.7, lw=0)
    ax.plot(s, fit, color="black", lw=1.4, label="quadratic fit")
    ax.plot(s, q6["lin"] * s, color="black", lw=1.1, ls="--", label="linear term only")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("signed anomaly $S$ (sd from local norm)")
    ax.set_ylabel("cumulative food-CPI response (pp)")
    ax.set_title("(b) Fitted shape, h = 6", fontsize=9)
    ax.legend(loc="upper center", fontsize=8)
    axes[0].set_xlabel("growing-season outcome")
    save(fig, "fig2_shape")


def fig3_convexity_robustness():
    t = pd.read_csv(os.path.join(R, "table15_convexity_robustness.csv"))
    t = t.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(t["quad"], y + 0.16, xerr=1.96 * t["quad_se"], fmt="s",
                color="black", ms=4, lw=1.0, capsize=2.5, label="quadratic, $S^2$")
    ax.errorbar(t["lin"], y - 0.16, xerr=1.96 * t["lin_se"], fmt="o",
                color=GREY, ms=4, lw=1.0, capsize=2.5, mfc="white",
                label="linear, $S$")
    ax.axvline(0, color="black", lw=0.7, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([s.strip() for s in t["spec"]], fontsize=8)
    ax.set_xlabel("coefficient on the cumulative food-CPI response at h = 6 (pp)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(-0.7, len(t) - 0.3)
    save(fig, "fig3_convexity_robustness")


def fig4_credit():
    o = pd.read_csv(os.path.join(R, "table5_credit_heterogeneity.csv"))
    m = pd.read_csv(os.path.join(R, "table7_dml.csv"))
    pct = [10, 25, 50, 75, 90]
    cred = [11, 21, 42, 74, 121]
    row = o[o["h"] == 12].iloc[0]
    eff = [row[f"eff_p{p}"] for p in pct]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    ax = axes[0]
    ax.plot(cred, eff, "-o", color="black", ms=4.5, lw=1.2)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(cred)
    ax.set_xticklabels([str(c) for c in cred])
    ax.set_xlabel("private credit / GDP (%), sample percentiles")
    ax.set_ylabel("effect of a 1 sd anomaly at h = 12 (pp)")
    ax.set_title("(a) Local projections", fontsize=9)

    ax = axes[1]
    ax.errorbar(m["h"], m["theta2"], yerr=1.96 * m["theta2_se"], fmt="s",
                color="black", ms=4.5, lw=1.0, capsize=3)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("horizon h (months)")
    ax.set_ylabel(r"$\theta_2$: $|S| \times$ z(log credit)")
    ax.set_title("(b) Double machine learning", fontsize=9)
    save(fig, "fig4_credit_gradient")


def main():
    C.ascii_stdout()
    os.makedirs(FIG, exist_ok=True)
    print("building figures")
    fig1_irf()
    fig2_shape()
    fig3_convexity_robustness()
    fig4_credit()
    print("done ->", FIG)


if __name__ == "__main__":
    main()
