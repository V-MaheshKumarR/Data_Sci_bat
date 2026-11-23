"""Generates the chart images embedded in Statistics_Beginner_Slides.html.

Run:  python make_slide_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "assets"
OUT.mkdir(exist_ok=True)
CSV = BASE / "data" / "indian_engineering_student_placement.csv"

BLUE, RED, GREEN, ORANGE = "#1f6feb", "#d1495b", "#2e9e5b", "#e8952f"
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

rng = np.random.default_rng(42)
df = pd.read_csv(CSV)


def save(fig, name):
    fig.savefig(OUT / name, transparent=False, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. mean/median/mode
def chart_central_tendency():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))

    sym = rng.normal(70, 10, 8000)
    peak = np.round(sym).astype(int)
    mode_val = np.bincount(peak - peak.min()).argmax() + peak.min()
    ax[0].hist(sym, bins=45, color="#cfe0ff", edgecolor="white")
    for v, c, lbl in [(sym.mean(), RED, "mean"),
                      (np.median(sym), GREEN, "median"),
                      (mode_val, ORANGE, "mode")]:
        ax[0].axvline(v, color=c, lw=2.5, ls="--", label=f"{lbl} = {v:.1f}")
    ax[0].set_title("Symmetric data (exam scores): all three agree")
    ax[0].set_xlabel("marks")
    ax[0].legend(fontsize=9)

    salary = rng.lognormal(mean=np.log(6), sigma=0.85, size=8000)
    ax[1].hist(salary, bins=70, range=(0, 40), color="#ffd9df", edgecolor="white")
    for v, c, lbl in [(salary.mean(), RED, "mean"),
                      (np.median(salary), GREEN, "median")]:
        ax[1].axvline(v, color=c, lw=2.5, ls="--", label=f"{lbl} = {v:.1f}")
    ax[1].annotate("the long tail\nlives out here", xy=(24, 30), xytext=(24, 350),
                   fontsize=9, color=RED, ha="center",
                   arrowprops=dict(arrowstyle="->", color=RED))
    ax[1].set_title("Skewed data (salary): the tail drags the mean")
    ax[1].set_xlabel("salary (LPA)")
    ax[1].legend(fontsize=9)
    save(fig, "01_central_tendency.png")


# ---------------------------------------------------------------- 2. same mean, different spread
def chart_spread():
    a = np.array([7.0, 7.0, 7.0, 7.0, 7.0])
    b = np.array([4.0, 4.0, 7.0, 10.0, 10.0])
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.scatter(a, [2] * 5, s=320, color=GREEN, zorder=3,
               label=f"Class A   std = {a.std(ddof=1):.2f}")
    ax.scatter(b, [1] * 5, s=320, color=RED, zorder=3,
               label=f"Class B   std = {b.std(ddof=1):.2f}")
    ax.axvline(7.0, color="black", ls="--", lw=2, label="mean = 7.0 for BOTH")
    ax.set_yticks([1, 2], ["Class B", "Class A"])
    ax.set_xlim(3, 11)
    ax.set_xlabel("CGPA")
    ax.set_title("Same mean, completely different reality")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "02_spread.png")


# ---------------------------------------------------------------- 3. boxplot anatomy
def chart_boxplot_anatomy():
    data = np.concatenate([rng.normal(50, 8, 300), [12, 15, 88, 92]])
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.boxplot(data, orientation="horizontal", widths=0.45, patch_artist=True,
               boxprops=dict(facecolor="#cfe0ff", edgecolor=BLUE, lw=2),
               medianprops=dict(color=RED, lw=3),
               whiskerprops=dict(color=BLUE, lw=2),
               capprops=dict(color=BLUE, lw=2),
               flierprops=dict(marker="o", markerfacecolor=RED, markersize=8, alpha=.8))
    for v, lbl, c in [(lo, f"Q1-1.5·IQR\n{lo:.0f}", "gray"), (q1, f"Q1\n{q1:.0f}", BLUE),
                      (med, f"median\n{med:.0f}", RED), (q3, f"Q3\n{q3:.0f}", BLUE),
                      (hi, f"Q3+1.5·IQR\n{hi:.0f}", "gray")]:
        ax.annotate(lbl, xy=(v, 1.28), ha="center", fontsize=9, color=c, weight="bold")
    ax.annotate("", xy=(q1, 0.72), xytext=(q3, 0.72),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2))
    ax.text((q1 + q3) / 2, 0.64, f"IQR = {iqr:.0f}  (middle 50%)",
            ha="center", color=GREEN, weight="bold", fontsize=10)
    ax.text(hi + 6, 1.0, "outliers", color=RED, fontsize=10, weight="bold")
    ax.set_yticks([])
    ax.set_ylim(0.55, 1.5)
    ax.set_title("Anatomy of a boxplot — the 1.5 × IQR outlier rule")
    save(fig, "03_boxplot_anatomy.png")


# ---------------------------------------------------------------- 4. skew shapes
def chart_skew():
    sym = rng.normal(50, 10, 20000)
    right = rng.exponential(10, 20000)
    left = 100 - rng.exponential(10, 20000)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, d, t in zip(axes, [sym, right, left],
                        ["Symmetric\nmean ≈ median ≈ mode",
                         "Right / positive skew\nmean > median > mode",
                         "Left / negative skew\nmean < median < mode"]):
        ax.hist(d, bins=60, color="#cfe0ff", edgecolor="none", density=True)
        ax.axvline(np.mean(d), color=RED, lw=2.5, ls="--", label="mean")
        ax.axvline(np.median(d), color=GREEN, lw=2.5, label="median")
        ax.set_title(f"{t}\nskew = {stats.skew(d):.2f}", fontsize=10)
        ax.set_yticks([])
        ax.legend(fontsize=8)
    save(fig, "04_skew.png")


# ---------------------------------------------------------------- 5. empirical rule
def chart_empirical_rule():
    x = np.linspace(-4, 4, 800)
    y = stats.norm.pdf(x)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, y, color="black", lw=2.5)
    for k, c, lbl in [(3, "#dbe9ff", "99.7%  within 3σ"),
                      (2, "#a8caff", "95%  within 2σ"),
                      (1, "#5b9bff", "68%  within 1σ")]:
        m = (x >= -k) & (x <= k)
        ax.fill_between(x[m], y[m], color=c, label=lbl)
    for k in (-3, -2, -1, 0, 1, 2, 3):
        ax.axvline(k, color="white", lw=1)
    ax.set_xticks(range(-3, 4),
                  ["μ-3σ", "μ-2σ", "μ-σ", "μ", "μ+σ", "μ+2σ", "μ+3σ"])
    ax.set_yticks([])
    ax.set_title("The Empirical Rule:  68 – 95 – 99.7")
    ax.legend(loc="upper right")
    save(fig, "05_empirical_rule.png")


# ---------------------------------------------------------------- 6. z-score
def chart_zscore():
    mu, sd = 70.0, 10.0
    x = np.linspace(-4, 4, 800)
    y = stats.norm.pdf(x)
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.plot(x, y, color="black", lw=2)
    ax.fill_between(x, y, color="#eef4ff")
    for z, c, off in [(-1.5, RED, -0.55), (0.0, "gray", 0.0), (2.0, GREEN, 0.55)]:
        raw = mu + z * sd
        pct = stats.norm.cdf(z) * 100
        ax.axvline(z, color=c, lw=2.5)
        ax.annotate(f"marks {raw:.0f}\nz = {z:+.1f}\n{pct:.0f}th pct",
                    xy=(z, stats.norm.pdf(z)), xytext=(z + off, 0.45),
                    ha="center", fontsize=10, color=c, weight="bold")
    ax.set_xticks(range(-3, 4))
    ax.set_xlabel("z  =  (x − μ) / σ   — standard deviations from the mean")
    ax.set_yticks([])
    ax.set_ylim(0, 0.56)
    ax.set_title(f"Z-scores put every student on one scale   (exam marks: μ={mu:.0f}, σ={sd:.0f})")
    save(fig, "06_zscore.png")


# ---------------------------------------------------------------- 7. sampling techniques
def chart_sampling():
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1))
    g = np.array([[(r, c) for c in range(8)] for r in range(6)]).reshape(-1, 2)
    x, y = g[:, 1], g[:, 0]
    groups = x // 2
    palette = ["#1f6feb", "#2e9e5b", "#e8952f", "#d1495b"]

    def base(ax, colors, picked, title):
        ax.scatter(x, y, s=110, c=colors, alpha=.25, edgecolors="none")
        ax.scatter(x[picked], y[picked], s=110, c=np.array(colors)[picked], edgecolors="black", lw=1.4)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.invert_yaxis()

    grey = ["#7d8da1"] * len(x)
    base(axes[0], grey, rng.choice(len(x), 12, replace=False), "Simple random\n(pure lottery)")

    cols = [palette[i] for i in groups]
    strat = np.concatenate([rng.choice(np.where(groups == k)[0], 3, replace=False) for k in range(4)])
    base(axes[1], cols, strat, "Stratified\n(3 from each group)")

    base(axes[2], grey, np.arange(2, len(x), 5), "Systematic\n(every 5th unit)")

    chosen = rng.choice(4, 2, replace=False)
    base(axes[3], cols, np.where(np.isin(groups, chosen))[0], "Cluster\n(2 whole groups)")
    save(fig, "07_sampling.png")


# ---------------------------------------------------------------- 8. CLT
def chart_clt():
    pop = rng.exponential(scale=5, size=200_000)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1))
    axes[0].hist(pop, bins=70, color=RED, edgecolor="none")
    axes[0].set_title(f"POPULATION (exponential)\nskew = {stats.skew(pop):.2f}", fontsize=10)
    axes[0].set_yticks([])
    for ax, n in zip(axes[1:], [2, 10, 50]):
        means = pop[rng.integers(0, len(pop), (4000, n))].mean(axis=1)
        ax.hist(means, bins=45, color=BLUE, edgecolor="none")
        ax.set_title(f"means of samples, n = {n}\nskew = {stats.skew(means):.2f}", fontsize=10)
        ax.set_yticks([])
    fig.suptitle("Central Limit Theorem: the distribution of sample MEANS becomes normal",
                 y=1.06, fontsize=12, weight="bold")
    save(fig, "08_clt.png")


# ---------------------------------------------------------------- 9. standard error
def chart_standard_error():
    sigma = df["cgpa"].std()
    ns = np.array([10, 25, 50, 100, 400, 1600])
    se = sigma / np.sqrt(ns)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(ns, se, "o-", color=BLUE, lw=2.5, ms=9)
    for n, s in zip(ns, se):
        ax.annotate(f"{s:.3f}", (n, s), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("sample size  n")
    ax.set_ylabel("SE = σ/√n")
    ax.set_title("The √n law: quadruple n to halve the error  (100 → 400 halves SE)")
    save(fig, "09_standard_error.png")


# ---------------------------------------------------------------- 10. rejection regions
def chart_rejection_regions():
    x = np.linspace(-4, 4, 800)
    y = stats.norm.pdf(x)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.2))
    setups = [("Two-tailed\nH₁: μ ≠ μ₀", [(-4, -1.96), (1.96, 4)], "±1.96"),
              ("Right-tailed\nH₁: μ > μ₀", [(1.645, 4)], "+1.645"),
              ("Left-tailed\nH₁: μ < μ₀", [(-4, -1.645)], "−1.645")]
    for ax, (title, regions, lbl) in zip(axes, setups):
        ax.plot(x, y, color="black", lw=2)
        ax.fill_between(x, y, color="#eef4ff")
        for a, b in regions:
            m = (x >= a) & (x <= b)
            ax.fill_between(x[m], y[m], color=RED, alpha=.7)
            ax.axvline(a if a > -4 else b, color=RED, ls="--", lw=1.5)
        ax.set_title(f"{title}\ncritical z = {lbl}   (α = 0.05)", fontsize=10)
        ax.set_yticks([])
        ax.set_xticks(range(-3, 4))
    fig.suptitle("Rejection regions — red area = 5% of the total area",
                 y=1.05, fontsize=12, weight="bold")
    save(fig, "10_rejection_regions.png")


# ---------------------------------------------------------------- 11. worked z-test
def chart_worked_ztest():
    mu0, sigma, n, xbar = 20.0, 1.5, 36, 19.4
    se = sigma / np.sqrt(n)
    z = (xbar - mu0) / se
    crit = stats.norm.ppf(0.05)
    p = stats.norm.cdf(z)

    x = np.linspace(-4, 4, 800)
    y = stats.norm.pdf(x)
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.plot(x, y, color="black", lw=2)
    ax.fill_between(x, y, color="#eef4ff")
    m = x <= crit
    ax.fill_between(x[m], y[m], color=RED, alpha=.65, label=f"rejection region (α=0.05), z < {crit:.2f}")
    m2 = x <= z
    ax.fill_between(x[m2], y[m2], color="#7a1020", alpha=.9, label=f"p-value = {p:.4f}")
    ax.axvline(z, color=GREEN, lw=3, label=f"our z = {z:.2f}")
    ax.annotate(f"z = {z:.2f} lands inside the\nrejection region → REJECT H₀",
                xy=(z, 0.02), xytext=(-3.9, 0.28), fontsize=10, weight="bold", color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    ax.set_yticks([])
    ax.set_xlabel("z")
    ax.set_title("Printer speed: x̄=19.4 ppm, μ₀=20 ppm, σ=1.5, n=36  →  z = −2.4, p = 0.0082")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "11_worked_ztest.png")


# ---------------------------------------------------------------- 12. type I / II errors
def chart_error_types():
    x = np.linspace(-4, 7, 900)
    h0 = stats.norm.pdf(x, 0, 1)
    h1 = stats.norm.pdf(x, 2.8, 1)
    crit = stats.norm.ppf(0.95)

    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.plot(x, h0, color=BLUE, lw=2.5, label="H₀ true:  μ = μ₀")
    ax.plot(x, h1, color=GREEN, lw=2.5, label="H₁ true:  μ = μ₀ + effect")
    m = x >= crit
    ax.fill_between(x[m], h0[m], color=RED, alpha=.75, label="Type I error (α) — false positive")
    m2 = x <= crit
    ax.fill_between(x[m2], h1[m2], color=ORANGE, alpha=.55, label="Type II error (β) — false negative")
    ax.axvline(crit, color="black", ls="--", lw=2)
    ax.text(crit + .05, 0.42, "critical value", fontsize=9)
    ax.set_yticks([])
    ax.set_title("Type I vs Type II error — lowering α shrinks red but grows orange")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "12_error_types.png")


# ---------------------------------------------------------------- 13. confidence interval
def chart_confidence_interval():
    mu, sigma, n = 7.0, 0.8, 100
    se = sigma / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    misses = 0
    for i in range(40):
        xbar = rng.normal(mu, se)
        lo, hi = xbar - 1.96 * se, xbar + 1.96 * se
        hit = lo <= mu <= hi
        misses += not hit
        ax.plot([lo, hi], [i, i], color=BLUE if hit else RED, lw=2.2)
        ax.plot(xbar, i, "o", color=BLUE if hit else RED, ms=4)
    ax.axvline(mu, color="black", ls="--", lw=2, label="true μ = 7.0")
    ax.set_yticks([])
    ax.set_xlabel("CGPA")
    ax.set_title(f"40 different samples, 40 different 95% CIs — {misses} missed the true μ")
    ax.legend()
    save(fig, "13_confidence_interval.png")


# ---------------------------------------------------------------- 14. real-data dashboard
def chart_dataset_overview():
    cols = ["cgpa", "study_hours_per_day", "attendance_percentage",
            "sleep_hours", "backlogs", "coding_skill_rating"]
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(df[c], bins=min(30, df[c].nunique()), color="#cfe0ff", edgecolor="white")
        ax.axvline(df[c].mean(), color=RED, ls="--", lw=2)
        ax.axvline(df[c].median(), color=GREEN, lw=2)
        ax.set_title(f"{c}\nskew = {df[c].skew():+.2f}", fontsize=10)
        ax.set_yticks([])
    fig.suptitle("Our dataset at a glance — red = mean, green = median",
                 y=1.0, fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "14_dataset_overview.png")


# ---------------------------------------------------------------- 15. population vs sample
def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def _box(ax, x, y, w, h, text, face="#eef4ff", edge=BLUE, fs=11, weight="normal"):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                         facecolor=face, edgecolor=edge, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)


def chart_population_vs_sample():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _blank(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)

    ax.add_patch(mpatches.Circle((2.5, 2.2), 1.9, facecolor="#eef4ff", edgecolor=BLUE, lw=2.5))
    pts = rng.uniform(-1, 1, (160, 2))
    pts = pts[(pts ** 2).sum(1) < 0.85]
    ax.scatter(2.5 + pts[:, 0] * 1.8, 2.2 + pts[:, 1] * 1.8, s=16, color=BLUE, alpha=.55)
    picked = rng.choice(len(pts), 14, replace=False)
    ax.scatter(2.5 + pts[picked, 0] * 1.8, 2.2 + pts[picked, 1] * 1.8,
               s=48, color=ORANGE, edgecolors="black", lw=.8, zorder=3)
    ax.text(2.5, 4.35, "POPULATION", ha="center", fontsize=13, weight="bold", color=BLUE)
    ax.text(2.5, 0.05, "all engineering students in India   (N)", ha="center", fontsize=10)

    ax.add_patch(mpatches.Circle((7.6, 2.2), 1.25, facecolor="#fff6e8", edgecolor=ORANGE, lw=2.5))
    sp = rng.uniform(-1, 1, (40, 2))
    sp = sp[(sp ** 2).sum(1) < 0.8][:14]
    ax.scatter(7.6 + sp[:, 0] * 1.15, 2.2 + sp[:, 1] * 1.15,
               s=48, color=ORANGE, edgecolors="black", lw=.8)
    ax.text(7.6, 4.35, "SAMPLE", ha="center", fontsize=13, weight="bold", color=ORANGE)
    ax.text(7.6, 0.05, "the 5,000 rows in our CSV   (n)", ha="center", fontsize=10)

    ax.annotate("", xy=(6.2, 2.9), xytext=(4.6, 2.9),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color=ORANGE))
    ax.text(5.4, 3.15, "sample", ha="center", fontsize=10, color=ORANGE, weight="bold")
    ax.annotate("", xy=(4.6, 1.4), xytext=(6.2, 1.4),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color=GREEN))
    ax.text(5.4, 1.05, "infer / estimate", ha="center", fontsize=10, color=GREEN, weight="bold")

    ax.text(2.5, 1.6, "parameters\nμ , σ", ha="center", fontsize=12, weight="bold", color=BLUE)
    ax.text(7.6, 1.55, "statistics\nx̄ , s", ha="center", fontsize=12, weight="bold", color="#8a5a00")
    ax.set_title("We measure a sample to make a claim about the population", fontsize=13, weight="bold")
    save(fig, "15_population_vs_sample.png")


# ---------------------------------------------------------------- 16. data types tree
def chart_data_types():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    _blank(ax)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)

    _box(ax, 5.0, 4.1, 2.0, 0.7, "DATA", face="#0b2545", edge="#0b2545", fs=13, weight="bold")
    ax.texts[-1].set_color("white")

    _box(ax, 1.6, 2.8, 2.6, 0.65, "Numerical\n(quantitative)", face="#dbe9ff", fs=10, weight="bold")
    _box(ax, 7.8, 2.8, 2.6, 0.65, "Categorical\n(qualitative)", face="#ffe6cc", edge=ORANGE, fs=10, weight="bold")

    _box(ax, 0.2, 1.3, 2.2, 0.85, "Discrete\ncountable\nbacklogs, projects", fs=9)
    _box(ax, 3.0, 1.3, 2.2, 0.85, "Continuous\nmeasurable\ncgpa, attendance", fs=9)
    _box(ax, 6.6, 1.3, 2.2, 0.85, "Nominal\nno order\ngender, branch", face="#fff3e6", edge=ORANGE, fs=9)
    _box(ax, 9.4, 1.3, 2.2, 0.85, "Ordinal\nhas order\nstress, city_tier", face="#fff3e6", edge=ORANGE, fs=9)

    def link(x1, y1, x2, y2, c=BLUE):
        ax.plot([x1, x1, x2, x2], [y1, (y1 + y2) / 2, (y1 + y2) / 2, y2], color=c, lw=2)

    link(6.0, 4.1, 2.9, 3.45)
    link(6.0, 4.1, 9.1, 3.45, ORANGE)
    link(2.9, 2.8, 1.3, 2.15)
    link(2.9, 2.8, 4.1, 2.15)
    link(9.1, 2.8, 7.7, 2.15, ORANGE)
    link(9.1, 2.8, 10.5, 2.15, ORANGE)

    ax.text(6.0, 0.5, "The data type decides the statistic:  mean of 'branch' = nonsense,  mode of 'branch' = perfect.",
            ha="center", fontsize=11, style="italic", color="#444")
    save(fig, "16_data_types.png")


# ---------------------------------------------------------------- 17. probability venn
def chart_probability_venn():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    titles = ["Mutually exclusive\nP(A∩B) = 0",
              "Overlapping\nP(A∪B) = P(A)+P(B)−P(A∩B)",
              "Conditional\nP(A|B) = P(A∩B) / P(B)"]
    offsets = [1.9, 0.65, 0.65]
    for ax, t, off in zip(axes, titles, offsets):
        _blank(ax)
        ax.set_xlim(-2, 2); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
        ax.add_patch(mpatches.Rectangle((-1.95, -1.2), 3.9, 2.4, fill=False, ec="#999", lw=1.2))
        ax.add_patch(mpatches.Circle((-off / 2, 0), 0.8, facecolor=BLUE, alpha=.45, edgecolor=BLUE, lw=2))
        ax.add_patch(mpatches.Circle((off / 2, 0), 0.8, facecolor=ORANGE, alpha=.45, edgecolor=ORANGE, lw=2))
        ax.text(-off / 2 - 0.25, 0, "A", fontsize=15, weight="bold", ha="center", va="center")
        ax.text(off / 2 + 0.25, 0, "B", fontsize=15, weight="bold", ha="center", va="center")
        ax.text(1.8, 1.0, "S", fontsize=11, color="#666")
        ax.set_title(t, fontsize=10.5)
    axes[1].text(0, 0, "A∩B", fontsize=10, ha="center", va="center", weight="bold")
    axes[2].add_patch(mpatches.Circle((0.325, 0), 0.8, facecolor="none", edgecolor=GREEN, lw=3, ls="--"))
    axes[2].text(0.325, -1.05, "B becomes the new sample space",
                 fontsize=8.5, ha="center", color=GREEN, weight="bold")
    fig.suptitle("Probability rules as pictures", y=1.04, fontsize=13, weight="bold")
    save(fig, "17_probability_venn.png")


# ---------------------------------------------------------------- 18. distribution gallery
def chart_distribution_gallery():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.0))
    _ = [_blank(a) for a in axes]

    axes[0].bar(range(1, 7), [1 / 6] * 6, color=BLUE)
    axes[0].set_title("Uniform\nfair die roll", fontsize=10)

    axes[1].bar([0, 1], [0.35, 0.65], color=GREEN, width=.5)
    axes[1].set_xticks([0, 1], ["no", "yes"]); axes[1].tick_params(labelbottom=True)
    axes[1].set_title("Bernoulli\none student placed?", fontsize=10)

    k = np.arange(40, 81)
    axes[2].bar(k, stats.binom.pmf(k, 100, 0.6), color=ORANGE)
    axes[2].set_title("Binomial\n# placed out of 100", fontsize=10)

    k2 = np.arange(0, 14)
    axes[3].bar(k2, stats.poisson.pmf(k2, 4), color=RED)
    axes[3].set_title("Poisson\ntickets per hour", fontsize=10)

    xn = np.linspace(-4, 4, 400)
    axes[4].fill_between(xn, stats.norm.pdf(xn), color="#5b9bff")
    axes[4].set_title("Normal\nheight, CGPA, errors", fontsize=10)

    fig.suptitle("Five distributions every beginner should recognise", y=1.06, fontsize=13, weight="bold")
    save(fig, "18_distribution_gallery.png")


# ---------------------------------------------------------------- 19. hypothesis testing flow
def chart_hypothesis_flow():
    fig, ax = plt.subplots(figsize=(13, 3.2))
    _blank(ax)
    ax.set_xlim(0, 13); ax.set_ylim(0, 3)

    steps = ["1. State\nH₀ and H₁", "2. Choose α\nand tail", "3. Check\nassumptions",
             "4. Compute\nthe z statistic", "5. Get the\np-value", "6. Decide &\nexplain"]
    for i, s in enumerate(steps):
        _box(ax, 0.15 + i * 2.15, 1.3, 1.85, 1.0, s, fs=10, weight="bold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.15 + (i + 1) * 2.15, 1.8), xytext=(2.0 + i * 2.15, 1.8),
                        arrowprops=dict(arrowstyle="-|>", lw=2, color="#666"))

    _box(ax, 3.4, 0.15, 2.6, 0.8, "p ≤ α  →  REJECT H₀", face="#fdeff2", edge=RED, fs=10, weight="bold")
    _box(ax, 6.6, 0.15, 3.0, 0.8, "p > α  →  FAIL TO REJECT H₀", face="#eafaf1", edge=GREEN, fs=10, weight="bold")
    ax.set_title("The 6-step hypothesis testing recipe", fontsize=13, weight="bold")
    save(fig, "19_hypothesis_flow.png")


# ---------------------------------------------------------------- 20. z-test or t-test?
def chart_z_or_t():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _blank(ax)
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6)

    _box(ax, 3.8, 3.6, 3.4, 0.8, "Is the population σ known?", fs=11, weight="bold")
    _box(ax, 0.4, 2.0, 3.0, 0.8, "YES", face="#eafaf1", edge=GREEN, fs=11, weight="bold")
    _box(ax, 7.6, 2.0, 3.0, 0.8, "NO", face="#fff3e6", edge=ORANGE, fs=11, weight="bold")
    _box(ax, 0.4, 0.5, 3.0, 0.9, "Z-TEST\nz = (x̄−μ₀)/(σ/√n)", face="#dbe9ff", fs=10.5, weight="bold")
    _box(ax, 4.0, 0.5, 3.0, 0.9, "n ≥ 30 ?\nZ-test  (CLT)", face="#dbe9ff", fs=10.5, weight="bold")
    _box(ax, 7.6, 0.5, 3.0, 0.9, "n < 30 ?\nt-TEST", face="#ffe6cc", edge=ORANGE, fs=10.5, weight="bold")

    for x0, x1, c in [(5.5, 1.9, GREEN), (5.5, 9.1, ORANGE)]:
        ax.annotate("", xy=(x1, 2.8), xytext=(x0, 3.6),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color=c))
    ax.annotate("", xy=(1.9, 1.4), xytext=(1.9, 2.0), arrowprops=dict(arrowstyle="-|>", lw=2, color=GREEN))
    ax.annotate("", xy=(5.5, 1.4), xytext=(8.8, 2.0), arrowprops=dict(arrowstyle="-|>", lw=2, color=ORANGE))
    ax.annotate("", xy=(9.1, 1.4), xytext=(9.1, 2.0), arrowprops=dict(arrowstyle="-|>", lw=2, color=ORANGE))
    ax.set_title("Z-test or t-test?", fontsize=13, weight="bold")
    save(fig, "20_z_or_t.png")


if __name__ == "__main__":
    chart_central_tendency()
    chart_spread()
    chart_boxplot_anatomy()
    chart_skew()
    chart_empirical_rule()
    chart_zscore()
    chart_sampling()
    chart_clt()
    chart_standard_error()
    chart_rejection_regions()
    chart_worked_ztest()
    chart_error_types()
    chart_confidence_interval()
    chart_dataset_overview()
    chart_population_vs_sample()
    chart_data_types()
    chart_probability_venn()
    chart_distribution_gallery()
    chart_hypothesis_flow()
    chart_z_or_t()
    print(f"\nAll charts written to {OUT}")
