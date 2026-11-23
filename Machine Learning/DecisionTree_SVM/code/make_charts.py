"""Generates every chart used in DT_SVM_Slides.html.

Run:  python make_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_circles
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "assets"
OUT.mkdir(exist_ok=True)
RAW = BASE.parent / "linear_regression_project" / "raw_data"

BLUE, RED, GREEN, ORANGE, PURPLE = "#1f6feb", "#d1495b", "#2e9e5b", "#e8952f", "#7b4fbf"
CMAP = ListedColormap(["#ffd9df", "#cfe0ff"])          # Not Placed / Placed
POINTS = ListedColormap([RED, BLUE])

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 11,
    "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False,
})
rng = np.random.default_rng(7)


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ================================================================== data
def load():
    df = (pd.read_csv(RAW / "indian_engineering_student_placement.csv")
            .merge(pd.read_csv(RAW / "placement_targets.csv"), on="Student_ID"))
    df["placed"] = (df["placement_status"] == "Placed").astype(int)
    return df


df = load()

# two continuous-ish features so the 2-D pictures are readable
F1, F2 = "cgpa", "coding_skill_rating"
X2 = df[[F1, F2]].to_numpy(float)
X2[:, 1] += rng.normal(0, .13, len(X2))                 # jitter the 1-5 rating
y2 = df["placed"].to_numpy()
X2tr, X2te, y2tr, y2te = train_test_split(X2, y2, test_size=.3, random_state=42, stratify=y2)

NUM = ["cgpa", "tenth_percentage", "twelfth_percentage", "backlogs", "study_hours_per_day",
       "attendance_percentage", "projects_completed", "internships_completed",
       "coding_skill_rating", "communication_skill_rating", "aptitude_skill_rating",
       "hackathons_participated", "certifications_count", "sleep_hours", "stress_level"]
Xall = df[NUM]
Xtr, Xte, ytr, yte = train_test_split(Xall, df["placed"], test_size=.3,
                                      random_state=42, stratify=df["placed"])

placed = df[df["placed"] == 1]
Xr = placed[NUM]
yr = placed["salary_lpa"]
Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=.3, random_state=42)


def boundary(ax, model, X, y, title, step=.03):
    x_min, x_max = X[:, 0].min() - .3, X[:, 0].max() + .3
    y_min, y_max = X[:, 1].min() - .4, X[:, 1].max() + .4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap=CMAP, alpha=.85)
    idx = rng.choice(len(X), min(700, len(X)), replace=False)
    ax.scatter(X[idx, 0], X[idx, 1], c=y[idx], cmap=POINTS, s=9, alpha=.55,
               edgecolors="none")
    ax.set_xlabel("CGPA"); ax.set_ylabel("coding skill rating")
    ax.set_title(title, fontsize=11)
    ax.grid(False)


# ================================================================== 01 dataset
def c01_dataset():
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    counts = df["placement_status"].value_counts()
    ax[0].bar(counts.index, counts.values, color=[BLUE, RED], width=.55)
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"\n{v:,}\n({v/len(df)*100:.0f}%)", ha="center", va="top",
                   color="white", fontweight="bold")
    ax[0].set_title("TARGET 1 — placement_status  (classification)")
    ax[0].set_ylabel("students")

    ax[1].hist(placed["salary_lpa"], bins=40, color=GREEN, edgecolor="white")
    ax[1].set_title("TARGET 2 — salary_lpa of placed students  (regression)")
    ax[1].set_xlabel("salary (LPA)"); ax[1].set_ylabel("students")
    fig.suptitle("5,000 engineering students · 23 features · 2 questions",
                 y=1.04, fontsize=13, fontweight="bold")
    save(fig, "01_dataset.png")


# ================================================================== 02 two tasks
def c02_two_tasks():
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    s = df.sample(500, random_state=0)
    ax[0].scatter(s["cgpa"], s["coding_skill_rating"] + rng.normal(0, .12, len(s)),
                  c=s["placed"], cmap=POINTS, s=18, alpha=.7)
    ax[0].set_title("CLASSIFICATION\npredict a LABEL:  Placed / Not Placed")
    ax[0].set_xlabel("CGPA"); ax[0].set_ylabel("coding skill")

    sp = placed.sample(500, random_state=0)
    ax[1].scatter(sp["cgpa"], sp["salary_lpa"], color=GREEN, s=18, alpha=.5)
    ax[1].set_title("REGRESSION\npredict a NUMBER:  salary in LPA")
    ax[1].set_xlabel("CGPA"); ax[1].set_ylabel("salary (LPA)")
    save(fig, "02_two_tasks.png")


# ================================================================== 03 the tree
def c03_tree_diagram():
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40,
                                 random_state=0).fit(Xtr, ytr)
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(clf, feature_names=NUM, class_names=["Not Placed", "Placed"],
              filled=True, rounded=True, fontsize=9, impurity=False, proportion=True, ax=ax)
    ax.set_title("A real decision tree trained on our students (max_depth=3)",
                 fontsize=14, fontweight="bold")
    save(fig, "03_tree_diagram.png")


def c04_first_split():
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for a, feat, cut in [(ax[0], "cgpa", 7.5), (ax[1], "coding_skill_rating", 2.5)]:
        left = df[df[feat] <= cut]["placed"].mean() * 100
        right = df[df[feat] > cut]["placed"].mean() * 100
        a.bar([f"{feat}\n<= {cut}", f"{feat}\n> {cut}"], [left, right],
              color=[ORANGE, BLUE], width=.5)
        a.axhline(df["placed"].mean() * 100, color="black", ls="--",
                  label=f"before split: {df['placed'].mean()*100:.0f}% placed")
        for i, v in enumerate([left, right]):
            a.text(i, v - 6, f"{v:.0f}%", ha="center", color="white", fontweight="bold")
        a.set_ylabel("% placed"); a.set_ylim(0, 105); a.legend(fontsize=8)
        a.set_title(f"Split on {feat}")
    fig.suptitle("How a tree picks a question: the split that separates best wins",
                 y=1.03, fontsize=13, fontweight="bold")
    save(fig, "04_first_split.png")


def c05_depth_boundaries():
    for d in [1, 2, 3, 5, 12, None]:
        clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X2tr, y2tr)
        acc = clf.score(X2te, y2te)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        lbl = "None (grown fully)" if d is None else d
        boundary(ax, clf, X2, y2, f"max_depth = {lbl}   |   test accuracy = {acc:.3f}")
        save(fig, f"05_depth_{d if d else 'full'}.png")


def c06_overfitting():
    depths = range(1, 21)
    tr, te = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        tr.append(m.score(Xtr, ytr)); te.append(m.score(Xte, yte))
    best = depths[int(np.argmax(te))]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(depths, tr, "o-", color=BLUE, lw=2, label="training accuracy")
    ax.plot(depths, te, "s-", color=RED, lw=2, label="test accuracy")
    ax.axvline(best, color=GREEN, ls="--", lw=2, label=f"sweet spot: depth {best}")
    ax.fill_between(depths, te, tr, color=ORANGE, alpha=.18)
    ax.annotate("the gap = OVERFITTING\n(memorising, not learning)", xy=(17, (tr[16]+te[16])/2),
                xytext=(9.5, .90), fontsize=10, color="#8a5a00", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#8a5a00"))
    ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy"); ax.set_xticks(list(depths)[::2])
    ax.set_title("Deeper is NOT better", fontweight="bold"); ax.legend()
    save(fig, "06_overfitting.png")


def c07_tree_regression():
    x = placed[["cgpa"]].to_numpy()
    y = placed["salary_lpa"].to_numpy()
    grid = np.linspace(x.min(), x.max(), 600).reshape(-1, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, d in zip(axes, [1, 3, 10]):
        m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(x, y)
        ax.scatter(x, y, s=8, alpha=.18, color="#9bb8e8")
        ax.plot(grid, m.predict(grid), color=RED, lw=2.5)
        ax.set_title(f"max_depth = {d}", fontsize=11)
        ax.set_xlabel("CGPA")
    axes[0].set_ylabel("salary (LPA)")
    fig.suptitle("A regression tree predicts a STAIRCASE — the average salary inside each box",
                 y=1.04, fontsize=13, fontweight="bold")
    save(fig, "07_tree_regression.png")


def c08_importance():
    clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
    imp = pd.Series(clf.feature_importances_, index=NUM).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(imp.index, imp.values, color=BLUE)
    for i, v in enumerate(imp.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("feature importance")
    ax.set_title("The tree tells you WHICH features it used", fontweight="bold")
    save(fig, "08_importance.png")


# ================================================================== SVM concept
Xb, yb = make_blobs(n_samples=90, centers=2, cluster_std=1.75, random_state=6)


def svm_panel(ax, model, X, y, title, show_sv=True):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z > 0, cmap=CMAP, alpha=.8)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="black",
               linestyles=["--", "-", "--"], linewidths=[1.2, 2.2, 1.2])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=POINTS, s=34, edgecolors="white", lw=.6)
    if show_sv and hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=170, facecolors="none",
                   edgecolors=GREEN, lw=2.2, label=f"{len(sv)} support vectors")
        ax.legend(fontsize=9, loc="upper left")
    ax.set_title(title, fontsize=11); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])


def c09_margin():
    m = SVC(kernel="linear", C=0.4).fit(Xb, yb)
    fig, ax = plt.subplots(figsize=(8, 5))
    svm_panel(ax, m, Xb, yb, "")
    ax.set_title("SVM finds the WIDEST STREET between the two classes",
                 fontsize=13, fontweight="bold")
    ax.text(.02, .02, "solid line = the decision boundary\n"
                      "dashed lines = the edges of the street (the margin)\n"
                      "circled points = support vectors, the only points that matter",
            transform=ax.transAxes, fontsize=9.5, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=.85))
    save(fig, "09_margin.png")


def c10_C():
    for C in [0.01, 1, 100]:
        m = SVC(kernel="linear", C=C).fit(Xb, yb)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        svm_panel(ax, m, Xb, yb, f"C = {C}   →   {'WIDE street, some mistakes allowed' if C < 1 else 'NARROW street, few mistakes allowed' if C > 10 else 'balanced'}")
        save(fig, f"10_C_{C}.png")


Xc, yc = make_circles(n_samples=180, factor=.4, noise=.11, random_state=3)


def c11_kernel():
    for k, note in [("linear", "a straight line CANNOT split this"),
                    ("rbf", "RBF bends the boundary — problem solved"),
                    ("poly", "polynomial curves it too")]:
        m = SVC(kernel=k, C=10, gamma="scale", degree=3).fit(Xc, yc)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        svm_panel(ax, m, Xc, yc, f"kernel = '{k}'   |   accuracy = {m.score(Xc, yc):.2f}\n{note}",
                  show_sv=False)
        save(fig, f"11_kernel_{k}.png")


def c12_gamma():
    for g, note in [(0.1, "too smooth — underfits"),
                    (1.0, "just right"),
                    (30, "islands around single points — overfits")]:
        m = SVC(kernel="rbf", C=10, gamma=g).fit(Xc, yc)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        svm_panel(ax, m, Xc, yc, f"gamma = {g}   →   {note}", show_sv=False)
        save(fig, f"12_gamma_{g}.png")


def c13_svr():
    x = np.linspace(5, 10, 120).reshape(-1, 1)
    y = 2.6 * x.ravel() - 5 + rng.normal(0, 1.6, len(x))
    eps = 2.0
    m = SVR(kernel="linear", C=100, epsilon=eps).fit(x, y)
    pred = m.predict(x)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.scatter(x, y, s=26, color="#9bb8e8", label="students")
    ax.plot(x, pred, color=RED, lw=2.5, label="SVR prediction")
    ax.fill_between(x.ravel(), pred - eps, pred + eps, color=ORANGE, alpha=.28,
                    label=f"the ε-tube (ε = {eps})")
    inside = np.abs(y - pred) <= eps
    ax.scatter(x[~inside], y[~inside], s=60, facecolors="none", edgecolors=GREEN, lw=1.8,
               label="outside the tube → these are the only ones that cost anything")
    ax.set_xlabel("CGPA"); ax.set_ylabel("salary (LPA)")
    ax.set_title("SVR: 'close enough is free' — errors inside the tube are ignored",
                 fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "13_svr_tube.png")


def c14_scaling():
    sub = df.sample(1200, random_state=1)
    Xs = sub[["cgpa", "attendance_percentage"]].to_numpy()
    ys = sub["placed"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, model, title in [
        (axes[0], SVC(kernel="rbf", C=1), "WITHOUT scaling\nattendance (44-99) drowns out CGPA (5-10)"),
        (axes[1], make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1)),
         "WITH StandardScaler\nboth features get an equal vote"),
    ]:
        model.fit(Xs, ys)
        x_min, x_max = Xs[:, 0].min() - .3, Xs[:, 0].max() + .3
        y_min, y_max = Xs[:, 1].min() - 3, Xs[:, 1].max() + 3
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, zz, cmap=CMAP, alpha=.85)
        ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=POINTS, s=10, alpha=.6)
        ax.set_title(f"{title}\naccuracy = {model.score(Xs, ys):.3f}", fontsize=10.5)
        ax.set_xlabel("CGPA"); ax.set_ylabel("attendance %"); ax.grid(False)
    fig.suptitle("⚠️ SVM measures DISTANCE — always scale your features",
                 y=1.05, fontsize=13, fontweight="bold")
    save(fig, "14_scaling.png")


# ================================================================== evaluation
def fit_models():
    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
    sv = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, probability=True,
                                             random_state=0)).fit(Xtr, ytr)
    return dt, sv


def c15_accuracy_trap():
    dt, sv = fit_models()
    dumb = np.ones(len(yte))
    scores = {
        "Always say\n'Placed'": (accuracy_score(yte, dumb), recall_score(yte, dumb, pos_label=0)),
        "Decision Tree": (dt.score(Xte, yte), recall_score(yte, dt.predict(Xte), pos_label=0)),
        "SVM (RBF)": (sv.score(Xte, yte), recall_score(yte, sv.predict(Xte), pos_label=0)),
    }
    names = list(scores)
    acc = [scores[n][0] for n in names]
    rec = [scores[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    xpos = np.arange(len(names))
    ax.bar(xpos - .2, acc, .4, color=BLUE, label="accuracy")
    ax.bar(xpos + .2, rec, .4, color=RED, label="recall for 'Not Placed'")
    for i, (a, r) in enumerate(zip(acc, rec)):
        ax.text(i - .2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + .2, r, f"{r:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xpos, names); ax.set_ylim(0, 1.12); ax.legend()
    ax.set_title("The accuracy trap: 86% accuracy while helping ZERO struggling students",
                 fontweight="bold", fontsize=12)
    save(fig, "15_accuracy_trap.png")


def c16_confusion():
    dt, sv = fit_models()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m, name in [(axes[0], dt, "Decision Tree"), (axes[1], sv, "SVM (RBF)")]:
        ConfusionMatrixDisplay.from_estimator(
            m, Xte, yte, display_labels=["Not Placed", "Placed"],
            cmap="Blues", colorbar=False, ax=ax)
        pred = m.predict(Xte)
        ax.set_title(f"{name}\nprecision {precision_score(yte, pred):.2f} · "
                     f"recall {recall_score(yte, pred):.2f} · F1 {f1_score(yte, pred):.2f}",
                     fontsize=11)
        ax.grid(False)
    fig.suptitle("Confusion matrix — where exactly did it go wrong?",
                 y=1.04, fontsize=13, fontweight="bold")
    save(fig, "16_confusion.png")


def c17_roc():
    dt, sv = fit_models()
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for m, name, c in [(dt, "Decision Tree", BLUE), (sv, "SVM (RBF)", RED)]:
        pr = m.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, pr)
        ax.plot(fpr, tpr, lw=2.5, color=c, label=f"{name}  (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guessing (AUC = 0.5)")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve — higher and more to the left is better", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "17_roc.png")


def c18_regression_eval():
    dtr = DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=0).fit(Xrtr, yrtr)
    svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=.5)).fit(Xrtr, yrtr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, m, name in [(axes[0], dtr, "Decision Tree Regressor"),
                        (axes[1], svr, "SVR (RBF)")]:
        pred = m.predict(Xrte)
        ax.scatter(yrte, pred, s=14, alpha=.35, color=BLUE)
        lims = [yrte.min(), yrte.max()]
        ax.plot(lims, lims, color=RED, lw=2, ls="--", label="perfect prediction")
        ax.set_xlabel("actual salary (LPA)"); ax.set_ylabel("predicted salary (LPA)")
        ax.set_title(f"{name}\nMAE {mean_absolute_error(yrte, pred):.2f} · "
                     f"RMSE {np.sqrt(((yrte-pred)**2).mean()):.2f} · "
                     f"R² {r2_score(yrte, pred):.2f}", fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle("Regression: how close are the dots to the dashed line?",
                 y=1.04, fontsize=13, fontweight="bold")
    save(fig, "18_regression_eval.png")


def c19_showdown():
    dt, sv = fit_models()
    dtr = DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=0).fit(Xrtr, yrtr)
    svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=.5)).fit(Xrtr, yrtr)

    clf_scores = {
        "accuracy":  [dt.score(Xte, yte), sv.score(Xte, yte)],
        "precision": [precision_score(yte, dt.predict(Xte)), precision_score(yte, sv.predict(Xte))],
        "recall":    [recall_score(yte, dt.predict(Xte)), recall_score(yte, sv.predict(Xte))],
        "F1":        [f1_score(yte, dt.predict(Xte)), f1_score(yte, sv.predict(Xte))],
    }
    reg_scores = {
        "R²":   [r2_score(yrte, dtr.predict(Xrte)), r2_score(yrte, svr.predict(Xrte))],
        "MAE":  [mean_absolute_error(yrte, dtr.predict(Xrte)),
                 mean_absolute_error(yrte, svr.predict(Xrte))],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    xpos = np.arange(len(clf_scores))
    axes[0].bar(xpos - .2, [v[0] for v in clf_scores.values()], .4, color=BLUE, label="Decision Tree")
    axes[0].bar(xpos + .2, [v[1] for v in clf_scores.values()], .4, color=ORANGE, label="SVM")
    for i, k in enumerate(clf_scores):
        axes[0].text(i - .2, clf_scores[k][0], f"{clf_scores[k][0]:.2f}", ha="center",
                     va="bottom", fontsize=8.5)
        axes[0].text(i + .2, clf_scores[k][1], f"{clf_scores[k][1]:.2f}", ha="center",
                     va="bottom", fontsize=8.5)
    axes[0].set_xticks(xpos, list(clf_scores)); axes[0].set_ylim(0, 1.15)
    axes[0].set_title("CLASSIFICATION — will they be placed?"); axes[0].legend(fontsize=9)

    xpos = np.arange(len(reg_scores))
    axes[1].bar(xpos - .2, [v[0] for v in reg_scores.values()], .4, color=BLUE, label="Decision Tree")
    axes[1].bar(xpos + .2, [v[1] for v in reg_scores.values()], .4, color=ORANGE, label="SVR")
    for i, k in enumerate(reg_scores):
        axes[1].text(i - .2, reg_scores[k][0], f"{reg_scores[k][0]:.2f}", ha="center",
                     va="bottom", fontsize=8.5)
        axes[1].text(i + .2, reg_scores[k][1], f"{reg_scores[k][1]:.2f}", ha="center",
                     va="bottom", fontsize=8.5)
    axes[1].set_xticks(xpos, list(reg_scores))
    axes[1].set_title("REGRESSION — what salary? (R² higher = better, MAE lower = better)")
    axes[1].legend(fontsize=9)
    fig.suptitle("Head to head on OUR data", y=1.04, fontsize=13, fontweight="bold")
    save(fig, "19_showdown.png")

    print("\n--- numbers for the slides ---")
    for k, v in clf_scores.items():
        print(f"{k:10s} DT {v[0]:.3f}   SVM {v[1]:.3f}")
    for k, v in reg_scores.items():
        print(f"{k:10s} DT {v[0]:.3f}   SVR {v[1]:.3f}")


def c20_forest():
    dt = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    dtp = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    names = ["1 tree\n(fully grown)", "1 tree\n(pruned)", "200 trees\n(Random Forest)"]
    vals = [dt.score(Xte, yte), dtp.score(Xte, yte), rf.score(Xte, yte)]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(names, vals, color=[RED, ORANGE, GREEN], width=.55)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 1.08); ax.set_ylabel("test accuracy")
    ax.set_title("One tree is weak — a FOREST of them is strong", fontweight="bold")
    save(fig, "20_forest.png")


if __name__ == "__main__":
    c01_dataset(); c02_two_tasks()
    c03_tree_diagram(); c04_first_split(); c05_depth_boundaries()
    c06_overfitting(); c07_tree_regression(); c08_importance()
    c09_margin(); c10_C(); c11_kernel(); c12_gamma(); c13_svr(); c14_scaling()
    c15_accuracy_trap(); c16_confusion(); c17_roc(); c18_regression_eval()
    c19_showdown(); c20_forest()
    print(f"\nAll charts written to {OUT}")
