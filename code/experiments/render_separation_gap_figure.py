#!/usr/bin/env python3
"""
Two-panel figure for the S-vs-Δ relationship.

Panel A: Calibration trajectories — f_obs vs f_eff for all 4 families across ρ values.
         Diagonal = perfect calibration. Distance below diagonal = calibration gap.

Panel B: Summary — S vs Δ with linear fit and R² (4 points: WW, Kunitz, SH3, Conotoxin).
"""
import csv, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = pathlib.Path(".")
for parent in SCRIPT_PATH.parents:
    if (parent / "code" / "data").is_dir() and (parent / "paper" / "sections").is_dir():
        REPO_ROOT = parent
        break

if REPO_ROOT == pathlib.Path("."):
    raise RuntimeError("Could not locate repository root from script path")

DATA = REPO_ROOT / "code" / "data"
OUT  = REPO_ROOT / "paper" / "sections" / "figs" / "fig2_separation_vs_gap.pdf"

# --- load per-family sweep data ---
families = {
    "WW":         {"color": "#e74c3c", "marker": "s",  "dir": "ww",              "label": "WW ($S={S:.2f}$)"},
    "Kunitz":     {"color": "#2980b9", "marker": "o",  "dir": "kunitz",          "label": "Kunitz ($S={S:.2f}$)"},
    "SH3":        {"color": "#27ae60", "marker": "^",  "dir": "sh3",             "label": "SH3 ($S={S:.2f}$)"},
    "Conotoxin":  {"color": "#8e44ad", "marker": "D",  "dir": "omega_conotoxin", "label": "$\\omega$-Conotoxin ($S={S:.2f}$)"},
}

# load sweep data and S from multi_family_comparison.csv
comp_path = DATA / "multi_family_comparison.csv"
with open(comp_path) as f:
    comp_rows = {r["family"]: r for r in csv.DictReader(f)}

for fam in families:
    # load S from comparison table
    families[fam]["S"] = float(comp_rows[fam]["separation_index"])
    families[fam]["label"] = families[fam]["label"].format(S=families[fam]["S"])

    # load sweep
    path = DATA / families[fam]["dir"] / "multiplicity_sweep.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    families[fam]["rho"]   = np.array([float(r["ρ"]) for r in rows])
    families[fam]["f_eff"] = np.array([float(r["f_eff"]) for r in rows])
    families[fam]["f_obs"] = np.array([float(r["f_obs"]) for r in rows])

# compute gap at max ρ
for fam in families:
    d = families[fam]
    d["Delta"] = d["f_eff"][-1] - d["f_obs"][-1]

# --- order by S ascending ---
fam_order = sorted(families.keys(), key=lambda f: families[f]["S"])

# --- figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5),
                                gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.35})

# ── Panel A: calibration trajectories ──
ax1.plot([0, 1], [0, 1], ls="--", color="#bbb", lw=1.2, zorder=0, label="_nolegend_")
ax1.fill_between([0, 1], [0, 1], [0, 0], color="#f0f0f0", alpha=0.5, zorder=0)

# per-family ρ annotation config
rho_labels = {
    "WW":         {1: (-0.01, -0.04), 500: (0.01, -0.04)},
    "Kunitz":     {1: (0.01, -0.04),  500: (-0.10, 0.02)},
    "SH3":        {1: (0.01, 0.02),   500: (-0.10, 0.02)},
    "Conotoxin":  {1: (0.01, 0.02),   500: (-0.10, -0.04)},
}

for fam_name in fam_order:
    d = families[fam_name]
    ax1.plot(d["f_eff"], d["f_obs"], color=d["color"], lw=2.2, zorder=2, alpha=0.85)
    ax1.scatter(d["f_eff"], d["f_obs"], color=d["color"], marker=d["marker"],
                s=50, zorder=3, edgecolors="white", linewidths=0.5)

    # annotate select ρ values
    for i, rho in enumerate(d["rho"]):
        if rho in rho_labels.get(fam_name, {}):
            ox, oy = rho_labels[fam_name][rho]
            ax1.annotate(f"$\\rho$={int(rho)}", (d["f_eff"][i], d["f_obs"][i]),
                        fontsize=6.5, color=d["color"], alpha=0.8,
                        xytext=(d["f_eff"][i] + ox, d["f_obs"][i] + oy))

    # draw gap arrow at max ρ
    fe, fo = d["f_eff"][-1], d["f_obs"][-1]
    gap = fe - fo
    if gap > 0.05:
        # stagger arrows horizontally
        x_offsets = {"WW": -0.04, "Kunitz": 0.01, "Conotoxin": -0.01}
        x_arrow = fe + x_offsets.get(fam_name, 0)
        ax1.annotate("", xy=(x_arrow, fo), xytext=(x_arrow, fe),
                     arrowprops=dict(arrowstyle="<->", color=d["color"], lw=1.5, alpha=0.6))
        mid = (fe + fo) / 2
        x_txt = x_arrow + 0.015
        ax1.text(x_txt, mid, f"$\\Delta$={gap:.2f}",
                fontsize=8, color=d["color"], va="center", fontstyle="italic")

ax1.set_xlabel("Effective designated fraction  $f_{\\mathrm{eff}}$", fontsize=11)
ax1.set_ylabel("Observed phenotype fraction  $f_{\\mathrm{obs}}$", fontsize=11)
ax1.set_xlim(0, 1.08)
ax1.set_ylim(0, 1.08)
ax1.set_aspect("equal")
ax1.text(0.05, 0.95, "perfect\ncalibration", transform=ax1.transAxes,
         fontsize=8, color="#999", va="top", fontstyle="italic")

# legend
handles = [Line2D([0], [0], color=families[f]["color"], marker=families[f]["marker"],
                   lw=2, markersize=6, markeredgecolor="white", markeredgewidth=0.5,
                   label=families[f]["label"]) for f in fam_order]
ax1.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
ax1.text(-0.12, 1.05, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold")

# ── Panel B: S vs Δ (Pfam families only for linear fit; conotoxin as open point) ──
pfam_order = [f for f in fam_order if f != "Conotoxin"]
S_pfam   = np.array([families[f]["S"] for f in pfam_order])
gap_pfam = np.array([families[f]["Delta"] for f in pfam_order])

# plot Pfam points (filled)
for i, fam_name in enumerate(pfam_order):
    ax2.scatter(S_pfam[i], gap_pfam[i], color=families[fam_name]["color"],
                marker=families[fam_name]["marker"], s=120, zorder=3,
                edgecolors="white", linewidths=1)

# linear fit on Pfam families only
slope = np.sum((S_pfam - S_pfam.mean()) * (gap_pfam - gap_pfam.mean())) / \
        np.sum((S_pfam - S_pfam.mean())**2)
intercept = gap_pfam.mean() - slope * S_pfam.mean()
ss_res = np.sum((gap_pfam - (intercept + slope * S_pfam))**2)
ss_tot = np.sum((gap_pfam - gap_pfam.mean())**2)
r_sq = 1 - ss_res / ss_tot

x_fit = np.linspace(0.05, 0.90, 100)
y_fit = intercept + slope * x_fit
ax2.plot(x_fit, np.clip(y_fit, 0, None), ls="--", color="#e67e22", lw=2, alpha=0.8)
ax2.fill_between(x_fit, np.clip(y_fit, 0, None), alpha=0.06, color="#e67e22")

# conotoxin as open marker (not in fit)
d_ctx = families["Conotoxin"]
ax2.scatter(d_ctx["S"], d_ctx["Delta"], color=d_ctx["color"],
            marker=d_ctx["marker"], s=120, zorder=3,
            facecolors="none", edgecolors=d_ctx["color"], linewidths=2)

# label all points
label_offsets = {
    "WW":        (0.015, 0.02),
    "Kunitz":    (0.015, 0.02),
    "SH3":       (0.015, -0.04),
    "Conotoxin": (-0.22, -0.04),
}
for fam_name in fam_order:
    d = families[fam_name]
    ox, oy = label_offsets[fam_name]
    lbl = "$\\omega$-Cntx" if fam_name == "Conotoxin" else fam_name
    ax2.annotate(lbl, (d["S"], d["Delta"]),
                xytext=(d["S"] + ox, d["Delta"] + oy),
                fontsize=10, color=d["color"], fontweight="bold")

# equation box
ax2.text(0.42, 0.55, f"$\\Delta \\approx {intercept:.2f} {slope:+.1f}\\,S$\n$R^2 = {r_sq:.2f}$\n(Pfam families)",
         fontsize=9, color="#e67e22", fontstyle="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e67e22", alpha=0.8))

ax2.set_xlabel("Fisher separation index  $S$", fontsize=11)
ax2.set_ylabel("Calibration gap  $\\Delta = f_{\\mathrm{eff}} - f_{\\mathrm{obs}}$", fontsize=11)
ax2.set_xlim(0.05, 0.90)
ax2.set_ylim(-0.05, 0.75)
ax2.text(-0.15, 1.05, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

plt.tight_layout()
fig.savefig(str(OUT), dpi=300)
fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200)
print(f"Saved: {OUT}")
print(f"Saved: {str(OUT).replace('.pdf', '.png')}")
print(f"\nLinear fit: Δ ≈ {intercept:.3f} {slope:+.3f} S  (R² = {r_sq:.3f})")
for f in fam_order:
    print(f"  {f:12s}: S={families[f]['S']:.3f}, Δ={families[f]['Delta']:.3f}")
