#!/usr/bin/env python3
"""
Two-panel figure for the S-vs-Δ relationship (5 Pfam families + conotoxin).

Panel A: Calibration trajectories — f_obs vs f_eff for all 6 families across ρ values.
         Diagonal = perfect calibration. Distance below diagonal = calibration gap.

Panel B: Summary — S vs Δ with linear fit and R² (5 Pfam + conotoxin open marker).
"""
import csv, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = pathlib.Path(".")
for parent in SCRIPT_PATH.parents:
    if (parent / "code" / "data").is_dir() and (parent / "paper-arxiv" / "sections").is_dir():
        REPO_ROOT = parent
        break

if REPO_ROOT == pathlib.Path("."):
    raise RuntimeError("Could not locate repository root from script path")

DATA = REPO_ROOT / "code" / "data"
OUTS = [REPO_ROOT / "paper-jcim"  / "sections" / "figs" / "fig2_separation_vs_gap.pdf",
        REPO_ROOT / "paper-arxiv" / "sections" / "figs" / "fig2_separation_vs_gap.pdf"]

# --- family definitions ---
families = {
    "WW":         {"color": "#e74c3c", "marker": "s",  "dir": "ww",              "label": "WW ($S={S:.2f}$)"},
    "Forkhead":   {"color": "#e67e22", "marker": "p",  "dir": "forkhead",        "label": "Forkhead ($S={S:.2f}$)"},
    "Kunitz":     {"color": "#2980b9", "marker": "o",  "dir": "kunitz",          "label": "Kunitz ($S={S:.2f}$)"},
    "SH3":        {"color": "#27ae60", "marker": "^",  "dir": "sh3",             "label": "SH3 ($S={S:.2f}$)"},
    "Homeobox":   {"color": "#16a085", "marker": "v",  "dir": "homeobox",        "label": "Homeobox ($S={S:.2f}$)"},
    "Conotoxin":  {"color": "#8e44ad", "marker": "D",  "dir": "omega_conotoxin", "label": "$\\omega$-Conotoxin ($S={S:.2f}$)"},
}

# Load every manuscript-facing value from the canonical replicated source.
comp_path = DATA / "multi_family_comparison_6fam_aggregated.csv"
with open(comp_path) as f:
    comp_rows = {r["family"]: r for r in csv.DictReader(f)}

for fam in families:
    if fam not in comp_rows:
        raise ValueError(f"Missing {fam} from {comp_path}")
    families[fam]["S"] = float(comp_rows[fam]["separation_index"])
    families[fam]["Delta"] = float(comp_rows[fam]["cal_gap_mean"])
    families[fam]["Delta_std"] = float(comp_rows[fam]["cal_gap_std"])
    families[fam]["fit_included"] = comp_rows[fam]["fit_included"].lower() == "true"
    families[fam]["label"] = families[fam]["label"].format(S=families[fam]["S"])

    path = DATA / families[fam]["dir"] / "multiplicity_sweep_aggregated.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    families[fam]["rho"]   = np.array([float(r["rho"]) for r in rows])
    families[fam]["f_eff"] = np.array([float(r["f_eff"]) for r in rows])
    families[fam]["f_obs"] = np.array([float(r["f_obs_mean"]) for r in rows])
    families[fam]["f_obs_std"] = np.array([float(r["f_obs_std"]) for r in rows])

# --- order by S ascending ---
fam_order = sorted(families.keys(), key=lambda f: families[f]["S"])

# --- figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.35})

# ── Panel A: calibration trajectories ──
ax1.plot([0, 1], [0, 1], ls="--", color="#bbb", lw=1.2, zorder=0, label="_nolegend_")
ax1.fill_between([0, 1], [0, 1], [0, 0], color="#f0f0f0", alpha=0.5, zorder=0)

# per-family ρ annotation config (only annotate select families to avoid clutter)
rho_labels = {
    "WW":         {1: (-0.01, -0.04), 500: (0.01, -0.04)},
    "Kunitz":     {1: (0.01, -0.04),  500: (-0.10, 0.02)},
    "SH3":        {1: (0.01, 0.02),   500: (-0.10, 0.02)},
    "Conotoxin":  {1: (0.01, 0.02),   500: (-0.10, -0.04)},
}

for fam_name in fam_order:
    d = families[fam_name]
    ax1.plot(d["f_eff"], d["f_obs"], color=d["color"], lw=2.2, zorder=2, alpha=0.85)
    ax1.errorbar(d["f_eff"], d["f_obs"], yerr=d["f_obs_std"], color=d["color"],
                 marker=d["marker"], markersize=6, linestyle="none", capsize=2,
                 zorder=3, markeredgecolor="white", markeredgewidth=0.5)

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
    if gap > 0.08:
        x_offsets = {"WW": -0.04, "Kunitz": 0.01, "Forkhead": -0.06, "Conotoxin": -0.01}
        x_arrow = fe + x_offsets.get(fam_name, 0)
        ax1.annotate("", xy=(x_arrow, fo), xytext=(x_arrow, fe),
                     arrowprops=dict(arrowstyle="<->", color=d["color"], lw=1.5, alpha=0.6))
        mid = (fe + fo) / 2
        x_txt = x_arrow + 0.015
        ax1.text(x_txt, mid, f"$\\Delta$={gap:.2f}",
                fontsize=7.5, color=d["color"], va="center", fontstyle="italic")

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
ax1.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.9)
ax1.text(-0.12, 1.05, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold")

# ── Panel B: S vs Δ (Pfam families for linear fit; conotoxin as open point) ──
pfam_order = [f for f in fam_order if families[f]["fit_included"]]
S_pfam   = np.array([families[f]["S"] for f in pfam_order])
gap_pfam = np.array([families[f]["Delta"] for f in pfam_order])
gap_sd_pfam = np.array([families[f]["Delta_std"] for f in pfam_order])

# plot Pfam points (filled)
for i, fam_name in enumerate(pfam_order):
    ax2.errorbar(S_pfam[i], gap_pfam[i], yerr=gap_sd_pfam[i],
                 color=families[fam_name]["color"], marker=families[fam_name]["marker"],
                 markersize=10, capsize=3, linestyle="none", zorder=3,
                 markeredgecolor="white", markeredgewidth=1)

# linear fit on Pfam families only
slope = np.sum((S_pfam - S_pfam.mean()) * (gap_pfam - gap_pfam.mean())) / \
        np.sum((S_pfam - S_pfam.mean())**2)
intercept = gap_pfam.mean() - slope * S_pfam.mean()
ss_res = np.sum((gap_pfam - (intercept + slope * S_pfam))**2)
ss_tot = np.sum((gap_pfam - gap_pfam.mean())**2)
r_sq = 1 - ss_res / ss_tot

x_fit = np.linspace(S_pfam.min(), S_pfam.max(), 100)
y_fit = intercept + slope * x_fit
ax2.plot(x_fit, y_fit, ls="--", color="#e67e22", lw=2, alpha=0.8)

# conotoxin as open marker (not in fit)
d_ctx = families["Conotoxin"]
ax2.scatter(d_ctx["S"], d_ctx["Delta"], color=d_ctx["color"],
            marker=d_ctx["marker"], s=120, zorder=3,
            facecolors="none", edgecolors=d_ctx["color"], linewidths=2)
ax2.errorbar(d_ctx["S"], d_ctx["Delta"], yerr=d_ctx["Delta_std"],
             color=d_ctx["color"], capsize=3, linestyle="none", zorder=2)

# label all points
label_offsets = {
    "WW":        (0.015, 0.02),
    "Forkhead":  (0.015, 0.02),
    "Kunitz":    (0.015, 0.02),
    "SH3":       (0.015, -0.04),
    "Homeobox":  (0.015, -0.04),
    "Conotoxin": (-0.04, -0.05),
}
for fam_name in fam_order:
    d = families[fam_name]
    ox, oy = label_offsets[fam_name]
    lbl = "$\\omega$-Cntx" if fam_name == "Conotoxin" else fam_name
    ax2.annotate(lbl, (d["S"], d["Delta"]),
                xytext=(d["S"] + ox, d["Delta"] + oy),
                fontsize=9.5, color=d["color"], fontweight="bold")

# equation box
ax2.text(0.55, 0.88, f"$\\Delta \\approx {intercept:.2f} {slope:+.1f}\\,S$\n$R^2 = {r_sq:.2f}$\n(Pfam families)",
         fontsize=9, color="#e67e22", fontstyle="italic", transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e67e22", alpha=0.8),
         va="top")

ax2.set_xlabel("Separation index  $S$", fontsize=11)
ax2.set_ylabel("Calibration gap  $\\Delta = f_{\\mathrm{eff}} - f_{\\mathrm{obs}}$", fontsize=11)
ax2.set_xlim(0.05, 0.90)
ax2.set_ylim(-0.05, 0.75)
ax2.text(-0.15, 1.05, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

plt.tight_layout()
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=300)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200)
    print(f"Saved: {out}")
    print(f"Saved: {str(out).replace('.pdf', '.png')}")
print(f"\nLinear fit (Pfam only): Δ ≈ {intercept:.3f} {slope:+.3f} S  (R² = {r_sq:.3f})")
for f in fam_order:
    print(f"  {f:12s}: S={families[f]['S']:.3f}, Δ={families[f]['Delta']:.3f}")
