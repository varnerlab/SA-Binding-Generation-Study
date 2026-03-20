#!/usr/bin/env python3
"""
Publication figure: Phase transition shifts rightward with increasing ρ.
Entropy normalized by log(K) so all curves live in [0, 1].
Styled to match fig2_separation_vs_gap.
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

DATA = REPO_ROOT / "code" / "data" / "kunitz" / "entropy_curves.csv"
OUT  = REPO_ROOT / "paper" / "sections" / "figs" / "fig5_entropy_curves.pdf"

# --- load data ---
curves = {}  # rho -> {beta, H, K_eff, beta_star}
with open(DATA) as f:
    for row in csv.DictReader(f):
        rho = float(row["rho"])
        if rho not in curves:
            curves[rho] = {"beta": [], "H": [], "K_eff": float(row["K_eff"]),
                           "beta_star": float(row["beta_star"])}
        curves[rho]["beta"].append(float(row["beta"]))
        curves[rho]["H"].append(float(row["H"]))

for rho in curves:
    curves[rho]["beta"] = np.array(curves[rho]["beta"])
    curves[rho]["H"] = np.array(curves[rho]["H"])

K = 99  # Kunitz family size
log_K = np.log(K)

# --- styling ---
rho_order = sorted(curves.keys())
palette = {
    1.0:    "#7f8c8d",   # gray
    5.0:    "#2980b9",   # blue
    20.0:   "#e74c3c",   # red
    100.0:  "#8e44ad",   # purple
    1000.0: "#27ae60",   # green
}

# --- figure ---
fig, ax = plt.subplots(figsize=(7, 4.5))

for rho in rho_order:
    d = curves[rho]
    color = palette[rho]
    H_norm = d["H"] / log_K

    # main curve
    ax.plot(d["beta"], H_norm, color=color, lw=2.5, alpha=0.9, zorder=2)

    # β* dashed line
    ax.axvline(d["beta_star"], color=color, ls="--", lw=1.2, alpha=0.5, zorder=1)

    # label at β* with K_eff
    # place label at the inflection point
    idx_star = np.argmin(np.abs(d["beta"] - d["beta_star"]))
    H_at_star = H_norm[idx_star]

    # small annotation near the inflection
    ax.plot(d["beta_star"], H_at_star, "o", color=color, markersize=5, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8)

# --- horizontal reference lines ---
# log(K)/log(K) = 1.0 (uniform)
ax.axhline(1.0, color="#bbb", ls=":", lw=1, zorder=0)
ax.text(0.12, 1.02, "$H = \\log K$ (uniform)", fontsize=8, color="#999", va="bottom")

# log(K_des)/log(K) = log(32)/log(99)
H_des_norm = np.log(32) / log_K
ax.axhline(H_des_norm, color="#bbb", ls=":", lw=1, zorder=0)
ax.text(0.12, H_des_norm + 0.02, f"$\\log K_{{\\mathrm{{des}}}}$ / $\\log K$",
        fontsize=8, color="#999", va="bottom")

# --- legend ---
handles = []
for rho in rho_order:
    d = curves[rho]
    K_eff_str = f"{d['K_eff']:.0f}" if d["K_eff"] == int(d["K_eff"]) else f"{d['K_eff']:.1f}"
    label = f"$\\rho$ = {int(rho)}  ($K_{{\\mathrm{{eff}}}}$ = {K_eff_str}, $\\beta^*$ = {d['beta_star']:.1f})"
    handles.append(Line2D([0], [0], color=palette[rho], lw=2.5, label=label))

ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.95,
          edgecolor="#ddd", borderpad=0.8)

# --- axes ---
ax.set_xscale("log")
ax.set_xlabel("Inverse temperature  $\\beta$", fontsize=12)
ax.set_ylabel("Normalized attention entropy  $H(\\beta) \\;/\\; \\log K$", fontsize=12)
ax.set_xlim(0.1, 500)
ax.set_ylim(-0.05, 1.12)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# light grid
ax.grid(True, alpha=0.15, which="both")
ax.grid(True, alpha=0.08, which="minor")

# annotate the rightward shift
ax.annotate("", xy=(9.3, 0.15), xytext=(4.3, 0.15),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.8))
ax.text(1.8, 0.10, "increasing $\\rho$\nshifts $\\beta^*$ rightward",
        fontsize=9, color="#555", ha="center", va="top", fontstyle="italic")

plt.tight_layout()
fig.savefig(str(OUT), dpi=300)
fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200)
print(f"Saved: {OUT}")
print(f"log(K) = {log_K:.3f}")
for rho in rho_order:
    d = curves[rho]
    print(f"  ρ={int(rho):5d}: K_eff={d['K_eff']:.1f}, β*={d['beta_star']:.2f}, "
          f"H(0)/logK={d['H'][0]/log_K:.3f}")
