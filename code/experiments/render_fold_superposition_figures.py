#!/usr/bin/env python3
"""
Render combined structural superposition figure for Kunitz and ω-conotoxin.
Top row: Kunitz (two views), Bottom row: ω-conotoxin (two views).
SA variants colored by per-residue pLDDT; reference in gray with alpha.
"""

import subprocess
import tempfile
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB import PDBParser
from scipy.interpolate import make_interp_spline
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TMALIGN = os.path.join(BASE, "code", "bin", "TMalign")
FIG_DIR = os.path.join(BASE, "paper", "figs")

parser = PDBParser(QUIET=True)

# pLDDT colormap (AlphaFold-style, matching parent paper)
PLDDT_COLORS = ["#ff0000", "#ff8800", "#ffff00", "#00cc00", "#00cccc", "#0000ff"]
PLDDT_CMAP = LinearSegmentedColormap.from_list("plddt", PLDDT_COLORS, N=256)
PLDDT_MIN, PLDDT_MAX = 50, 100


def extract_ca_coords_and_bfactors(pdb_path):
    """Extract CA atom coordinates and B-factors (pLDDT for ESMFold) from a PDB."""
    structure = parser.get_structure("s", pdb_path)
    coords, bfactors = [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    ca = residue["CA"]
                    coords.append(ca.get_vector().get_array())
                    bfactors.append(ca.get_bfactor())
        break
    return np.array(coords), np.array(bfactors)


def run_tmalign_and_get_aligned(mobile_pdb, target_pdb):
    """Run TM-align with -m flag and parse rotation matrix."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as mf:
        matrix_file = mf.name
    try:
        result = subprocess.run(
            [TMALIGN, mobile_pdb, target_pdb, "-m", matrix_file],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        tm_score = None
        for line in lines:
            if line.startswith("TM-score=") and "Chain_1" in line:
                tm_score = float(line.split("=")[1].split("(")[0].strip())

        with open(matrix_file) as f:
            mat_lines = f.readlines()

        t = np.zeros(3)
        U = np.zeros((3, 3))
        for ml in mat_lines:
            parts = ml.strip().split()
            if len(parts) == 5:
                try:
                    idx = int(parts[0])
                    if idx in (0, 1, 2):
                        t[idx] = float(parts[1])
                        U[idx, 0] = float(parts[2])
                        U[idx, 1] = float(parts[3])
                        U[idx, 2] = float(parts[4])
                except ValueError:
                    continue

        coords, bfactors = extract_ca_coords_and_bfactors(mobile_pdb)
        aligned_coords = (U @ coords.T).T + t
        return aligned_coords, bfactors, tm_score
    finally:
        os.unlink(matrix_file)


def smooth_backbone(coords, factor=5):
    """Smoothly interpolate CA backbone trace using cubic B-spline."""
    n = len(coords)
    if n < 4:
        return coords, np.arange(n)
    t = np.arange(n)
    t_smooth = np.linspace(0, n - 1, n * factor)
    smooth_coords = np.zeros((len(t_smooth), 3))
    for dim in range(3):
        spl = make_interp_spline(t, coords[:, dim], k=3)
        smooth_coords[:, dim] = spl(t_smooth)
    return smooth_coords, t_smooth


def smooth_values(values, n_orig, factor=5):
    """Interpolate per-residue values to match smoothed backbone."""
    t = np.arange(n_orig)
    t_smooth = np.linspace(0, n_orig - 1, n_orig * factor)
    spl = make_interp_spline(t, values, k=3)
    return np.clip(spl(t_smooth), PLDDT_MIN, PLDDT_MAX)


def plot_backbone_plddt(ax, coords, bfactors, linewidth=2.5, label=None):
    """Plot backbone trace colored by per-residue pLDDT."""
    n_orig = len(coords)
    smooth_c, _ = smooth_backbone(coords)

    # Scale bfactors: ESMFold uses 0-1 scale
    if bfactors.max() <= 1.0:
        bfactors = bfactors * 100.0
    smooth_b = smooth_values(bfactors, n_orig)

    # Draw segment by segment
    for i in range(len(smooth_c) - 1):
        val = smooth_b[i]
        normed = (val - PLDDT_MIN) / (PLDDT_MAX - PLDDT_MIN)
        color = PLDDT_CMAP(np.clip(normed, 0, 1))
        ax.plot(
            [smooth_c[i, 0], smooth_c[i+1, 0]],
            [smooth_c[i, 1], smooth_c[i+1, 1]],
            [smooth_c[i, 2], smooth_c[i+1, 2]],
            color=color, linewidth=linewidth, alpha=1.0,
            solid_capstyle='round'
        )

    # Invisible line for legend
    if label:
        ax.plot([], [], [], color=PLDDT_CMAP(0.8), linewidth=linewidth, label=label)


def plot_backbone_solid(ax, coords, color='#AAAAAA', linewidth=3.5, alpha=0.45, label=None):
    """Plot a solid-color smoothed backbone trace."""
    smooth_c, _ = smooth_backbone(coords)
    ax.plot(smooth_c[:, 0], smooth_c[:, 1], smooth_c[:, 2],
            color=color, linewidth=linewidth, alpha=alpha, label=label,
            solid_capstyle='round')


def set_equal_aspect(ax, all_coords):
    """Set equal aspect ratio for 3D plot with tight framing."""
    stacked = np.vstack(all_coords)
    center = stacked.mean(axis=0)
    max_range = (stacked.max(axis=0) - stacked.min(axis=0)).max() / 2.0
    margin = max_range * 0.72  # very tight crop
    ax.set_xlim(center[0] - margin, center[0] + margin)
    ax.set_ylim(center[1] - margin, center[1] + margin)
    ax.set_zlim(center[2] - margin, center[2] + margin)


def clean_axes(ax):
    """Remove all axis decoration."""
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.xaxis.line.set_visible(False)
    ax.yaxis.line.set_visible(False)
    ax.zaxis.line.set_visible(False)
    ax.grid(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')


def render_panel(ax, ref_coords, aligned_coords, aligned_bfactors, tm_score,
                 ref_label, sa_label, elev, azim, panel_label, show_legend=False):
    """Render one panel: reference (gray) + SA variant (pLDDT-colored)."""
    # Reference behind
    plot_backbone_solid(ax, ref_coords, color='#AAAAAA', linewidth=4.0, alpha=0.45,
                       label=ref_label)

    # SA variant on top, colored by pLDDT
    plot_backbone_plddt(ax, aligned_coords, aligned_bfactors, linewidth=2.5,
                       label=sa_label)

    # N/C termini
    ax.scatter(*ref_coords[0], color='#666666', s=40, zorder=10,
              edgecolors='white', linewidths=0.5, alpha=0.6)
    ax.scatter(*ref_coords[-1], color='#666666', s=40, zorder=10,
              edgecolors='white', linewidths=0.5, alpha=0.6)
    offset = 2.0
    ax.text(ref_coords[0][0], ref_coords[0][1], ref_coords[0][2] + offset, 'N',
            fontsize=7, fontweight='bold', ha='center', color='#555555', zorder=11)
    ax.text(ref_coords[-1][0], ref_coords[-1][1], ref_coords[-1][2] + offset, 'C',
            fontsize=7, fontweight='bold', ha='center', color='#555555', zorder=11)

    all_c = [ref_coords, aligned_coords]
    set_equal_aspect(ax, all_c)
    ax.view_init(elev=elev, azim=azim)
    ax.dist = 6.5  # zoom camera in (default is 10)
    clean_axes(ax)

    # Panel label
    ax.text2D(0.02, 0.95, panel_label, transform=ax.transAxes,
             fontsize=14, fontweight='bold', va='top')

    if show_legend:
        leg = ax.legend(loc='upper right', fontsize=6.5, framealpha=0.9,
                       handlelength=1.8, borderpad=0.4, edgecolor='#CCCCCC')
        leg.get_frame().set_linewidth(0.5)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- Data ----
    # Kunitz: best SA variant by TM-score
    kunitz_ref_pdb = os.path.join(BASE, "code/data/kunitz/structures/1BPI_A.pdb")
    kunitz_sa_pdb = os.path.join(BASE, "code/data/kunitz/structures/SA_strong_SA_strong_0022.pdb")

    # Conotoxin: best SA variant by TM-score
    conot_ref_pdb = os.path.join(BASE, "code/data/omega_conotoxin/structures/1OMG_A.pdb")
    conot_sa_pdb = os.path.join(BASE, "code/data/omega_conotoxin/structures/SA_strong_SA_strong_0024.pdb")

    # ---- Align ----
    print("=== Kunitz ===")
    kunitz_ref, _ = extract_ca_coords_and_bfactors(kunitz_ref_pdb)
    kunitz_aligned, kunitz_bf, kunitz_tm = run_tmalign_and_get_aligned(kunitz_sa_pdb, kunitz_ref_pdb)
    k_plddt = kunitz_bf * 100 if kunitz_bf.max() <= 1.0 else kunitz_bf
    print(f"  SA_strong_0022: TM = {kunitz_tm:.4f}, mean pLDDT = {k_plddt.mean():.1f}")

    print("\n=== Conotoxin ===")
    conot_ref, _ = extract_ca_coords_and_bfactors(conot_ref_pdb)
    conot_aligned, conot_bf, conot_tm = run_tmalign_and_get_aligned(conot_sa_pdb, conot_ref_pdb)
    c_plddt = conot_bf * 100 if conot_bf.max() <= 1.0 else conot_bf
    print(f"  SA_strong_0024: TM = {conot_tm:.4f}, mean pLDDT = {c_plddt.mean():.1f}")

    # ---- Figure: 2x2 grid, compact ----
    fig = plt.figure(figsize=(10, 8))

    # Use gridspec for tight control
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, figure=fig,
                          left=-0.12, right=1.12, top=0.93, bottom=0.13,
                          wspace=-0.30, hspace=-0.05)

    # Top row: Kunitz (A, B)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    render_panel(ax1, kunitz_ref, kunitz_aligned, kunitz_bf, kunitz_tm,
                 "BPTI (1BPI, exp.)", f"SA strong (TM = {kunitz_tm:.2f})",
                 elev=15, azim=-65, panel_label='A', show_legend=False)

    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    render_panel(ax2, kunitz_ref, kunitz_aligned, kunitz_bf, kunitz_tm,
                 "BPTI (1BPI, exp.)", f"SA strong (TM = {kunitz_tm:.2f})",
                 elev=15, azim=25, panel_label='B')

    # Bottom row: Conotoxin (C, D)
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    render_panel(ax3, conot_ref, conot_aligned, conot_bf, conot_tm,
                 "MVIIA (1OMG, NMR)", f"SA strong (TM = {conot_tm:.2f})",
                 elev=15, azim=-65, panel_label='C', show_legend=False)

    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    render_panel(ax4, conot_ref, conot_aligned, conot_bf, conot_tm,
                 "MVIIA (1OMG, NMR)", f"SA strong (TM = {conot_tm:.2f})",
                 elev=15, azim=25, panel_label='D')

    # Row labels
    fig.text(0.50, 0.96, 'Kunitz Domain', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.50, 0.50, r'$\omega$-Conotoxin', ha='center', fontsize=12, fontweight='bold')

    # pLDDT colorbar
    cbar_ax = fig.add_axes([0.25, 0.045, 0.50, 0.018])
    norm = matplotlib.colors.Normalize(vmin=PLDDT_MIN, vmax=PLDDT_MAX)
    sm = matplotlib.cm.ScalarMappable(cmap=PLDDT_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('pLDDT', fontsize=10, fontweight='bold')
    cbar.set_ticks([50, 60, 70, 80, 90, 100])
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(os.path.join(FIG_DIR, "fold_superposition_combined.pdf"), dpi=300, facecolor='white')
    plt.close()
    print(f"\nSaved: {os.path.join(FIG_DIR, 'fold_superposition_combined.pdf')}")
