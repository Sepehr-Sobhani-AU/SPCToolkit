"""
Generates ``surface_region_growing_algorithm.png`` — the pedagogical figure
embedded in SURFACE_REGION_GROWING.md.

Run from anywhere::

    python plugins/020_Points/020_Clustering/_make_algorithm_figure.py

The figure is a 3D synthetic analog of the algorithm: one slightly-curved
planar surface plus clutter, on a uniform voxel grid, viewed isometrically.
Four panels show setup, boundary detection, the per-voxel RANSAC + distance
filter, and the converged surface.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


VOXEL_SIZE = 1.0
WORLD = (0.0, 8.0, 0.0, 8.0, 1.0, 5.0)  # xmin, xmax, ymin, ymax, zmin, zmax
DIST_THRESHOLD = 0.25
# Isometric — flatter than the canonical 35.26° so the planar surface reads better.
ISO_ELEV = 24.0
ISO_AZIM = -55.0

# Colours
COL_SEED = "#16a34a"          # green — current surface
COL_NEW = "#bbf7d0"            # very pale green — accepted candidate-voxel points
COL_NEW_EDGE = "#15803d"       # dark green outline
COL_CLUTTER = "#94a3b8"        # slate gray — non-surface points
COL_BOUNDARY = "#16a34a"
COL_CANDIDATE = "#facc15"      # yellow — candidate voxel fill
COL_INTERIOR = "#cbd5e1"       # pale gray — interior seed voxel
COL_PLANE = "#dc2626"          # red — fitted plane / surface line
COL_NORMAL = "#7c3aed"         # purple — reference normal arrow


# -- Scene synthesis -------------------------------------------------------


def surface_z(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Slightly tilted + curved planar surface used as the 'true' surface."""
    return 3.0 + 0.03 * (x - 4.0) + 0.05 * (y - 4.0) + 0.2 * np.sin(0.45 * x)


def build_scene(rng: np.random.Generator):
    """Return (surface_points, clutter_points, seed_mask) all shape (N, 3)."""
    # Surface samples (denser than rendered — most stay in the background)
    n_surf = 700
    x_s = rng.uniform(0.4, 7.6, n_surf)
    y_s = rng.uniform(0.4, 7.6, n_surf)
    z_s = surface_z(x_s, y_s) + rng.normal(0, 0.04, n_surf)
    surf = np.column_stack([x_s, y_s, z_s])

    # Clutter: scattered noise points away from the surface.
    n_clut = 160
    x_c = rng.uniform(0.4, 7.6, n_clut)
    y_c = rng.uniform(0.4, 7.6, n_clut)
    z_c = rng.uniform(1.2, 4.8, n_clut)
    clut = np.column_stack([x_c, y_c, z_c])
    keep = np.abs(clut[:, 2] - surface_z(clut[:, 0], clut[:, 1])) > 0.5
    clut = clut[keep]

    # Seed = surface samples in the middle 3×3 voxel-region
    seed_mask = (
        (surf[:, 0] >= 3.0) & (surf[:, 0] < 6.0)
        & (surf[:, 1] >= 3.0) & (surf[:, 1] < 6.0)
    )
    return surf, clut, seed_mask


def voxel_of(pt: np.ndarray) -> tuple[int, int, int]:
    k = np.floor(pt / VOXEL_SIZE).astype(int)
    return (int(k[0]), int(k[1]), int(k[2]))


def voxel_keys(pts: np.ndarray) -> set[tuple[int, int, int]]:
    return {voxel_of(p) for p in pts}


def neighbours26(v):
    return [
        (v[0] + dx, v[1] + dy, v[2] + dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]


# -- 3D drawing primitives -------------------------------------------------


def cube_faces(vx: int, vy: int, vz: int, size: float = VOXEL_SIZE):
    """Six quad faces of a voxel cube as a list of (4,3) ndarrays."""
    x0, y0, z0 = vx * size, vy * size, vz * size
    x1, y1, z1 = x0 + size, y0 + size, z0 + size
    return [
        np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]),  # bottom
        np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),  # top
        np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),  # y0 face
        np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]),  # y1 face
        np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]),  # x1 face
        np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]),  # x0 face
    ]


def draw_voxel(ax, vx, vy, vz, color, alpha=0.22, edge_color=None, lw=0.6):
    faces = cube_faces(vx, vy, vz)
    pc = Poly3DCollection(
        faces,
        facecolor=color,
        alpha=alpha,
        edgecolor=edge_color if edge_color is not None else (0.4, 0.4, 0.4, 0.5),
        linewidth=lw,
    )
    ax.add_collection3d(pc)


def style_axes(ax, title: str) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = WORLD
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ISO_ELEV, azim=ISO_AZIM)
    ax.set_title(title, fontsize=11, loc="left", pad=2)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    # Lighten the panes
    for pane_attr in ("xaxis", "yaxis", "zaxis"):
        ax_attr = getattr(ax, pane_attr)
        ax_attr.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax_attr.pane.set_edgecolor((0.7, 0.7, 0.7, 0.25))
    # Hide axis spines/labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


# -- Panels ----------------------------------------------------------------


def panel_setup(ax, surf, clut, seed_mask) -> None:
    seed_pts = surf[seed_mask]
    rest = np.vstack([surf[~seed_mask], clut])

    ax.scatter(rest[:, 0], rest[:, 1], rest[:, 2], s=2, c=COL_CLUTTER, alpha=0.45)
    ax.scatter(seed_pts[:, 0], seed_pts[:, 1], seed_pts[:, 2],
               s=6, c=COL_SEED, alpha=0.95)

    # Picked point near the middle of the seed
    target = np.array([4.5, 4.5, surface_z(np.array([4.5]), np.array([4.5]))[0]])
    picked_idx = int(np.argmin(np.linalg.norm(seed_pts - target, axis=1)))
    picked = seed_pts[picked_idx]
    ax.scatter([picked[0]], [picked[1]], [picked[2]],
               s=160, marker="*", edgecolors="black",
               facecolor="#fde68a", linewidths=1.0, zorder=10)

    # Reference normal (approximate surface gradient × tangent)
    eps = 0.1
    zx = (surface_z(np.array([picked[0] + eps]), np.array([picked[1]]))[0]
          - surface_z(np.array([picked[0] - eps]), np.array([picked[1]]))[0]) / (2 * eps)
    zy = (surface_z(np.array([picked[0]]), np.array([picked[1] + eps]))[0]
          - surface_z(np.array([picked[0]]), np.array([picked[1] - eps]))[0]) / (2 * eps)
    normal = np.array([-zx, -zy, 1.0])
    normal /= np.linalg.norm(normal)
    arrow_len = 1.8
    ax.quiver(
        picked[0], picked[1], picked[2],
        normal[0] * arrow_len, normal[1] * arrow_len, normal[2] * arrow_len,
        color=COL_NORMAL, linewidth=2.0, arrow_length_ratio=0.22,
    )
    ax.text(
        picked[0] + normal[0] * arrow_len + 0.15,
        picked[1] + normal[1] * arrow_len + 0.15,
        picked[2] + normal[2] * arrow_len + 0.2,
        "ref normal", color=COL_NORMAL, fontsize=9,
    )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_SEED,
               markersize=7, label="Seed cluster"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_CLUTTER,
               markersize=6, label="Rest of cloud"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#fde68a",
               markeredgecolor="black", markersize=12, label="Picked point"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.95)
    style_axes(ax, "1. Setup: seed cluster + reference normal")


def panel_boundary(ax, surf, clut, seed_mask) -> None:
    seed_pts = surf[seed_mask]
    rest_pts = np.vstack([surf[~seed_mask], clut])

    seed_vox = voxel_keys(seed_pts)
    unassigned_vox = voxel_keys(rest_pts)
    boundary_vox = {
        v for v in seed_vox
        if any(n in unassigned_vox for n in neighbours26(v))
    }
    interior_vox = seed_vox - boundary_vox
    candidate_vox = {
        n for v in boundary_vox
        for n in neighbours26(v)
        if n in unassigned_vox
    }

    # Draw furthest first (rough painter's ordering for isometric view)
    cam_dir = np.array([np.cos(np.deg2rad(ISO_AZIM)) * np.cos(np.deg2rad(ISO_ELEV)),
                        np.sin(np.deg2rad(ISO_AZIM)) * np.cos(np.deg2rad(ISO_ELEV)),
                        np.sin(np.deg2rad(ISO_ELEV))])

    def sort_key(v):
        return -np.dot(np.array(v) + 0.5, cam_dir)

    for v in sorted(candidate_vox, key=sort_key):
        draw_voxel(ax, *v, COL_CANDIDATE, alpha=0.22, edge_color=(0.7, 0.5, 0.1, 0.6))
    for v in sorted(interior_vox, key=sort_key):
        draw_voxel(ax, *v, COL_INTERIOR, alpha=0.30, edge_color=(0.5, 0.5, 0.5, 0.5))
    for v in sorted(boundary_vox, key=sort_key):
        draw_voxel(ax, *v, COL_BOUNDARY, alpha=0.30, edge_color=(0.1, 0.4, 0.2, 0.7))

    ax.scatter(rest_pts[:, 0], rest_pts[:, 1], rest_pts[:, 2],
               s=2, c=COL_CLUTTER, alpha=0.35)
    ax.scatter(seed_pts[:, 0], seed_pts[:, 1], seed_pts[:, 2],
               s=6, c=COL_SEED, alpha=0.9)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_BOUNDARY,
               alpha=0.35, markersize=12, label="Boundary voxel"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_INTERIOR,
               alpha=0.4, markersize=12, label="Interior seed voxel"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_CANDIDATE,
               alpha=0.35, markersize=12, label="Candidate voxel"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.95)
    style_axes(ax, "2. Find boundary voxels + their unassigned neighbours")


def panel_ransac(ax, surf, clut, seed_mask, rng) -> None:
    seed_pts = surf[seed_mask]

    # Pick a boundary voxel on the +x edge of the seed and a candidate to its right.
    bvx, bvy = 5, 4
    bvz = int(np.floor(surface_z(np.array([bvx + 0.5]), np.array([bvy + 0.5]))[0]))
    boundary_v = (bvx, bvy, bvz)
    candidate_v = (bvx + 1, bvy, bvz)

    # Fit plane to surface points inside the boundary voxel
    bx0, by0, bz0 = bvx * VOXEL_SIZE, bvy * VOXEL_SIZE, bvz * VOXEL_SIZE
    in_b = (
        (surf[:, 0] >= bx0) & (surf[:, 0] < bx0 + VOXEL_SIZE)
        & (surf[:, 1] >= by0) & (surf[:, 1] < by0 + VOXEL_SIZE)
        & (surf[:, 2] >= bz0) & (surf[:, 2] < bz0 + VOXEL_SIZE)
    )
    b_pts = surf[in_b]
    centroid = b_pts.mean(axis=0)
    centred = b_pts - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    plane_normal = vh[-1]
    if plane_normal[2] < 0:
        plane_normal = -plane_normal

    # Draw a finite plane patch spanning the boundary + candidate voxels.
    cx0 = candidate_v[0] * VOXEL_SIZE
    xs = np.linspace(bx0 - 0.05, cx0 + VOXEL_SIZE + 0.05, 8)
    ys = np.linspace(by0 - 0.05, by0 + VOXEL_SIZE + 0.05, 8)
    X, Y = np.meshgrid(xs, ys)
    # n · (x - c) = 0  →  z = c_z - (n_x (x-c_x) + n_y (y-c_y)) / n_z
    Z = (centroid[2]
         - (plane_normal[0] * (X - centroid[0])
            + plane_normal[1] * (Y - centroid[1])) / plane_normal[2])
    ax.plot_surface(X, Y, Z, color=COL_PLANE, alpha=0.30, edgecolor=COL_PLANE,
                    linewidth=0.3, antialiased=True, shade=False)

    # Distance-threshold slab: two parallel translucent patches.
    offset = plane_normal * DIST_THRESHOLD
    for sgn in (+1, -1):
        ax.plot_surface(X + sgn * offset[0], Y + sgn * offset[1], Z + sgn * offset[2],
                        color=COL_PLANE, alpha=0.10, edgecolor="none", shade=False)

    # Voxels (boundary + candidate)
    draw_voxel(ax, *boundary_v, COL_BOUNDARY, alpha=0.20,
               edge_color=(0.1, 0.4, 0.2, 0.7))
    draw_voxel(ax, *candidate_v, COL_CANDIDATE, alpha=0.20,
               edge_color=(0.7, 0.5, 0.1, 0.7))

    # Classify candidate-voxel points by perpendicular distance to plane
    cy0 = bvy * VOXEL_SIZE
    in_c = (
        (surf[:, 0] >= cx0) & (surf[:, 0] < cx0 + VOXEL_SIZE)
        & (surf[:, 1] >= cy0) & (surf[:, 1] < cy0 + VOXEL_SIZE)
        & (surf[:, 2] >= bz0) & (surf[:, 2] < bz0 + VOXEL_SIZE)
    )
    c_surf_pts = surf[in_c]
    # also any nearby clutter in same voxel for context
    in_c_clut = (
        (clut[:, 0] >= cx0) & (clut[:, 0] < cx0 + VOXEL_SIZE)
        & (clut[:, 1] >= cy0) & (clut[:, 1] < cy0 + VOXEL_SIZE)
        & (clut[:, 2] >= bz0) & (clut[:, 2] < bz0 + VOXEL_SIZE)
    )
    c_clut_pts = clut[in_c_clut]
    candidate_pts = np.vstack([c_surf_pts, c_clut_pts]) if len(c_clut_pts) else c_surf_pts

    dists = (candidate_pts - centroid) @ plane_normal
    accepted = candidate_pts[np.abs(dists) <= DIST_THRESHOLD]
    rejected = candidate_pts[np.abs(dists) > DIST_THRESHOLD]

    # Context: everything except the boundary + candidate voxels
    other_mask = ~(in_b | in_c)
    other_surf = surf[other_mask]
    ax.scatter(other_surf[:, 0], other_surf[:, 1], other_surf[:, 2],
               s=2, c=COL_CLUTTER, alpha=0.20)
    other_clut_mask = ~in_c_clut
    other_clut = clut[other_clut_mask]
    ax.scatter(other_clut[:, 0], other_clut[:, 1], other_clut[:, 2],
               s=2, c=COL_CLUTTER, alpha=0.20)

    # Boundary surface points
    ax.scatter(b_pts[:, 0], b_pts[:, 1], b_pts[:, 2],
               s=16, c=COL_SEED, edgecolors="black", linewidths=0.4, zorder=8)
    # Accepted candidate points stand out — pale fill + dark green outline
    if len(accepted) > 0:
        ax.scatter(accepted[:, 0], accepted[:, 1], accepted[:, 2],
                   s=42, c=COL_NEW, edgecolors=COL_NEW_EDGE, linewidths=1.0, zorder=9)
    if len(rejected) > 0:
        ax.scatter(rejected[:, 0], rejected[:, 1], rejected[:, 2],
                   s=18, c=COL_CLUTTER, edgecolors="black", linewidths=0.4, zorder=8)

    handles = [
        Line2D([0], [0], color=COL_PLANE, lw=2, label="Fitted plane"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COL_PLANE,
               alpha=0.3, markersize=12, label="±distance_threshold"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_NEW,
               markeredgecolor=COL_NEW_EDGE, markersize=9, label="Accepted points"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.95)
    style_axes(ax, "3. Fit plane; accept candidate points within ±threshold")


def panel_converged(ax, surf, clut) -> None:
    ax.scatter(clut[:, 0], clut[:, 1], clut[:, 2],
               s=2, c=COL_CLUTTER, alpha=0.5, label="Non-surface (label = -1)")
    ax.scatter(surf[:, 0], surf[:, 1], surf[:, 2],
               s=6, c=COL_NEW, edgecolors=COL_SEED, linewidths=0.2,
               alpha=0.95, label="Surface (label = 0)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    style_axes(ax, "4. Converged: full surface grown across the cloud")


# -- Driver ----------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(11)
    surf, clut, seed_mask = build_scene(rng)

    fig = plt.figure(figsize=(15.0, 11.0), dpi=140)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    panel_setup(ax1, surf, clut, seed_mask)
    panel_boundary(ax2, surf, clut, seed_mask)
    panel_ransac(ax3, surf, clut, seed_mask, rng)
    panel_converged(ax4, surf, clut)

    fig.suptitle(
        "Surface Region Growing — 3D schematic of the four algorithm phases",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(__file__).with_name("surface_region_growing_algorithm.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
