import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# =========================
# CONFIG
# =========================
file_path = "pref_simplex_gpi_minecart.csv"
# file_path = "pref_simplex_env_minecart.csv"
# file_path = "pref_simplex_random_minecart.csv"
# file_path = "pref_simplex_envelope_minecart_interDiffSampleEnv.csv"

right_angle = True      # True = (t,s), False = barycentric simplex
show_points = True
show_contours = False

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(file_path)

W = df[["w0", "w1", "w2"]].values
R = df[["r0", "r1", "r2"]].values

# numerical safety
W = W / (W.sum(axis=1, keepdims=True) + 1e-12)

ts = df["t"].values
ss = df["s"].values

# =========================
# PROJECTIONS
# =========================
def barycentric_3d_to_2d(w):
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    return (
        w[:, 0:1] * v0 +
        w[:, 1:2] * v1 +
        w[:, 2:3] * v2
    )

def right_angle_2d(t, s):
    return np.stack([t, s], axis=1)

# =========================
# CHOOSE PROJECTION
# =========================
if right_angle:
    XY = right_angle_2d(ts, ss)
    title = "(t,s)"
else:
    XY = barycentric_3d_to_2d(W)
    title = "Preference Simplex"

x = XY[:, 0]
y = XY[:, 1]

tri = Triangulation(x, y)

# =========================
# PLOT
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

obj_cols = ["r0", "r1", "r2"]

for i, col in enumerate(obj_cols):

    ax = axes[i]
    # for spine in ax.spines.values():
    #     spine.set_visible(False)
    vals = R[:, i]

    # -------------------------
    # Contours
    # -------------------------
    if show_contours:
        cf = ax.tricontourf(tri, vals, levels=20, cmap="viridis")

        ax.tricontour(
            tri,
            vals,
            levels=10,
            colors="black",
            linewidths=0.5,
            alpha=0.5,
        )

        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(col)

    # -------------------------
    # Normalisation (FIXED)
    # -------------------------
    min_val = vals.min()
    max_val = vals.max()

    # if i in [0, 1]:  # maximise objectives
    vals_clip = np.clip(vals, np.percentile(vals, 10), np.percentile(vals, 95))
    vals_norm = (vals_clip - vals_clip.min()) / (vals_clip.max() - vals_clip.min() + 1e-8)
    # vals_norm = (vals - min_val) / (max_val - min_val + 1e-8)
    # else:  # minimise objective (r2)
        # vals_norm = (max_val - vals) / (max_val - min_val + 1e-8)
    # for i in range(len(vals_norm)):
        # print(vals[i], vals_norm[i])

    # -------------------------
    # Scatter plot
    # -------------------------
    alpha = np.where(np.isclose(vals, 0.0), 1.0, 0.6)

    sc = ax.scatter(
        x,
        y,
        # c=vals,
        c=vals_norm,
        cmap="viridis",
        s=40,
        alpha=alpha,
        vmin=0.0,
        vmax=1.0,
    )

    ticks = [0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    ax.set_yticklabels([f"{t:.2f}" for t in ticks])
    plt.colorbar(sc, ax=ax).set_label(col)

    # -------------------------
    # Simplex boundary
    # -------------------------
    if not right_angle:
        v0 = np.array([0.0, 0.0])              # w0 = 1
        v1 = np.array([1.0, 0.0])              # w1 = 1
        v2 = np.array([0.5, np.sqrt(3) / 2])   # w2 = 1

        simplex = np.array([v0, v1, v2, v0])
        ax.plot(simplex[:, 0], simplex[:, 1], "k-", lw=0.5,alpha=0.5)

        # vertex labels
        ax.text(v0[0]-0.04, v0[1] - 0.04, "w0=1")
        ax.text(v1[0], v1[1] - 0.04, "w1=1")
        ax.text(v2[0], v2[1] + 0.02, "w2=1")

        ticks = [0.25, 1/3, 0.5, 2/3, 0.75]
        for t in ticks:

            # ======================
            # GRIDLINES
            # ======================

            # constant w0 = t
            p1 = t * v0 + (1 - t) * v1
            p2 = t * v0 + (1 - t) * v2
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="black",
                lw=0.5,
                alpha=0.5,
            )

            # constant w1 = t
            p1 = t * v1 + (1 - t) * v0
            p2 = t * v1 + (1 - t) * v2
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="grey",
                lw=0.5,
                alpha=0.4,
            )

            # constant w2 = t
            p1 = t * v2 + (1 - t) * v0
            p2 = t * v2 + (1 - t) * v1
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="grey",
                lw=0.5,
                alpha=0.4,
            )

            # ======================
            # TICK LABELS
            # ======================

            # w2 ticks along bottom edge
            p = (1 - t) * v0 + t * v1
            ax.text(
                p[0],
                p[1] - 0.045,
                f"{t:.1f}",
                ha="center",
                fontsize=8,
            )

            # w1 ticks along left edge
            p = (1 - t) * v0 + t * v2
            ax.text(
                p[0] - 0.04,
                p[1],
                f"{t:.1f}",
                ha="right",
                va="center",
                fontsize=8,
            )

            # w0 ticks along right edge
            p = (1 - t) * v1 + t * v2
            ax.text(
                p[0] + 0.04,
                p[1],
                f"{t:.1f}",
                ha="left",
                va="center",
                fontsize=8,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(f"{title} → {col}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    if right_angle:
        ax.set_xlabel("t (w0)")
        ax.set_ylabel("s (w1)")

plt.tight_layout()

out_file = (
    f"pref_simplex{file_path.partition('pref_simplex')[-1]}"
    .replace(".csv", ".png")
)

print(out_file)
plt.savefig(out_file, dpi=300)
plt.show()

# =========================
# PARALLEL COORDINATES
# =========================
df_sorted = df.sort_values("w0") if "w0" in df.columns else df

plt.figure(figsize=(8, 5))

for i in range(len(df_sorted)):
    plt.plot(obj_cols, df_sorted[obj_cols].iloc[i].values, alpha=0.3)

plt.ylabel("Reward")
plt.title("Objective trade-offs (parallel coordinates)")
plt.grid(True, alpha=0.2)

out_file = (
    f"pref_simplex{file_path.partition('reward_track')[-1]}"
    .replace(".csv", ".png")
)

plt.savefig(out_file, dpi=300)
# plt.show()