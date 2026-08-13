import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np


def pareto_front_2d(points):
    is_pareto = np.ones(points.shape[0], dtype=bool)

    for i, p in enumerate(points):
        if not is_pareto[i]:
            continue

        dominates = np.all(points >= p, axis=1) & np.any(points > p, axis=1)
        if np.any(dominates):
            is_pareto[i] = False

    return points[is_pareto]


def plot_pareto_from_csv(folder=".", pattern="pref_line_gpi*.csv"):
    files = glob.glob(os.path.join(folder, pattern))

    plt.figure(figsize=(7, 6))

    for f in files:
        df = pd.read_csv(f)
        algo = df["algo"].iloc[0]

        r_cols = [c for c in df.columns if c.startswith("r")]
        if len(r_cols) < 2:
            continue

        points = df[[r_cols[0], r_cols[1]]].values

        pf = pareto_front_2d(points)

        plt.scatter(points[:, 0], points[:, 1], alpha=0.3, label=f"{algo} (all)")
        plt.scatter(pf[:, 0], pf[:, 1], label=f"{algo} (Pareto)", s=60)

    plt.xlabel("r0")
    plt.ylabel("r1")
    plt.title("Pareto Front (Non-dominated points)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pareto_front.png")
    plt.show()


if __name__ == "__main__":
    plot_pareto_from_csv("./")