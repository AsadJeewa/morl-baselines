import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from mpl_toolkits.mplot3d import Axes3D


def clean_label(path):
    name = os.path.basename(path).replace(".csv", "")
    name = name.replace("default", "")
    name = name.replace("wandb_export_", "")
    return name


def plot_wandb_pareto(files,
                      out_file_2d="pareto_compare_2d.png",
                      out_file_3d="pareto_compare_3d.png"):

    fig2d, ax2d = plt.subplots(figsize=(8, 6))
    has_2d = False

    fig3d = plt.figure(figsize=(9, 7))
    ax3d = fig3d.add_subplot(111, projection="3d")
    has_3d = False

    for f in files:

        df = pd.read_csv(f)

        obj_cols = [c for c in df.columns if c.startswith("objective_")]
        n_obj = len(obj_cols)

        if n_obj == 2:

            pts = df[obj_cols].astype(float).values

            ax2d.scatter(
                pts[:, 0],
                pts[:, 1],
                s=30,
                alpha=0.7,
                label=clean_label(f)
            )

            has_2d = True

        elif n_obj == 3:

            pts = df[obj_cols].astype(float).values

            ax3d.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=20,
                alpha=0.7,
                label=clean_label(f)
            )

            has_3d = True

        else:
            print(f"Skipping {f}: {n_obj} objectives")

    if has_2d:
        ax2d.set_xlabel("Objective 1")
        ax2d.set_ylabel("Objective 2")
        ax2d.set_title("2D Pareto Front Comparison")
        ax2d.grid(True)
        ax2d.legend()
        fig2d.tight_layout()
        fig2d.savefig(out_file_2d, dpi=300)

    if has_3d:
        ax3d.set_xlabel("Objective 1")
        ax3d.set_ylabel("Objective 2")
        ax3d.set_zlabel("Objective 3")
        ax3d.set_title("3D Pareto Front Comparison")
        ax3d.legend()
        fig3d.tight_layout()
        fig3d.savefig(out_file_3d, dpi=300)

    plt.show()


if __name__ == "__main__":
    files = glob.glob("wandb_export_*.csv")
    plot_wandb_pareto(files)