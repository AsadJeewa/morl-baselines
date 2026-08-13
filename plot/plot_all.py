import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


# ----------------------------
# LINE PLOTS
# ----------------------------
def plot_line_all(files):
    plt.figure(figsize=(8, 6))

    for f in files:
        df = pd.read_csv(f)
        algo = df["algo"].iloc[0]
        r_cols = [c for c in df.columns if c.startswith("r")]

        for i, r in enumerate(r_cols):
            plt.plot(df["t"], df[r], label=f"{algo}-obj{i}")

    plt.xlabel("t (preference)")
    plt.ylabel("Return")
    plt.title("1D Preference Line (All)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pref_line_all.png")
    plt.show()


def plot_line_per_objective(files):
    # detect objectives
    df0 = pd.read_csv(files[0])
    r_cols = [c for c in df0.columns if c.startswith("r")]

    for obj_idx, r in enumerate(r_cols):
        plt.figure(figsize=(8, 5))

        for f in files:
            df = pd.read_csv(f)
            algo = df["algo"].iloc[0]

            plt.plot(df["t"], df[r], label=algo)

        plt.title(f"Objective {obj_idx}")
        plt.xlabel("t (preference)")
        plt.ylabel("Return")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"pref_line_obj{obj_idx}.png")
        plt.show()


# ----------------------------
# SIMPLEX PLOTS
# ----------------------------
def plot_simplex_all(files):
    fig, ax = plt.subplots(figsize=(8, 6))

    for f in files:
        df = pd.read_csv(f)
        algo = df["algo"].iloc[0]
        r_cols = [c for c in df.columns if c.startswith("r")]

        for i, r in enumerate(r_cols):
            sc = ax.scatter(
                df["t"],
                df["s"],
                c=df[r],
                alpha=0.6,
                label=f"{algo}-obj{i}"
            )

    ax.set_xlabel("t")
    ax.set_ylabel("s")
    ax.set_title("Simplex Preference Space (All)")
    plt.colorbar(sc, ax=ax)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pref_simplex_all.png")
    plt.show()


def plot_simplex_per_objective(files):
    df0 = pd.read_csv(files[0])
    r_cols = [c for c in df0.columns if c.startswith("r")]

    for obj_idx, r in enumerate(r_cols):
        plt.figure(figsize=(7, 6))

        for f in files:
            df = pd.read_csv(f)
            algo = df["algo"].iloc[0]

            sc = plt.scatter(
                df["t"],
                df["s"],
                c=df[r],
                alpha=0.6,
                label=algo
            )

        plt.xlabel("t")
        plt.ylabel("s")
        plt.title(f"Simplex - Objective {obj_idx}")
        plt.colorbar(sc)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"pref_simplex_obj{obj_idx}.png")
        plt.show()


# ----------------------------
# DRIVER
# ----------------------------
def plot_all_csvs(folder=".", pattern="pref_*.csv"):
    files = glob.glob(os.path.join(folder, pattern))

    if not files:
        print("No CSV files found.")
        return

    line_files = []
    simplex_files = []

    for f in files:
        df = pd.read_csv(f)

        if "s" not in df.columns or df["s"].isna().all():
            line_files.append(f)
        else:
            simplex_files.append(f)

    if line_files:
        plot_line_all(line_files)
        plot_line_per_objective(line_files)

    # if simplex_files:
    #     plot_simplex_all(simplex_files)
    #     plot_simplex_per_objective(simplex_files)


if __name__ == "__main__":
    plot_all_csvs("./")