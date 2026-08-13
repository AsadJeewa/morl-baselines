import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# =========================
# CONFIG
# =========================
file_pattern = "pref_line_*.csv"
files = glob.glob(file_pattern)

# =========================
# GET REWARD COLS (ASSUME CONSISTENT)
# =========================
r_cols = [c for c in pd.read_csv(files[0]).columns if c.startswith("r")]

# =========================
# GLOBAL MIN / MAX (ACROSS ALL FILES)
# =========================
all_vals = []

for f in files:
    df_tmp = pd.read_csv(f)
    all_vals.append(df_tmp[r_cols].values)

all_vals = np.concatenate(all_vals, axis=0)

r_min = all_vals.min(axis=0)
r_max = all_vals.max(axis=0)

# =========================
# PLOT
# =========================
plt.figure(figsize=(9, 6))

for file_path in files:

    df = pd.read_csv(file_path)

    t = df["t"].values
    idx = np.argsort(t)

    base = os.path.basename(file_path).replace(".csv", "")
    label = base.replace("pref_line_", "").replace("_default", "")

    for i, r in enumerate(r_cols):

        vals = df[r].values

        # =========================
        # NORMALISE USING GLOBAL MIN/MAX
        # =========================
        vals_norm = (vals - r_min[i]) / (r_max[i] - r_min[i] + 1e-8)

        plt.plot(
            t[idx],
            # vals_norm[idx],
            vals[idx],
            linewidth=2,
            label=f"{label}_{r}"
        )

# =========================
# CLEAN LEGEND (NO DUPLICATES)
# =========================
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys())

plt.xlabel("t (w0)")
plt.ylabel("Normalised Return")
plt.title("Preference Lines Across Algorithms (Normalised)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pref_lines_normalised.png", dpi=300)
plt.show()