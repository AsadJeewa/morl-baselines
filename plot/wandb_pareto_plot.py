import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 👇 replace with your actual filename
filename = "plot/wandb_export_2026-06-10T14_48_11.189+02_00.csv"

df = pd.read_csv(filename)
num_cols = len(df.columns)

fig = plt.figure()

if num_cols >= 3:
    # 3D Plotting
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], s=40)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
    plt.title("3D Pareto Front")
elif num_cols == 2:
    # 2D Plotting fallback
    ax = fig.add_subplot(111)
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], s=40)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    plt.title("2D Pareto Front")
else:
    print(f"Error: Dataset only has {num_cols} column(s). Visualisation requires at least 2.")

plt.show()