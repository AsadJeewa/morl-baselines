import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("hl_action_log.csv")
steps = df["Step"]
actions = df["HighLevelAction"]

# Step plot
plt.figure(figsize=(12,3))
plt.step(steps, actions, where='post', linewidth=2)
plt.yticks([0,1,2])
plt.xlabel("Step")
plt.ylabel("High-Level Policy Choice")
plt.title("High-Level Controller Choices Over Time")
plt.grid(True, linestyle='--', alpha=0.5)

# Save figure
plt.savefig("hl_action_step_plot.png", dpi=300)
plt.show()