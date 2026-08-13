import pandas as pd

csv_path = "pref_simplex_gpi_minecart.csv"
# csv_path = "pref_simplex_env_minecart.csv"
df = pd.read_csv(csv_path)

df = df.select_dtypes(include="number")
r_cols = [c for c in df.columns if c.startswith("r")]
df = df[r_cols]
summary = pd.DataFrame({
    "min": df.min(),
    "max": df.max(),
    "mean": df.mean(),
    "std": df.std(),
    "median": df.median(),
    "mode": df.mode().iloc[0],
    "p5": df.quantile(0.05),
    "p95": df.quantile(0.95),
})

print(summary)