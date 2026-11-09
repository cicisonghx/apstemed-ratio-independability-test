import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams['font.family'] = 'Calibri'

# ---------- 路径设置 ----------
project_path = Path("/Users/cicisong/APS/TEMED ratio independability")
data_path = project_path / 'data' / 'ratio independability test.xlsx'
outputs_dir = project_path / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(data_path)
formulation_col = "Formulation type"
ratio_col = "APS to TEMED ratio"
dh_col = "average_dH(nm)"
df[formulation_col] = df[formulation_col].astype(str).str.strip()

formulations = sorted(df[formulation_col].unique().tolist())
colors_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
colors = {f: colors_list[i % len(colors_list)] for i, f in enumerate(formulations)}

fig, ax = plt.subplots(figsize=(8, 5))

for f in formulations:
    sub = df[df[formulation_col] == f].copy()
    sub = sub.rename(columns={ratio_col: "x", dh_col: "y"})
    sub = sub.dropna(subset=["x", "y"])
    if len(sub) < 4:
        continue

    model = smf.ols("y ~ x + I(x**2)", data=sub).fit()
    b0, b1, b2 = model.params["Intercept"], model.params["x"], model.params["I(x ** 2)"]

    x_grid = np.linspace(sub["x"].min(), sub["x"].max(), 200)
    y_grid = b0 + b1 * x_grid + b2 * x_grid**2
    y_grid_masked = np.where(y_grid >= 0, y_grid, np.nan)

    ax.scatter(sub["x"], sub["y"], color=colors[f], alpha=0.6, label=f"{f} data")
    ax.plot(x_grid, y_grid_masked, color=colors[f], label=f"{f} fit")

ax.axvspan(0.8, 1.2, color="gray", alpha=0.15, label="0.8–1.2 reference region")
ax.set_ylim(bottom=0)
ax.set_xlabel("APS/TEMED ratio", fontname="Calibri", fontsize=12)
ax.set_ylabel("dH (nm)", fontname="Calibri", fontsize=12)

legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), prop={"family": "Calibri", "size": 10})

plt.tight_layout()
plt.savefig(outputs_dir / "formulation_ratio_curves.png", dpi=600, bbox_inches="tight")
plt.show()

print(f"已保存至: {outputs_dir / 'formulation_ratio_curves.png'}")