"""
full equation: y = b0 + b1*x + b2*F2 + b3*F3 + b4*(x*F2) + b5*(x*F3)
simplified equation: y = b0 + b1*x + b2*F2 + b3*F3
"""

from load_data import load_data
from pathlib import Path
from build_dummies import build_dummies
from run_nested_ftest import run_nested_ftest
import pandas as pd

project_path = Path(__file__).resolve().parent.parent
data_path = project_path / 'data' / 'ratio independability test.xlsx'
X_COL = 'APS to TEMED ratio'
GROUP_COL = 'Formulation type'
REFERENCE_GROUP = 'F1'

if __name__ == "__main__":
    df = load_data(data_path)
    df, groups = build_dummies(df, GROUP_COL, REFERENCE_GROUP)
    outputs_dir = project_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for y_col in ['average_dH(nm)', 'average_PDI']:
        print(f"\n{'=' * 20} {y_col} {'=' * 20}")
        full, reduced, F, p = run_nested_ftest(df, groups, y_col=y_col, x_col=X_COL)

        results = {
            "y_variable": [y_col],          # 用循环变量 y_col，不是写死的 Y_COL
            "F_statistic": [F],
            "p_value": [p],
            "rss_reduced": [reduced.ssr],
            "rss_full": [full.ssr],
            "df_num": [reduced.df_resid - full.df_resid],
            "df_den": [full.df_resid],
        }
        results_df = pd.DataFrame(results)
        results_df.to_csv(outputs_dir / f"nested_ftest_{y_col}.csv", index=False)
