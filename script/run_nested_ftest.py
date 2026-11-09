import statsmodels.formula.api as smf
from scipy import stats

def run_nested_ftest(df, groups, y_col, x_col):
    d = df.rename(columns={y_col: 'y', x_col: 'x'})

    dummy_terms = '+'.join(groups)
    interaction_terms = "+".join(f"x:{g}" for g in groups)
    interaction_terms_sq = "+".join(f"I(x**2):{g}" for g in groups)

    reduced_formula = f'y ~ x + I(x**2) + {dummy_terms}'
    full_formula = f'y ~ x + I(x**2) + {dummy_terms} + {interaction_terms} + {interaction_terms_sq}'

    reduced = smf.ols(reduced_formula, data=d).fit()
    full = smf.ols(full_formula, data=d).fit()

    rss_reduced = reduced.ssr
    rss_full = full.ssr
    df_reduced = reduced.df_resid
    df_full = full.df_resid

    df_num = df_reduced - df_full
    df_den = df_full

    F = ((rss_reduced - rss_full) / df_num) / (rss_full / df_den)
    p = 1 - stats.f.cdf(F, df_num, df_den)

    print(f"reduced model: {reduced_formula}")
    print(f"full model: {full_formula}")
    print(f"RSS_reduced = {rss_reduced:.4f} (df={df_reduced})")
    print(f"RSS_full    = {rss_full:.4f} (df={df_full})")
    print(f"F({df_num},{df_den}) = {F:.4f}, p = {p:.4f}")
    print()
    print(full.summary())

    return full, reduced, F, p
