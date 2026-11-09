def build_dummies(df, group_col, reference_group):
    groups = [g for g in df[group_col].unique() if g != reference_group]
    for g in groups:
        df[g] = (df[group_col] == g).astype(int)
    return df, groups