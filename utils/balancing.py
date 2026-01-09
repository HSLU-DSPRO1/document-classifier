import pandas as pd

def cap_per_class(df, label_col="doc_type", max_per_class=1000, random_state=42):
    dfs = []
    for label, sub in df.groupby(label_col):
        if len(sub) > max_per_class:
            sub = sub.sample(max_per_class, random_state=random_state)
        dfs.append(sub)
    return pd.concat(dfs, ignore_index=True)
