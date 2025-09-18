import pandas as pd

RENEWABLES = [
    'Biomasse [GWh]',
    'Wasserkraft [GWh]',
    'Wind Offshore [GWh]',
    'Wind Onshore [GWh]',
    'Photovoltaik [GWh]',
    'Pumpspeicher [GWh]',
    'Sonstige Erneuerbare [GWh]'
    ]

FOSSILS = [
    'Kernenergie [GWh]',
    'Braunkohle [GWh]',
    'Steinkohle [GWh]',
    'Erdgas [GWh]',
    'Sonstige Konventionelle [GWh]'
    ]


def load_smard_data(csv_file):
    """
    Load SMARD data
    
    in: csv with SMARD data
    out: adapted pd.Dataframe
    
    - transform date format
    - add 'Year column'
    - rename col headers; MWh -> GWh
    - add 'Total columns'
    """
    df = pd.read_csv(
        csv_file,
        delimiter=';',
        thousands='.',
        decimal=',',
        na_values='-',
        dtype={0: str, 1: str}
    )
    
    # transform date format
    df['Datum von'] = pd.to_datetime(df['Datum von'], format="%d.%m.%Y", errors="coerce")
    df['Datum bis'] = pd.to_datetime(df['Datum bis'], format="%d.%m.%Y", errors="coerce")
    
    # add column 'Year'
    df['Year'] = df['Datum von'].dt.year
    cols = df.columns.tolist()
    new_order = cols[:2] + ['Year'] + cols[2:-1]
    df = df[new_order]
    
    # rename column headers
    cols = df.columns.tolist()
    cols[3:] = [c.replace('[MWh] Berechnete Auflösungen', '[GWh]').strip() for c in cols[3:]]
    df.columns = cols

    # transform MWh -> GWh
    df.iloc[:, 3:] = df.iloc[:, 3:] / 1000
    
    # add column 'Total renewable', 'Total fossil', 'Total
    df['Total renewable'] = df[RENEWABLES].sum(axis=1)
    df['Total fossil'] = df[FOSSILS].sum(axis=1)
    df['Total'] = df['Total renewable'] + df['Total fossil']
    
    return df


def select_features(df_corr, target, threshold_corr=0.7, threshold_target=0.2):
    """
    Feature Selection
    in: dataframe of correlation matrix, str: target_col_name, thresholds
    out: selected features, sorted pd.Series of corr values (to target)
        
    threshold_corr - threshold for correlation between predictor variables
    threshold_target - threshold for correlation between a predictor and the target variable
    """
    
    if target not in df_corr.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    
    # select features with corr to target >= threshold_target=0.2
    corr_target = df_corr[target].drop(target)
    corr_target_sorted = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index)
    selected = corr_target_sorted[abs(corr_target_sorted) >= threshold_target].index.tolist()
    
    # check multicollinearity
    final_features = []
    while selected:
        feat = selected.pop(0)
        final_features.append(feat)
        # remove features with high correlation
        selected = [
            f for f in selected if abs(df_corr.loc[feat, f]) < threshold_corr
        ]
    print(f'Corr matrix - selected features: {final_features}')

    return final_features, corr_target.sort_values(key=abs, ascending=False)
