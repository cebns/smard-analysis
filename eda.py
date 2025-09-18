import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from utils import load_smard_data, RENEWABLES, FOSSILS

'''
    Script to visualize SMARD data from Bundesnetzagentur
    Use this script like a jupyter notebook.
'''
#%% Load data

if __name__ == "__main__":
    df_elec_daily = load_smard_data('data/Realisierte_Erzeugung_2019_2024_Tag.csv')
    df_prices_daily = pd.read_csv('data/Gro_handelspreise_2019_2024_Tag.csv', delimiter=';', thousands='.',
                                  decimal=',', na_values="-", dtype={0: str, 1: str}
                                  )
    
    df_prices_daily['Datum von'] = pd.to_datetime(df_prices_daily['Datum von'], format="%d.%m.%Y", errors="coerce")
    df_prices_daily['Datum bis'] = pd.to_datetime(df_prices_daily['Datum bis'], format="%d.%m.%Y", errors="coerce")
    
    
#%% Overview all electricity sources 2019-2024
    
    feature_cols = df_elec_daily.columns[3:].to_list()
    years = sorted(df_elec_daily['Year'].unique())
    
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(14, 3*len(feature_cols)), sharex=True)
    
    for i, col in enumerate(feature_cols):
        for year in years:
            df_year = df_elec_daily[df_elec_daily['Year'] == year]
            axes[i].plot(
                df_year['Datum von'],
                df_year[col],
                label=str(year),
                alpha=0.85
                )
        axes[i].set_title(f'{col} - Daily Electricity Generation 2019-2024')
        axes[i].tick_params(labelbottom=True)
        axes[i].set_ylabel('GWh/day')
        
    plt.tight_layout()
    plt.show()
    
#%% Smoothed Overview 2019-2024 (sliding window)
    
    # choose size of sliding window
    n = 7
    
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(14, 3*len(feature_cols)), sharex=True)
    
    for i, col in enumerate(feature_cols):
        for year in years:
            df_year = df_elec_daily[df_elec_daily['Year'] == year]
            axes[i].plot(
                df_year['Datum von'],
                df_year[col].rolling(window=n).mean(),
                label=str(year),
                alpha=0.85
                )
            
        axes[i].set_title(f'{col} - Smoothed Daily Electricity Generation 2019-2024')
        axes[i].tick_params(labelbottom=True)
        axes[i].set_ylabel('GWh/day')
        
    plt.tight_layout()
    plt.show()
    
#%% Renewable vs. Fossil Electricity - Overview
    
    n= 7
    
    plt.figure(figsize=(28,6))
    plt.plot(df_elec_daily['Datum von'], df_elec_daily['Total renewable'].rolling(window=n).mean(), label='Renewable', color='green')
    plt.plot(df_elec_daily['Datum von'], df_elec_daily['Total fossil'].rolling(window=n).mean(), label='Fossil', color='brown')
    plt.fill_between(
        df_elec_daily['Datum von'],
        df_elec_daily['Total renewable'],
        df_elec_daily['Total fossil'],
        where=df_elec_daily['Total renewable'] > df_elec_daily['Total fossil'],
        color='green',
        alpha=0.2,
        interpolate=True,
        label='EE > Fossil'
    )
    plt.xlabel('Year')
    plt.ylabel('GWh/day')
    plt.title('Renewable vs. Fossil Electricity 2019–2024')
    plt.legend()
    plt.show()
    
#%% Renewable vs. Fossil - Detailed Smoothed Overview
    
    # sliding window size
    n = 14
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    
    # left plot: Renewable
    for col in RENEWABLES:
        axes[0].plot(
            df_elec_daily['Datum von'],
            df_elec_daily[col].rolling(window=n, min_periods=1).mean(),
            label=col,
        )
    axes[0].set_title(f'Renewable Energies (Smoothed, {n}-Day Mean)')
    axes[0].set_ylabel('GWh/day')
    
    # right plot: Fossils
    for col in FOSSILS:
        axes[1].plot(
            df_elec_daily['Datum von'],
            df_elec_daily[col].rolling(window=n, min_periods=1).mean(),
            label=col,
        )
    axes[1].set_title(f'Fossil Energies (Smoothed, {n}-Day Mean)')
    
    for ax in axes:
        ax.set_xlabel('Year')
        ax.tick_params(labelbottom=True)
        ax.legend(loc='upper right', fontsize='6')
    
    plt.tight_layout()
    plt.show()
    
#%% Overview Wholesale Eelctricity Prices

    # sliding window size
    n = 30
    
    target_col = 'Deutschland/Luxemburg [€/MWh] Berechnete Auflösungen'
    
    plt.plot(
        df_prices_daily['Datum von'],
        df_prices_daily[target_col],
        label='Daily price',
        alpha = 0.5
        )
    
    plt.plot(df_prices_daily['Datum von'],
             df_prices_daily[target_col].rolling(n, min_periods=5).mean(),
             label=f'{n}-day mean')
    
    
    plt.title('Wholesale Trading Price')
    plt.ylabel('€/MWh')
    
    plt.xlabel('Year')
    plt.tick_params(labelbottom=True)
    plt.legend(loc='upper right', fontsize='10')
    
    plt.tight_layout()
    plt.show()

#%% Correlation Matrix

    # Atomkraft Nein Danke and add col prices to df_elec_daily
    feature_cols.remove('Kernenergie [GWh]')
    df_elec_daily = df_elec_daily.drop('Kernenergie [GWh]', axis=1)
    
    df_elec_daily['Day-ahead-price GER'] = (
        df_prices_daily[target_col]
        .shift(-1)
        .values
    )
    target_col = 'Day-ahead-price GER'
    
    # choose energy features and target price
    corr_cols = feature_cols[:-3]
    corr_cols.append(target_col)
    
    # Correlation plot
    corr = df_elec_daily[corr_cols].corr(method='pearson')
        
    plt.figure(figsize=(10,8))
    plt.imshow(corr, aspect='auto', cmap='coolwarm')
    plt.xticks(range(len(corr.columns)), [c.replace(' [GWh]', '') for c in corr.columns], rotation=90)
    plt.yticks(range(len(corr.index)), [c.replace(' [GWh]', '') for c in corr.index])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('plots/corr_heatmap.png', dpi=200)
    
    # save correlations
    print(corr.round(2))
    corr.round(3).to_csv('plots/correlation_matrix.csv', sep=',')
