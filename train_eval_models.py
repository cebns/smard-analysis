import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, pearsonr
#from sklearn import tree

from utils import load_smard_data, select_features

#%%

if __name__ == "__main__":
    # Load and preprocess data
    df_elec_daily = load_smard_data('data/Realisierte_Erzeugung_2019_2024_Tag.csv')
    df_prices_daily = pd.read_csv('data/Gro_handelspreise_2019_2024_Tag.csv', delimiter=';',
                                  thousands=".", decimal=",", na_values="-", dtype={0: str, 1: str})

    df_prices_daily['Datum von'] = pd.to_datetime(df_prices_daily['Datum von'], format="%d.%m.%Y", errors="coerce")
    df_elec_daily['Day-ahead-price GER'] = (df_prices_daily['Deutschland/Luxemburg [€/MWh] Berechnete Auflösungen'].shift(-1).values)
    
    df = df_elec_daily.drop(columns=['Kernenergie [GWh]', 'Total renewable', 'Total fossil', 'Total'])
    df = df.dropna()
    
    #%% Select features
    
    df_corr_matrix = pd.read_csv('plots/correlation_matrix.csv', sep=',', index_col=0)
    
    feature_cols, _ = select_features(df_corr_matrix, target='Day-ahead-price GER',
                                            threshold_target=0.2, threshold_corr=0.5)
    
# =============================================================================
#     feature_cols = df_elec_daily.columns.to_list()[3:-4]
#     feature_cols.remove('Kernenergie [GWh]')
# =============================================================================
    
    target_col = 'Day-ahead-price GER'
    
    # train test split - 2024 is test dataset, rest is train data
    df_train = df[df['Year'] < 2024]
    df_test  = df[df['Year'] >= 2024]
    
    X_train, y_train = df_train[feature_cols].values, df_train[target_col].to_numpy()
    X_test, y_test = df_test[feature_cols].values, df_test[target_col].values

    #%% 1. Linear Regression
    
    preds = {}
    
    pipe_lin = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
        ])
    
    # fit and predict
    pipe_lin.fit(X_train, y_train)
    y_pred_lin = pipe_lin.predict(X_test)
    preds['Linear'] = y_pred_lin
    
    print(f'[Linear Regression]  R²: {r2_score(y_test, y_pred_lin):.3f} \tMSE: {mean_squared_error(y_test, y_pred_lin):.1f}')
    
    #%% 2. Ridge Regression
    
    alphas = [1., 10., 100., 1000., 10000.]

    pipe_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error'))
    ])
    
    # fit and predict
    pipe_ridge.fit(X_train, y_train)
    y_pred_ridge = pipe_ridge.predict(X_test)
    preds['Ridge'] = y_pred_ridge
    
    print(f'[Ridge Regression]\tR²: {r2_score(y_test, y_pred_ridge):.3f} \tMSE: {mean_squared_error(y_test, y_pred_ridge):.1f} \t alpha: {pipe_ridge.named_steps['model'].alpha_:.0f}')
    
    #%% 3. Decision Tree
    hyperparams = {'max_depth': [5, 6, 7,8], 'max_leaf_nodes': [30, 35, 40, 45, 50],
                   'max_features': [5, 7, 9, 11]}
    
    # fit/optimize and predict
    reg = DecisionTreeRegressor(random_state=42)
    reg = GridSearchCV(reg, hyperparams, cv=5)
    reg.fit(X_train, y_train)
    y_pred_dectree_train = reg.predict(X_train)
    y_pred_dectree_test = reg.predict(X_test)
    preds['Decision Tree'] = y_pred_dectree_test
    
    print(f'\nMSE DecTree Train: {mean_squared_error(y_train,  y_pred_dectree_train):.3f} \t MSE DecTree Test: {mean_squared_error(y_test,  y_pred_dectree_test):.3f}')
    print('Best DT parameters:', reg.best_params_)
    
# =============================================================================
#     best_tree = reg.best_estimator_
# 
#     plt.figure(figsize=(20, 10))
#     tree.plot_tree(
#         best_tree,
#         feature_names=feature_cols,
#         filled=True,
#         rounded=True,
#         fontsize=8
#     )
#     plt.show()
# =============================================================================
    
    #%% 4. Random Forest
    print('\nStart RF optimization')
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True, max_features="sqrt", max_samples=0.8)
    rf_params = {
        'n_estimators': randint(120, 240),
        'max_depth': randint(6, 16),
        'min_samples_leaf': randint(3, 15),
    }
    
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=20, cv=3, scoring='neg_mean_squared_error',
                                n_jobs=-1, random_state=42, verbose=1)
    rf_search.fit(X_train, y_train)
    
    y_pred = rf_search.predict(X_test)
    print('Best params:', rf_search.best_params_)
    print('MSE Test:', mean_squared_error(y_test, y_pred))
    preds['Random Forest'] = rf_search.predict(X_test)
    
    #%% 5. RF with all features except Kernenergie
    
    feature_cols_all = df_elec_daily.columns.to_list()[3:-4]
    feature_cols_all.remove('Kernenergie [GWh]')
    
    X_train_all, y_train_all = df_train[feature_cols_all].values, df_train[target_col].to_numpy()
    X_test_all, y_test_all = df_test[feature_cols_all].values, df_test[target_col].values
    
    print('\nStart RF optimization with all features')
    rf_more_feats = RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True, max_features="sqrt", max_samples=0.8)
    
    rf_search_2 = RandomizedSearchCV(rf_more_feats, rf_params, n_iter=20, cv=3, scoring='neg_mean_squared_error',
                                n_jobs=-1, random_state=42, verbose=1)
    rf_search_2.fit(X_train_all, y_train_all)
    
    y_pred = rf_search_2.predict(X_test_all)
    print('Best params:', rf_search_2.best_params_)
    print('MSE Test:', mean_squared_error(y_test_all, y_pred))
    preds['Random Forest All Feats'] = rf_search_2.predict(X_test_all)
    
    #%% 6. MLP with Lags
    
    import torch.nn as nn
    import torch.optim as optim
    from models import MLP
    
    # select hyperparameters
    lr = 0.001
    input_size = X_train.shape[1]
    hidden_sizes = [32, 16]
    output_size = 1
    
    loss_fn = nn.MSELoss()
    
    # create lag features n-7 ... n-1 for short trends, maybe also n-14, n-30
    
    # initialize MLP and optimizer
    model_mlp = MLP(input_size, hidden_sizes, output_size)
    optimizer = optim.Adam(model_mlp.parameters(), lr=lr)
    # train MLP
    
    # evaluate MLP
    
    #%% 7. Evaluation
    
    models = list(preds.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10), sharey='row')
    
    for i, model in enumerate(models):
        y_pred = preds[model]
        residuals = y_test - y_pred
    
        # Regression Fit (nur für optische Hilfslinie)
        reg = LinearRegression().fit(y_test.reshape(-1, 1), y_pred)
        y_fit = reg.predict(y_test.reshape(-1, 1))
    
        # Pearson-Korrelation
        r, _ = pearsonr(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        # Scatterplot: Pred vs True
        ax1 = axes[0, i]
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='Ideal', alpha=0.7)
        ax1.plot(y_test, y_fit, color='red', linestyle='-', label='Linear Fit')
    
        ax1.set_title(f'{model} – Prediction vs Actual')
        ax1.set_xlabel('Actual Price (€/MWh)')
        ax1.set_ylabel('Predicted Price (€/MWh)')
        ax1.legend(title=f'R² = {r2:.2f}\nRMSE = {rmse:.0f}\nr = {r:.2f}', loc='upper left')
    
        # Residual Plot
        ax2 = axes[1, i]
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax2.set_title(f'{model} – Residuals')
        ax2.set_xlabel('Predicted Price (€/MWh)')
        ax2.set_ylabel('Residual')
    
    plt.tight_layout()
    plt.savefig('plots/eval_all_models.png', dpi=200)
    plt.show()

    # Lollipop plot
    rmse_scores = {'Linear': root_mean_squared_error(y_test, preds['Linear']),
                  'Ridge': root_mean_squared_error(y_test, preds['Ridge']),
                  'Decision Tree': root_mean_squared_error(y_test, preds['Decision Tree']),
                  'Random Forest': root_mean_squared_error(y_test, preds['Random Forest']),
                  'Random Forest All Feats': root_mean_squared_error(y_test, preds['Random Forest All Feats'])
                  }
    # sort MSE scores
    rmse_scores = dict(sorted(rmse_scores.items(), key=lambda item: item[1], reverse=True))
    
    # Plot
    plt.figure(figsize=(8, 5))
    for i, (model, rmse) in enumerate(rmse_scores.items()):
        plt.plot(i, rmse, 'o', markersize=8, label=model)
        plt.vlines(i, ymin=0, ymax=rmse-1, linestyles='dashed', colors='gray')
    
    plt.xticks(range(len(rmse_scores)), rmse_scores.keys())
    plt.ylabel('Test RMSE [€/MWh]')
    plt.title('Comparison of Model Performance on 2024 Test Dataset')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/RMSE_all_models.png', dpi=200)
    plt.show()