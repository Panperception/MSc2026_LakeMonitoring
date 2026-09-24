import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plot_dir = os.path.join(base_dir,'plots')
os.makedirs(plot_dir, exist_ok=True)

## GBR - Gradient Boosting Regressor
# Setup dataset for GBR modeling
features = ['temp_C','precip_mm','evap_mm','snow_melt_mm','snow_depth_cm']
X = df_lakes[features]
y = df_lakes['relative_area']
print("GBR Modeling:")
print(f"X\n{X}")
print(f"\ny\n{y}")

X_filter = df_filter_lakes[features]
y_filter = df_filter_lakes['relative_area']

# Compute range of years and train-test split years (ratio 70/30)
years = np.sort(df_lakes['year'].unique())
years_cnt = len(years)
train_cnt = int(np.floor(years_cnt * 0.70))
train_years = years[:train_cnt]
test_years = years[train_cnt:]
print("\nTrain years:", train_years)
print("Test years:", test_years)

# Split dataset
train_split, test_split = df_lakes['year'].isin(train_years), df_lakes['year'].isin(test_years)
X_train, y_train = X[train_split], y[train_split]
X_test, y_test = X[test_split], y[test_split]

# Train GBR model
print("\nTrain GBR model")
gbr_model = GradientBoostingRegressor(random_state=63)
gbr_model.fit(X_train, y_train)

# Compute evaluation metrics on train dataset
pred = gbr_model.predict(X_train)
MAE = round(float(mean_absolute_error(y_train, pred)),3)
RMSE = round(float(np.sqrt(mean_squared_error(y_train, pred))),3)
R2 = round(float(r2_score(y_train, pred)),3)
print(f"\nGBR train dataset evaluation metrics: MAE = {MAE}, RMSE = {RMSE}, R2 = {R2}")

for (dataset,X_analysis,y_analysis) in [('Test',X_test,y_test), ('Significant/Moderate Category',X_filter,y_filter)]:
    print(f"\n\nAnalysis on {dataset} Lakes Dataset:")

    # Compute evaluation metrics on analysis dataset
    pred = gbr_model.predict(X_analysis)
    MAE = round(float(mean_absolute_error(y_analysis, pred)),3)
    RMSE = round(float(np.sqrt(mean_squared_error(y_analysis, pred))),3)
    R2 = round(float(r2_score(y_analysis, pred)),3)
    print(f"\nGBR evaluation metrics: MAE = {MAE}, RMSE = {RMSE}, R2 = {R2}")

    ## Permutation Importance
    # Calculate permutation importance using negative MAE on analysis dataset using trained GBR model
    perm_imp = permutation_importance(gbr_model, X_analysis, y_analysis,
                                      n_repeats=30, random_state=63,
                                      scoring='neg_mean_absolute_error')

    # Extract mean permutation importance for each feature
    df_perm_imp = pd.DataFrame({'feature': X_analysis.columns,
        'permutation_importance': perm_imp.importances_mean})

    # Sort by feature importance
    df_perm_imp = df_perm_imp.sort_values('permutation_importance', ascending=False).reset_index(drop=True)
    print("\nPermutation Importance:")
    print(df_perm_imp)


    ## SHAP explainer
    # Calculate SHAP values on analysis dataset using trained GBR model
    explainer = shap.Explainer(gbr_model)
    shap_vals = explainer(X_analysis)
    shap_vals = shap_vals.values if hasattr(shap_vals, 'values') else shap_vals

    shap_results = []
    for i, feature in enumerate(X_analysis.columns):
        # Calculate average SHAP magnitude
        magnitude = float(np.mean(np.abs(shap_vals[:, i])))

        # Calculate SHAP direction using Spearman correlation
        direction, _ = spearmanr(X_analysis[feature], shap_vals[:, i])

        shap_results.append({'feature': feature, 'magnitude': magnitude, 'direction': float(direction)})

    df_shap = pd.DataFrame(shap_results)

    # Sort by feature importance based on SHAP magnitude
    df_shap = df_shap.sort_values('magnitude', ascending=False).reset_index(drop=True)
    print("\nSHAP Analysis:")
    print(df_shap)


    ## Plot feature importance
    plt.figure(figsize=(8, 4))

    df_pi_plot = df_perm_imp.sort_values('permutation_importance', ascending=True)
    plt.barh(df_pi_plot['feature'],df_pi_plot['permutation_importance'])

    plt.xlabel('Mean permutation importance',fontsize=10)
    plt.ylabel('Climate feature',fontsize=10)
    plt.title('Permutation Feature Importance',fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Fig_PI_{dataset.split( )[-1]}.png", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved: {plot_dir}/Fig_PI_{dataset.split( )[-1]}.png")

    print("\n")
    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_vals,X_analysis,show=False)

    ax = plt.gca()
    ax.set_xlabel('SHAP values',fontsize=9)
    ax.set_ylabel('Climate feature',fontsize=9)
    ax.set_title('SHAP Explainer Summary',fontsize=11)
    ax.tick_params(axis='both', labelsize=9)

    fig = plt.gcf()
    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel('Feature value', fontsize=9)
    cbar_ax.tick_params(labelsize=9)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
      label.set_color('black')
      label.set_alpha(1)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Fig_SHAP_{dataset.split( )[-1]}.png", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved: {plot_dir}/Fig_SHAP_{dataset.split( )[-1]}.png")