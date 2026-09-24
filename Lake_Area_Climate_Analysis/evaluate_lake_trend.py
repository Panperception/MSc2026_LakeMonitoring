import os, shutil
import statsmodels.api as sm
import numpy as np
import pandas as pd

trend_dir = os.path.join(base_dir,'trend')
os.makedirs(trend_dir, exist_ok=True)

# Paths to CSV files
climate_file = f'{base_dir}/lake_climate.csv'
area_file = f'{base_dir}/lake_area.csv'
shutil.copy(climate_file, trend_dir)
shutil.copy(area_file, trend_dir)

# Function to compute linear trend per decade using OLS linear regression
def slope_per_decade(d: pd.DataFrame, y: str):
    if d.empty or d['year'].nunique() < 4:
        return np.nan, np.nan
    X = sm.add_constant(d['year'])
    m = sm.OLS(d[y], X).fit()
    slope_decade = float(m.params['year'] * 10.0)
    p = float(m.pvalues['year'])
    return round(slope_decade,4), round(p,4)

# Function to classify parameter trend based on slope and p values
def classify_trend(slope, p):
    if slope == 0: category = 'stable'
    elif p < 0.05: category = 'significant'
    elif p < 0.1: category = 'moderate'
    else: category = 'stable'

    if category != 'stable':
      if slope > 0: category = category + ' (↑)'
      elif slope < 0: category = category + ' (↓)'

    return category

# Load CSV files
df_area = pd.read_csv(area_file)
df_climate = pd.read_csv(climate_file)
df_climate = df_climate.drop(columns=['system:index', '.geo'])

# Merge dataframes on shared features - lake, year
df_lakes = pd.merge(df_area, df_climate, on=['lake', 'year'], how='inner')

# Preview merged dataframe
print("Merged lake area and climate DataFrame:")
print(df_lakes.info(),"\n")

# Check null entries
null_count = df_lakes.isnull().sum()
if null_count.any():
    print("Null values found:", null_count[null_count > 0])
    raise ValueError("ERROR: Found null values. Stopping.")

# Clean snow variables
for feature in ['snow_melt_mm', 'snow_depth_cm']:
    df_lakes[feature] = df_lakes[feature].mask(df_lakes[feature].abs() < 1e-10, 0)

print(df_lakes.head())

# Compute initial->final years lake area change
# Compute baseline area = maximum lake area value on the first three years
baseline = (df_lakes.sort_values(['lake', 'year'])
                    .groupby('lake').head(3)
                    .groupby('lake')['area_km2']
                    #.median())
                    .max())
df_lakes['baseline_area_km2'] = df_lakes['lake'].map(baseline)

# Compute endline area = maximum lake area value on last three years
endline = (df_lakes.sort_values(['lake', 'year'])
                   .groupby('lake').tail(3)
                   .groupby('lake')['area_km2']
                   #.median())
                   .max())

# Compute baseline->endline lake area trend
df_area = (endline.to_frame('endline_area_km2').join(baseline.rename('baseline_area_km2'))).reset_index()
df_area['area_change_%'] = (df_area['endline_area_km2'] - df_area['baseline_area_km2']) / df_area['baseline_area_km2'] * 100
df_area_sorted = df_area.loc[df_area['area_change_%'].abs().sort_values(ascending=False).index]
print("\nLake area trend Initial vs Final years:")
print(df_area_sorted.reset_index(drop=True).head(5))
#print(df_lakes[df_lakes['lake']=='lake_17'])

# Compute year-to-year area change in percentage
df_lakes['y2y_change_%'] = (df_lakes.sort_values(['lake', 'year'])
                                    .groupby('lake')['area_km2']
                                    .pct_change(fill_method=None) * 100)

# Compute relative area from baseline area
df_lakes['relative_area'] = df_lakes['area_km2'] / df_lakes['baseline_area_km2']

print("\nRelative area & year-to-year change:")
print(df_lakes[['lake','year','area_km2','y2y_change_%','relative_area']])

# Compute lake area and climate variables trend per decade using linear regression
lakes_trend = []
for lake, df_lake in df_lakes.groupby('lake',sort=False):
    sA, pA = slope_per_decade(df_lake, 'relative_area')
    sT, pT = slope_per_decade(df_lake, 'temp_C')
    sP, pP = slope_per_decade(df_lake, 'precip_mm')
    sE, pE = slope_per_decade(df_lake, 'evap_mm')
    sC, pC = slope_per_decade(df_lake, 'snow_melt_mm')
    sD, pD = slope_per_decade(df_lake, 'snow_depth_cm')
    lakes_trend.append({
        'lake': lake,
        'area_trend_per_decade': sA, 'area_trend_p_value': pA,
        'temp_trend_per_decade': sT, 'temp_trend_p_value': pT,
        'precip_trend_per_decade': sP, 'precip_trend_p_value': pP,
        'evap_trend_per_decade': sE, 'evap_trend_p_value': pE,
        'snowm_trend_per_decade': sC, 'snowm_trend_p_value': pC,
        'snowd_trend_per_decade': sD, 'snowd_trend_p_value': pD
    })
df_trend = pd.DataFrame(lakes_trend)
print("\nLake area and climate trend per decade:")
print(df_trend.head(5))

# Compute lake area and climate variables trend per decade using linear regression across all lakes
df_lakegrp = (df_lakes.groupby('year',sort=False)
                      .agg(mean_area=('relative_area', 'mean'), mean_temp=('temp_C', 'mean'),
                           mean_precip=('precip_mm', 'mean'), mean_evap=('evap_mm', 'mean'),
                           mean_snowm=('snow_melt_mm', 'mean'), mean_snowd=('snow_depth_cm', 'mean'))
                      .reset_index())
sA, pA = slope_per_decade(df_lakegrp, 'mean_area')
sT, pT = slope_per_decade(df_lakegrp, 'mean_temp')
sP, pP = slope_per_decade(df_lakegrp, 'mean_precip')
sE, pE = slope_per_decade(df_lakegrp, 'mean_evap')
sC, pC = slope_per_decade(df_lakegrp, 'mean_snowm')
sD, pD = slope_per_decade(df_lakegrp, 'mean_snowd') 
df_grp_trend = pd.DataFrame({'trend_per_decade': [sA, sT, sP, sE, sC, sD], 'p_value': [pA, pT, pP, pE, pC, pD]},
                            index=['area','temperature','precipitation','evaporation','snow_melt','snow_depth'])
df_grp_trend.index.name = 'feature'
print("\nLake area and Climate variables trend per decade across all lakes:")
print(df_grp_trend)

# Compute lake area trend category
# Merge initial->final years lake area change with lake area trend per decade
df_trend = df_trend.merge(df_area[['lake','area_change_%']], on='lake', how='left')
df_trend['area_category'] = np.select(
    [(df_trend['area_change_%'].abs() > 10) & (df_trend['area_trend_p_value'] < 0.05) &
        (np.sign(df_trend['area_change_%']) == np.sign(df_trend['area_trend_per_decade'])),
     ((df_trend['area_change_%'].abs() > 10) & (df_trend['area_trend_p_value'] < 0.10) &
        (np.sign(df_trend['area_change_%']) == np.sign(df_trend['area_trend_per_decade']))) |
        (df_trend['area_trend_p_value'] < 0.05)],
    ['significant', 'moderate'],
    default='stable')
df_trend['area_category'] = np.where(
    df_trend['area_category'] == 'stable', 'stable',
    np.where(df_trend['area_trend_per_decade'] > 0,
             df_trend['area_category'] + ' (↑)' , df_trend['area_category'] + ' (↓)')
)

df_area_trend = df_trend.loc[:,['lake','area_change_%','area_trend_per_decade','area_trend_p_value','area_category']].copy()

df_trend['temp_category'] = df_trend.apply(lambda r: classify_trend(r['temp_trend_per_decade'],r['temp_trend_p_value']),axis=1)
df_trend['precip_category'] = df_trend.apply(lambda r: classify_trend(r['precip_trend_per_decade'],r['precip_trend_p_value']),axis=1)
df_trend['evap_category'] = df_trend.apply(lambda r: classify_trend(r['evap_trend_per_decade'],r['evap_trend_p_value']),axis=1)
df_trend['snowm_category'] = df_trend.apply(lambda r: classify_trend(r['snowm_trend_per_decade'],r['snowm_trend_p_value']),axis=1)
df_trend['snowd_category'] = df_trend.apply(lambda r: classify_trend(r['snowd_trend_per_decade'],r['snowd_trend_p_value']),axis=1)

df_trend['area_category'] = pd.Categorical(df_area_trend['area_category'],
                            categories=['significant (↑)','significant (↓)','moderate (↑)','moderate (↓)','stable'],ordered=True)
df_trend = df_trend[['lake','area_category','area_change_%', 'area_trend_per_decade', 'temp_category', 'temp_trend_per_decade', 'precip_category', 'precip_trend_per_decade', 'evap_category', 'evap_trend_per_decade', 'snowm_category', 'snowm_trend_per_decade', 'snowd_category', 'snowd_trend_per_decade']]
df_trend_final = df_trend[df_trend['area_category'].str.contains('significant|moderate')]
df_trend_sorted = df_trend_final.sort_values(['area_category','area_change_%'],ascending=[True,False],
                      key=lambda x: x.abs() if x.name == 'area_change_%' else x).reset_index(drop=True)
print("\nLake area and climate category:")
print(df_trend_sorted)

# Filter lakes data for significant and moderate categories
filter_lakes = df_trend_final['lake']
df_filter_lakes = df_lakes[df_lakes['lake'].isin(filter_lakes)].copy()
df_filter_lakes.reset_index(drop=True)

# Save lake area trend dataframe as CSV file
csv_file = f"{trend_dir}/lake_area_trend.csv"
df_area_trend.to_csv(csv_file, index=False)
print(f"\nLake area trend information saved at: {csv_file}")
# Save lake area and climate trend dataframe as CSV file
csv_file = f"{trend_dir}/lake_trend.csv"
df_trend.to_csv(csv_file, index=False)
print(f"Lake area and climate trend information saved at: {csv_file}\n")