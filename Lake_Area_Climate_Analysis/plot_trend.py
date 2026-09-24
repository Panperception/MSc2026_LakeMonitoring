import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plot_dir = os.path.join(base_dir,'plots')
os.makedirs(plot_dir, exist_ok=True)

# Plot lake area trend
bar_colors = {'significant (↑)': 'red', 'significant (↓)': 'red', 'moderate (↑)': 'yellow', 'moderate (↓)': 'yellow', 'stable': 'blue'}

# Plot climate variables trend
features = [('area_change_%', 'Area Change (%) - Initial vs Final 3-year window', 'Lake Area Trend'),
            ('temp_trend_per_decade',   'Temperature Change (°C/decade)',    'Temperature Trend'),
            ('precip_trend_per_decade', 'Precipitation Change (mm/decade)', 'Precipitation Trend'),
            ('evap_trend_per_decade',   'Potential Evaportaion Change (mm/decade)',   'Potential Evaporation Trend'),
            ('snowm_trend_per_decade',  'Snow Melt Change (%/decade)',      'Snow Melt Trend'),
            ('snowd_trend_per_decade',  'Snow Depth Change (cm/decade)',    'Snow Depth Trend')]

fig, axes = plt.subplots(3, 2, figsize=(24, 30))

for ax, (feature, ylabel, title) in zip(axes.flat, features):
    f_category = feature.split('_')[0]+'_category'
    ax.bar(df_trend['lake'], df_trend[feature], color=df_trend[f_category].map(bar_colors))

    ax.axhline(0, linewidth=1)
    ax.set_xlabel('Lake')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=90)

    if feature != 'area_change_%':
      ax.legend(handles=[
          Patch(color='red', label='Significant (p < 0.05)'),
          Patch(color='yellow', label='Moderate (p < 0.10)'),
          Patch(color='blue', label='Stable')
      ])
      continue
    
    ax.legend(handles=[
        Patch(color='red', label='Significant (area_change > 10% & p < 0.05)'),
        Patch(color='yellow', label='Moderate (area_change > 10% & p < 0.10) | (area_change < 10% & p < 0.05)'),
        Patch(color='blue', label='Stable')
    ])

    # Add lake names for significant and moderate categories
    #for i, (lake_id, value, category) in enumerate(zip(df_trend['lake'], df_trend[feature], df_trend[f_category])):
    #    if str(category) in ['significant (↑)', 'moderate (↑)']:
    #            ax.text(i, value, lake_dict[lake_id], ha='center', va='bottom', fontsize=9)
    #    elif str(category) in ['significant (↓)', 'moderate (↓)']:
    #            ax.text(i, value, lake_dict[lake_id], ha='center', va='top', fontsize=9)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.18)
plt.savefig(f"{plot_dir}/Fig_lake_trend.png", dpi=300)
plt.show()
plt.close(fig)
print(f"Saved: {plot_dir}/Fig_lake_trend.png")