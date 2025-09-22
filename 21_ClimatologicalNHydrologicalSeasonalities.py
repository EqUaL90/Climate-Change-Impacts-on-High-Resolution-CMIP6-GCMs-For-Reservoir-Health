def assign_season(month):
    if month in [4, 5, 6, 7, 8, 9, 10]:
        return 'Wet'
    else:
        return 'Dry'

Bas_MME_Mean_Monthly['Season'] = Bas_MME_Mean_Monthly.index.month.map(assign_season)
Bas_MME_Mean_MonthlyRes['Season'] = Bas_MME_Mean_MonthlyRes.index.month.map(assign_season)
BaselineBCEnsF['Season'] = BaselineBCEnsF.index.month.map(assign_season)
BaselineBCEnsFRes['Season'] = BaselineBCEnsFRes.index.month.map(assign_season)

def assign_temporal_period(year):
    if 2026 <= year <= 2050:
        return 'Near Future'
    elif 2051 <= year <= 2080:
        return 'Far Future'
    else:
        return 'Out of Range'

Bas_MME_Mean_Monthly['Temporal Period'] = Bas_MME_Mean_Monthly.index.year.map(assign_temporal_period)
Bas_MME_Mean_MonthlyRes['Temporal Period'] = Bas_MME_Mean_MonthlyRes.index.year.map(assign_temporal_period)

#Use in paper
title_font = 22
label_font = 18
tick_font = 16
legend_font = 18
sns.set(style="whitegrid", rc={
    'axes.titlesize': title_font,
    'axes.labelsize': label_font,
    'xtick.labelsize': tick_font,
    'ytick.labelsize': tick_font,
    'legend.fontsize': legend_font
})


df_basin_melt = Bas_MME_Mean_Monthly.melt(id_vars=['Season', 'Temporal Period'], var_name='Scenario', value_name='Precipitation')
df_basin_melt['Source'] = 'Basin'

Bas_MME_Mean_MonthlyRes_renamed = Bas_MME_Mean_MonthlyRes.rename(
    columns={
        'SSP126_LSTM': 'SSP126',
        'SSP245_LSTM': 'SSP245',
        'SSP585_LSTM': 'SSP585'
    }
)
df_reservoir_melt = Bas_MME_Mean_MonthlyRes_renamed.melt(id_vars=['Season', 'Temporal Period'], var_name='Scenario', value_name='Precipitation')
df_reservoir_melt['Source'] = 'Reservoir'

df_combined = pd.concat([df_basin_melt, df_reservoir_melt], ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

df_near_dry = df_combined[(df_combined['Temporal Period'] == 'Near Future') & (df_combined['Season'] == 'Dry')]
df_near_wet = df_combined[(df_combined['Temporal Period'] == 'Near Future') & (df_combined['Season'] == 'Wet')]
df_far_dry = df_combined[(df_combined['Temporal Period'] == 'Far Future') & (df_combined['Season'] == 'Dry')]
df_far_wet = df_combined[(df_combined['Temporal Period'] == 'Far Future') & (df_combined['Season'] == 'Wet')]

sns.boxplot(data=df_near_dry, x='Scenario', y='Precipitation', hue='Source', palette='Paired',
            hue_order=['Basin', 'Reservoir'], ax=axes[0, 0], showfliers=False, legend=True)
axes[0, 0].set_title('NF - Dry Season', fontsize = title_font)
axes[0, 0].set_xlabel('SSP Scenario', fontsize = label_font)
axes[0, 0].set_ylabel('Precipitation (mm/month)', fontsize = label_font)
axes[0, 0].tick_params(labelsize = tick_font)

sns.boxplot(data=df_near_wet, x='Scenario', y='Precipitation', hue='Source', palette='Paired',
            hue_order=['Basin', 'Reservoir'], ax=axes[0, 1], showfliers=False, legend=False)
axes[0, 1].set_title('NF - Wet Season', fontsize = title_font)
axes[0, 1].set_xlabel('SSP Scenario', fontsize = label_font)
axes[0, 1].set_ylabel('Precipitation (mm/month)', fontsize = label_font)
axes[0, 1].tick_params(labelsize = tick_font)

sns.boxplot(data=df_far_dry, x='Scenario', y='Precipitation', hue='Source', palette='Paired',
            hue_order=['Basin', 'Reservoir'], ax=axes[1, 0], showfliers=False, legend=False)
axes[1, 0].set_title('FF - Dry Season', fontsize = title_font)
axes[1, 0].set_xlabel('SSP Scenario', fontsize = label_font)
axes[1, 0].set_ylabel('Precipitation (mm/month)', fontsize = label_font)
axes[1, 0].tick_params(labelsize = tick_font)

sns.boxplot(data=df_far_wet, x='Scenario', y='Precipitation', hue='Source', palette='Paired',
            hue_order=['Basin', 'Reservoir'], ax=axes[1, 1], showfliers=False, legend=False)
axes[1, 1].set_title('FF - Wet Season', fontsize = title_font)
axes[1, 1].set_xlabel('SSP Scenario', fontsize = label_font)
axes[1, 1].set_ylabel('Precipitation (mm/month)', fontsize = label_font)
axes[1, 1].tick_params(labelsize = tick_font)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2)
for ax in axes.flat:
    ax.grid(False)
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\10ClimSeasonalPrecipBoxPlot.png", dpi = 500)
plt.show()


def assign_flow_regime(month):
    if month in [6, 7, 8, 9, 10, 11]:
        return 'WF'
    elif month in [12, 1, 2, 3]:
        return 'BF'
    else:
        return None  # April and May are not included in the regimes

# Apply flow regime mapping
Bas_MME_Mean_Monthly['FlowRegime'] = Bas_MME_Mean_Monthly.index.month.map(assign_flow_regime)
Bas_MME_Mean_MonthlyRes['FlowRegime'] = Bas_MME_Mean_MonthlyRes.index.month.map(assign_flow_regime)

df_basin_melt = Bas_MME_Mean_Monthly.melt(id_vars=['Season', 'Temporal Period', 'FlowRegime'], 
                                           var_name='Scenario', value_name='Precipitation')
df_basin_melt['Source'] = 'Basin'

Bas_MME_Mean_MonthlyRes_renamed = Bas_MME_Mean_MonthlyRes.rename(columns={
    'SSP126': 'SSP126',
    'SSP245': 'SSP245',
    'SSP585': 'SSP585'
})
df_reservoir_melt = Bas_MME_Mean_MonthlyRes_renamed.melt(id_vars=['Season', 'Temporal Period', 'FlowRegime'], 
                                                         var_name='Scenario', value_name='Precipitation')
df_reservoir_melt['Source'] = 'Reservoir'

df_combined = pd.concat([df_basin_melt, df_reservoir_melt], ignore_index=True)

df_combined.head(2)

