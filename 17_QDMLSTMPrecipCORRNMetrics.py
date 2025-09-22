from sklearn.metrics import r2_score
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

# List of model columns (assuming they are identical in both DataFrames)
columns = true_mean_df.columns.tolist()

# Loop through each model–SSP combination
for idx, col in enumerate(columns):
    ax = axes[idx]

    # True vs predicted
    true_vals = true_mean_df[col]
    pred_vals = pred_mean_df[col]

    # Plot regression with 95% CI
    sns.regplot(x=true_vals, y=pred_vals, ax=ax, ci=95, line_kws={"color": "blue"}, scatter_kws={"alpha": 0.6})

    # Calculate R²
    r2 = r2_score(true_vals, pred_vals)

    # Annotate R² score on the plot
    ax.text(0.05, 0.95, f"R² = {r2:.2f}", transform=ax.transAxes, fontsize=16,
            verticalalignment='top')
    
    ax.set_title(col, fontsize=17)
    ax.set_xlabel("Raw Precip. SSP", fontsize=15)
    ax.set_ylabel("QDM-LSTM_Corrected", fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(False)

# Adjust layout

plt.tight_layout()
#plt.suptitle("True vs Predicted Values with R² and 95% Confidence Interval", fontsize=22, y=1.02)
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\13Correlation Plot of SelPrecipCorr.png", dpi = 500)
plt.show()


from hydroeval import evaluator, rmse, nse, pbias, kge
from sklearn.metrics import mean_absolute_error, r2_score

key_map = {
    'SelFutPrecAcc_SSP126_Rgdd': 'Tr_ERA_SSP126AccPrec_corr',
    'SelFutPrecAcc_SSP245_Rgdd': 'Tr_ERA_SSP245AccPrec_corr',
    'SelFutPrecAcc_SSP585_Rgdd': 'Tr_ERA_SSP585AccPrec_corr',
    'SelFutPrecCan_SSP126_Rgdd': 'Tr_ERA_SSP126CanPrc_corr',
    'SelFutPrecCan_SSP245_Rgdd': 'Tr_ERA_SSP245CanPrc_corr',
    'SelFutPrecCan_SSP585_Rgdd': 'Tr_ERA_SSP585CanPrc_corr',
    'SelFutPrecGFDL_SSP126_Rgdd': 'Tr_ERA_SSP126GFDLPrc_corr',
    'SelFutPrecGFDL_SSP245_Rgdd': 'Tr_ERA_SSP245GFDLPrc_corr',
    'SelFutPrecGFDL_SSP585_Rgdd': 'Tr_ERA_SSP585GFDLPrc_corr'
}

def evaluate_hydro_metrics(true_data_dict, pred_data_dict, key_map):
    results = []

    for key_true, key_pred in key_map.items():
        y_true = true_data_dict[key_true].mean(axis=1).values
        y_pred = pred_data_dict[key_pred].mean(axis=1).values

        rmse_val = rmse(y_pred, y_true)
        mae_val = mean_absolute_error(y_true, y_pred)
        bias_val = np.mean(y_pred - y_true)
        r2_val = r2_score(y_true, y_pred)
        corr_val = np.corrcoef(y_true, y_pred)[0, 1]
        nse_val = nse(y_pred, y_true)
        pbias_val = pbias(y_pred, y_true)
        kge_val = kge(y_pred, y_true)

        results.append({
            'Key': key_true,
            'RMSE': round(rmse_val, 2),
            'MAE': round(mae_val, 2),
            'Bias': round(bias_val, 2),
            'R2': round(r2_val, 2),
            'Corr': round(corr_val, 2),
            'NSE': round(nse_val, 2),
            'PBIAS': round(pbias_val, 2),
            'KGE': [kge_val]
        })

    return pd.DataFrame(results)

df_metricsERA_OnlyBC = evaluate_hydro_metrics(DFSel_Pr_Org_imputD, DFSel_Pr_ERA_Corr_imputD, key_map)
#df_metricsERA_OnlyBC.to_excel(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\3_SelEval (2015_2022)\df_metricsERA_OnlyBCSSP.xlsx", index = False)
print("Metrics for ERAOnly QDM SSP saved")
