from sklearn.metrics import r2_score
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

# List of model columns (assuming they are identical in both DataFrames)
columns = true_mean_dfTp.columns.tolist()

# Loop through each model–SSP combination
for idx, col in enumerate(columns):
    ax = axes[idx]

    # True vs predicted
    true_valsTp = true_mean_dfTp[col]
    pred_valsTp = pred_mean_dfTp[col]

    # Plot regression with 95% CI
    sns.regplot(x=true_valsTp, y=pred_valsTp, ax=ax, ci=95, line_kws={"color": "brown"}, scatter_kws={"alpha": 0.6})

    # Calculate R²
    r2 = r2_score(true_valsTp, pred_valsTp)

    # Annotate R² score on the plot
    ax.text(0.05, 0.95, f"R² = {r2:.2f}", transform=ax.transAxes, fontsize=16,
            verticalalignment='top')
    
    ax.set_title(col, fontsize=17)
    ax.set_xlabel("Raw Temp. SSP", fontsize=15)
    ax.set_ylabel("QDM-LSTM_Corrected", fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(False)

# Adjust layout
plt.tight_layout()
#plt.suptitle("True vs Predicted Values with R² and 95% Confidence Interval", fontsize=22, y=1.02)
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\13Correlation Plot of SelTempCorr.png", dpi = 500)
plt.show()

from hydroeval import evaluator, rmse, nse, pbias, kge
from sklearn.metrics import mean_absolute_error, r2_score

key_mapping = {
    'SelFutTempAcc_SSP126_Rgdd': 'Tr_ERA_SSP126AccTmp_corr',  #NC-based evaluation
    'SelFutTempAcc_SSP245_Rgdd': 'Tr_ERA_SSP245AccTmp_corr',
    'SelFutTempAcc_SSP585_Rgdd': 'Tr_ERA_SSP585AccTmp_corr',
    'SelFutTempCan_SSP126_Rgdd': 'Tr_ERA_SSP126CanTmp_corr',
    'SelFutTempCan_SSP245_Rgdd': 'Tr_ERA_SSP245CanTmp_corr',
    'SelFutTempCan_SSP585_Rgdd': 'Tr_ERA_SSP585CanTmp_corr',
    'SelFutTempGFDL_SSP126_Rgdd': 'Tr_ERA_SSP126GFDLTmp_corr',
    'SelFutTempGFDL_SSP245_Rgdd': 'Tr_ERA_SSP245GFDLTmp_corr',
    'SelFutTempGFDL_SSP585_Rgdd': 'Tr_ERA_SSP585GFDLTmp_corr'
}

def evaluate_hydro_metrics(true_data_dict, pred_data_dict, key_map):
    results = []

    for key_true, key_pred in key_mapping.items():
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

df_metricsERA_OnlyBCTp = evaluate_hydro_metrics(DFSel_Tp_Org_imputD, DFSel_Tp_ERA_Corr_imputD, key_mapping)
#df_metricsERA_OnlyBCTp.to_excel(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\3_SelEval (2015_2022)\df_metricsERA_OnlyBCSSP_Temp.xlsx", index = False)
print("Metrics for ERAOnly QDM SSP saved")
df_metricsERA_OnlyBCTp


#BC Future Temp Prediction
# === Imputation ===
SSP_imputed = DF_Tp_target_named_imputD

# === Scale input using training scalers ===
def scale_with_training_scalers(df, scalers):
    df_scaled = df.copy()
    for col in df.columns:
        if col in scalers:
            df_scaled[col] = scalers[col].transform(df[[col]])
        else:
            df_scaled[col] = df[col]
    return df_scaled

# === Create LSTM Sequences ===
def create_lstm_sequences_input(input_df, window_size=12):
    X = []
    for i in range(len(input_df) - window_size):
        X_seq = input_df.iloc[i:i+window_size].values
        X.append(X_seq)
    return np.array(X)

# === Predict on all SSPs ===
predictions_by_ssp = {}

for ssp_key, df in SSP_imputed.items():
    print(f"Temp Processing {ssp_key}...")
    scalers = all_scalers[list(all_scalers.keys())[0]]
    df_scaled = scale_with_training_scalers(df, scalers)
    
    X_input = create_lstm_sequences_input(df_scaled, window_size=12)
    
    y_pred_scaled = LSTM_TmpBest_04182025.predict(X_input)
    
    # Inverse transform each column
    y_pred_df = pd.DataFrame(y_pred_scaled, columns=df.columns)
    for col in y_pred_df.columns:
        if col in scalers:
            y_pred_df[col] = scalers[col].inverse_transform(y_pred_df[[col]])
    
    # Store
    predictions_by_ssp[ssp_key] = y_pred_df

# Now, predictions_by_ssp is a dictionary where:
# - keys are SSP dataset names (e.g., 'AccTmpSSP245_26_80')
# - values are DataFrames of predicted values per station

#######################################################################
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ssp_summary_stats = {}

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

for ssp_key in predictions_by_ssp:
    print(f"Temp Processing {ssp_key}...")

    y_pred_df = predictions_by_ssp[ssp_key]
    y_true_df = DF_Tp_target_named_imputD[ssp_key].iloc[12:].reset_index(drop=True)

    # Align columns
    common_cols = y_pred_df.columns.intersection(y_true_df.columns)
    y_pred_df = y_pred_df[common_cols]
    y_true_df = y_true_df[common_cols]

    maes, rmses, r2s = [], [], []

    for col in common_cols:
        try:
            mae, rmse, r2 = compute_metrics(y_true_df[col], y_pred_df[col])
            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)
        except:
            continue

    # Store mean metrics for this SSP
    ssp_summary_stats[ssp_key] = {
        'MAE_mean': np.mean(maes),
        'RMSE_mean': np.mean(rmses),
        'R2_mean': np.mean(r2s)
    }

# === Print SSP summary statistics
print("\n Mean Prediction Statistics per SSP:\n")
ssp_stats_dfTmp = pd.DataFrame.from_dict(ssp_summary_stats, orient='index')
print(ssp_stats_dfTmp.round(4))


#Org from LSTM
#Harvest df
# Harvest raw predictions from predictions_by_ssp (as used in the first block)
raw_pred_df = {}
raw_target_df = {}

for ssp_key in predictions_by_ssp.keys():
    model_name = key_mapping.get(ssp_key, ssp_key)

    pred_df = predictions_by_ssp[ssp_key]
    target_df = DF_Tp_target_named_imputD[ssp_key] #Seldsforfuture

    pred_mean = pred_df.mean(axis=1)
    target_mean = target_df.mean(axis=1)

    # Align length
    min_length = min(len(pred_mean), len(target_mean))
    pred_mean = pred_mean[:min_length]
    target_mean = target_mean[:min_length]

    # Create date index
    date_index = pd.date_range(start='2027-01-01', periods=min_length, freq='ME')

    # Clip and round if desired
    raw_pred_df[model_name] = [round(x, 2) for x in pred_mean.clip(lower=0)]
    raw_target_df[model_name] = [round(x, 2) for x in target_mean]

# Final DataFrames
LSTM_FutTmp = pd.DataFrame(raw_pred_df, index=date_index)
LSTM_FutTmp.index.name = 'Date'

CMIP6_OldFutTmp = pd.DataFrame(raw_target_df, index=date_index)
CMIP6_OldFutTmp.index.name = 'Date'

# Preview
print("Raw LSTM Predictions:")
print(LSTM_FutTmp.head())

print("\nRaw CMIP6 Targets:")
print(CMIP6_OldFutTmp.head())


