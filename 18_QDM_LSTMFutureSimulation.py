#N
# === Imputation ===
missing_station = ["P15", "P72", "P167", "P183", "P257", "P258"]
replace_station = ["P3", "P73", "P168", "P182", "P229", "P229"]
missing_to_replace = dict(zip(missing_station, replace_station))

def impute_dfs(dirtydata, mapping):
    imputed_dict = {}
    for key, df in dirtydata.items():
        df_copy = df.copy()
        for miss_col, rep_col in mapping.items():
            if miss_col in df_copy.columns and rep_col in df_copy.columns:
                df_copy[miss_col] = df_copy[miss_col].fillna(df_copy[rep_col])
        df_copy = df_copy.ffill().bfill()
        df_copy = df_copy.dropna(axis=1, how='all')
        df_copy = df_copy.apply(pd.to_numeric, errors='coerce')
        imputed_dict[key] = df_copy
    return imputed_dict

# Apply imputation
SSP_imputed = impute_dfs(DF_Pr_target_named_imputD, missing_to_replace)

# === Scale input using training scalers ===
def scale_with_training_scalers(df, scalers):
    df_scaled = df.copy()
    for col in df.columns:
        if col in scalers:
            df_scaled[col] = scalers[col].transform(df[[col]])
        else:
            df_scaled[col] = df[col]  # Leave as-is if scaler not found
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
    print(f"Processing {ssp_key}...")
    
    # Get corresponding training scalers (from best model training)
    scalers = all_scalers[list(all_scalers.keys())[0]]  # You can refine this if different scalers per station

    # Scale
    df_scaled = scale_with_training_scalers(df, scalers)
    
    # Generate sequences
    X_input = create_lstm_sequences_input(df_scaled, window_size=12)
    
    # Predict
    y_pred_scaled = LSTM_PrcBest_04162025.predict(X_input)
    
    # Inverse transform each column
    y_pred_df = pd.DataFrame(y_pred_scaled, columns=df.columns)
    for col in y_pred_df.columns:
        if col in scalers:
            y_pred_df[col] = scalers[col].inverse_transform(y_pred_df[[col]])
    
    # Store
    predictions_by_ssp[ssp_key] = y_pred_df

#N
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ssp_summary_stats = {}

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

for ssp_key in predictions_by_ssp:
    print(f"Processing {ssp_key}...")

    y_pred_df = predictions_by_ssp[ssp_key]
    y_true_df = DF_Pr_target_named_imputD[ssp_key].iloc[12:].reset_index(drop=True)

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
ssp_stats_df = pd.DataFrame.from_dict(ssp_summary_stats, orient='index')
print(ssp_stats_df.round(4))

#Harvest simulations

pred_only_df = {}
target_only_df = {}

for ssp_key, df in ssp_stats_df.items():
    model_name = key_mapping.get(ssp_key, ssp_key)
    clipped_pred = df['Predicted'].clip(lower=0)

    clipped_pred = [round(x, 2) for x in clipped_pred] 
    rounded_target = [round(x, 2) for x in df['Target']]
    
    pred_only_df[model_name] = clipped_pred
    target_only_df[model_name] = rounded_target

date_index = ssp_stats_df[list(ssp_stats_df.keys())[0]]['Date']

LSTM_FutPrc = pd.DataFrame(pred_only_df, index=date_index)
LSTM_FutPrc.index.name = 'Date'

CMIP6_OldFutPrc = pd.DataFrame(target_only_df, index=date_index)
CMIP6_OldFutPrc.index.name = 'Date'

print("LSTM Prc Predictions:")
print("\nOld and Biased CMIP6:")
