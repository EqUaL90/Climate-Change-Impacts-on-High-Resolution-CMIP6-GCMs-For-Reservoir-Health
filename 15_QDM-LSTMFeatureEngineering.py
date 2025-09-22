# Imputation function remains the same as your code
missing_station = ["P15", "P72", "P167", "P183", "P257", "P258"]
replace_station = ["P3", "P73", "P168", "P182", "P229", "P229"]
missing_to_replace = dict(zip(missing_station, replace_station))

def impute_dfs(dirtydata, mapping):
    imputed_dict = {}
    for key, df in dirtydata.items():
        df_copy = df.copy()
        if key in mapping:
            replace_key = mapping[key]
            if replace_key in dirtydata:
                df_copy[df_copy.isna()] = dirtydata[replace_key][df_copy.isna()]
        df_copy = df_copy.ffill().bfill()
        df_copy = df_copy.dropna(axis=1, how='all')
        df_copy = df_copy.apply(pd.to_numeric, errors='coerce')
        imputed_dict[key] = df_copy
    return imputed_dict

Org_combined_Pr_imputD = impute_dfs(Org_combined_Pr, missing_to_replace)
ERA_combined_Pr_imputD = impute_dfs(ERA_combined_Pr, missing_to_replace)

# Scaling function remains the same
def scale_dataframe(df):
    scalers = {}
    df_scaled = df.copy()
    for col in df.columns:
        scaler = MinMaxScaler()
        df_scaled[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    return df_scaled, scalers

# Create sequences function remains the same
def create_lstm_sequences(input_df, target_df, window_size=12):
    X, y = [], []
    for i in range(len(input_df) - window_size):
        X_seq = input_df.iloc[i:i+window_size].values
        y_seq = target_df.iloc[i+window_size].values
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)

# Prepare training data
X_all, y_all = [], []
all_scalers = {}

for key in ERA_combined_Pr_imputD.keys():
    era_df = ERA_combined_Pr_imputD[key].copy()
    org_key = key.replace("ERA", "Org")
    if org_key not in Org_combined_Pr_imputD:
        print(f"Skipping {key} (missing Org)")
        continue
    
    org_df = Org_combined_Pr_imputD[org_key].copy()

    era_df = era_df.ffill().bfill().dropna()
    org_df = org_df.ffill().bfill().dropna()

    era_df_scaled, era_scalers = scale_dataframe(era_df)
    org_df_scaled, _ = scale_dataframe(org_df)

    all_scalers[org_key] = era_scalers  # Save scalers for inverse-transform

    X_seq, y_seq = create_lstm_sequences(era_df_scaled, org_df_scaled, window_size=12)
    X_all.append(X_seq)
    y_all.append(y_seq)

X_all = np.concatenate(X_all)
y_all = np.concatenate(y_all)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
n_features = X_train.shape[2]
n_outputs = y_train.shape[1]
print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_val:", X_val.shape, "y_val:", y_val.shape)

Org_combined_Pr_imputD = impute_dfs(Org_combined_Pr, missing_to_replace)
ERA_combined_Pr_imputD = impute_dfs(ERA_combined_Pr, missing_to_replace)

print(Org_combined_Pr_imputD.keys()) #Target
print(ERA_combined_Pr_imputD.keys()) #Feature  
#dict_keys(['DF_Org_Acc_SSP126_Pr_90_22', 'DF_Org_Acc_SSP245_Pr_90_22', 'DF_Org_Acc_SSP585_Pr_90_22', 'DF_Org_Can_SSP126_Pr_90_22', 'DF_Org_Can_SSP245_Pr_90_22', 'DF_Org_Can_SSP585_Pr_90_22', 'DF_Org_GFDL_SSP126_Pr_90_22', 'DF_Org_GFDL_SSP245_Pr_90_22', 'DF_Org_GFDL_SSP585_Pr_90_22'])
#dict_keys(['DF_ERA_Acc_SSP126_Pr_90_22', 'DF_ERA_Acc_SSP245_Pr_90_22', 'DF_ERA_Acc_SSP585_Pr_90_22', 'DF_ERA_Can_SSP126_Pr_90_22', 'DF_ERA_Can_SSP245_Pr_90_22', 'DF_ERA_Can_SSP585_Pr_90_22', 'DF_ERA_GFDL_SSP126_Pr_90_22', 'DF_ERA_GFDL_SSP245_Pr_90_22', 'DF_ERA_GFDL_SSP585_Pr_90_22'])
