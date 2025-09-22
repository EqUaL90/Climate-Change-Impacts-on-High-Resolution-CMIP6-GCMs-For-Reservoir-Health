# Imputation function remains the same as your code
missing_station = ["T15", "T72", "T167", "T183", "T257", "T258"]
replace_station = ["T3", "T73", "T168", "T182", "T229", "T229"]
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

Org_combined_Tp_imputD = impute_dfs(Org_combined_Tp, missing_to_replace)
ERA_combined_Tp_imputD = impute_dfs(ERA_combined_Tp, missing_to_replace)

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

for key in ERA_combined_Tp_imputD.keys():
    era_df = ERA_combined_Tp_imputD[key].copy()
    org_key = key.replace("ERA", "Org")
    if org_key not in Org_combined_Tp_imputD:
        print(f"Skipping {key} (missing Org)")
        continue
    
    org_df = Org_combined_Tp_imputD[org_key].copy()

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


# Function to build the model
def build_lstm_model(n_lstm_units=64, learning_rate=0.001, dropout_rate=0.2):
    model = Sequential()
    model.add(Input(shape=(12, n_features)))
    model.add(LSTM(n_lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

model = KerasRegressor(
    model=build_lstm_model,
    n_lstm_units=64,
    learning_rate=0.001,
    dropout_rate=0.2,
    epochs=100,
    batch_size=32,
    verbose=0
)

search_space = {
    "n_lstm_units": Integer(32, 128),
    "learning_rate": Real(1e-5, 1e-2, prior='log-uniform'),
    "dropout_rate": Real(0.1, 0.5),
    "batch_size": Integer(16, 64)
}

opt = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=15,
    cv=3,
    verbose=1,
    random_state=42
)

opt.fit(X_train, y_train)

print("🔍 Best Parameters:")
print(opt.best_params_)

LSTM_TmpBest_04182025 = opt.best_estimator_

# Train the best model
historyTp = LSTM_TmpBest_04182025.fit(X_train, y_train, validation_data=(X_val, y_val), 
                         epochs=1000, batch_size=opt.best_params_['batch_size'], 
                         callbacks=[EarlyStopping(monitor='val_loss', patience=20)])

plt.plot(LSTM_TmpBest_04182025.history_['loss'], label='Train Loss')
plt.plot(LSTM_TmpBest_04182025.history_['val_loss'], label='Val Loss')
plt.title("Training History (Temp)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

plt.plot(LSTM_TmpBest_04182025.history_['loss'], label='Train Loss')
plt.plot(LSTM_TmpBest_04182025.history_['val_loss'], label='Val Loss')
plt.title("Training History (Temp)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(False)
plt.legend()
plt.show()

# Load model
from tensorflow.keras.models import load_model
LSTM_TmpBest_04182025= load_model(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\OutputModel\LSTM_TmpBest_04182025.keras")
print(".keras Model for LSTM temperature model loaded successfully")


#Simulation
############################
y_pred = LSTM_TmpBest_04182025.predict(X_val)
mae_per_station = np.mean(np.abs(y_val - y_pred), axis=0)

station_names = org_df.columns.tolist()
mae_df = pd.DataFrame({'Station': station_names, 'MAE': mae_per_station})
print("\n📊 MAE per station:")
print(mae_df.sort_values(by='MAE'))

################################
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
if len(y_val.shape) > 1 and y_val.shape[1] == 1:
    y_val = y_val.ravel()
    y_pred = y_pred.ravel()

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
bias = np.mean(y_pred - y_val)
pbias = 100 * np.sum(y_val - y_pred) / np.sum(y_val)
nse = 1 - (np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

print(f"\n📈 Validation Results fro Temperature:")
print(f"MAE:    {mae:.4f}")
print(f"MSE:    {mse:.4f}")
print(f"R²:     {r2:.4f}")
print(f"NSE:    {nse:.4f}")
print(f"Bias:   {bias:.4f}")
print(f"PBIAS:  {pbias:.2f}%")


from hydroeval import evaluator, rmse, nse, pbias, kge
from sklearn.metrics import mean_absolute_error, r2_score
Selpredictions_rmvdTp = Selpredictions
DFSel_Tp_Org_imputD_rmvd = DFSel_Tp_Org_imputD

# Define key mapping between DFSel_Pr_Org_imputD and Selpredictions
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

def safe_pearsonr(y_true, y_pred):
    if len(set(y_true)) > 1 and len(set(y_pred)) > 1: 
        return pearsonr(y_true, y_pred)
    else:
        return (float('nan'), float('nan')) 

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)
    corr, _ = safe_pearsonr(y_true, y_pred)
    nse_val = evaluator(nse, y_pred, y_true)[0]
    pbias_val = evaluator(pbias, y_pred, y_true)[0]
    #kge_val = evaluator(kge, y_pred, y_true)[0]
    kge_val = float(kge(y_pred, y_true)[0])  # if `kge(...)` returns a 1-element list/array


    return rmse, mae, bias, r2, corr, nse_val, pbias_val, kge_val

# Initialize metrics storage
ssp_metrics = {
    "Key": [],
    "RMSE": [],
    "MAE": [],
    "Bias": [],
    "R2": [],
    "Corr": [],
    "NSE": [],
    "PBIAS": [],
    "KGE": []
}

# Compute metrics based on mean across all stations (columns)
for original_key, prediction_key in key_mapping.items():
    y_true_df = DFSel_Tp_Org_imputD_rmvd[original_key].iloc[12:].reset_index(drop=True)
    y_pred_df = Selpredictions_rmvdTp[prediction_key].reset_index(drop=True)

    # Drop any columns with NaNs to align properly
    y_true_df = y_true_df.dropna(axis=1)
    y_pred_df = y_pred_df[y_true_df.columns]  # Align columns

    # Compute mean across stations (columns)
    y_true_mean = y_true_df.mean(axis=1).values
    y_pred_mean = y_pred_df.mean(axis=1).values

    # Skip if NaNs are present in the averaged series
    if np.isnan(y_true_mean).any() or np.isnan(y_pred_mean).any():
        continue

    # Compute metrics on the mean time series
    rmse, mae, bias, r2, corr, nse_val, pbias_val, kge_val = compute_metrics(y_true_mean, y_pred_mean)

    # Store results
    ssp_metrics["Key"].append(original_key)
    ssp_metrics["RMSE"].append(rmse)
    ssp_metrics["MAE"].append(mae)
    ssp_metrics["Bias"].append(bias)
    ssp_metrics["R2"].append(r2)
    ssp_metrics["Corr"].append(corr)
    ssp_metrics["NSE"].append(nse_val)
    ssp_metrics["PBIAS"].append(pbias_val)
    ssp_metrics["KGE"].append(kge_val)

# Create final metrics DataFrame
Selssp_metrics_dfTp = pd.DataFrame(ssp_metrics).round(2)
#Selssp_metrics_dfTp.to_excel(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\3_SelEval (2015_2022)\Selssp_metrics_dfLSTM_Temp.xlsx", index = False)
print("Temperature Metrics for LSTM-QDM SSP saved")


