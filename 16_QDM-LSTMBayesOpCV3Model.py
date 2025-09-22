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

best_model = opt.best_estimator_

# Train the best model
history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                         epochs=1000, batch_size=opt.best_params_['batch_size'], 
                         callbacks=[EarlyStopping(monitor='val_loss', patience=20)])
#Loss Plot
plt.plot(best_model.history_['loss'], label='Train Loss')
plt.plot(best_model.history_['val_loss'], label='Val Loss')
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()
#

y_pred = LSTM_PrcBest_04162025.predict(X_val)
mae_per_station = np.mean(np.abs(y_val - y_pred), axis=0)

station_names = org_df.columns.tolist()
mae_df = pd.DataFrame({'Station': station_names, 'MAE': mae_per_station})
print("\n📊 MAE per station:")
print(mae_df.sort_values(by='MAE'))


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

print(f"\n📈 Validation Results:")
print(f"MAE:    {mae:.4f}")
print(f"MSE:    {mse:.4f}")
print(f"R²:     {r2:.4f}")
print(f"NSE:    {nse:.4f}")
print(f"Bias:   {bias:.4f}")
print(f"PBIAS:  {pbias:.2f}%")

#Residual Plot
#Use in paper
era_input = X_val[:, -1, :].reshape(-1)

# Compute mean residuals for one of the subplots
mean_residuals = residuals.mean(axis=1)

# Reshape residuals for heatmap
residuals_2d = residuals.reshape((X_val.shape[0], -1))

# CDF data
sorted_residuals = np.sort(residuals.flatten())
cdf = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)

# 2x2 Plot
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 1. Scatter plot of residual bias
axs[0, 0].scatter(era_input, residuals.flatten(), alpha=0.1, color='skyblue')
axs[0, 0].axhline(0, color='gray', linestyle='--')
axs[0, 0].set_title("Residual Bias Function learned (LSTM)")
axs[0, 0].set_xlabel("ERA-adjusted")
axs[0, 0].set_ylabel("Residual Bias")
axs[0, 0].grid(False)

# 2. Cumulative distribution of residuals
axs[0, 1].plot(sorted_residuals, cdf, marker='.', linestyle='none')
axs[0, 1].axhline(0.5, color='gray', linestyle='--')
axs[0, 1].set_title("Cumulative Residual Plot")
axs[0, 1].set_xlabel("Residuals")
axs[0, 1].set_ylabel("Cumulative Probability")
axs[0, 1].grid(False)

# 3. Heatmap of residuals
sns.heatmap(residuals_2d, cmap='coolwarm', ax=axs[1, 0], cbar=True)
axs[1, 0].set_title("Residual Bias Heatmap")
axs[1, 0].set_xlabel("Stations")
axs[1, 0].set_ylabel("Time Steps")

# 4. Histogram of mean residuals
sns.histplot(mean_residuals, bins=40, kde=True, color='steelblue', ax=axs[1, 1])
axs[1, 1].axvline(0, color='red', linestyle='--', linewidth=1.5)
axs[1, 1].set_title("Histogram of Mean Residuals")
axs[1, 1].set_xlabel("Mean Residual per Sample")
axs[1, 1].set_ylabel("Frequency")
axs[1, 1].grid(False)

plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\12ResidualPlot.png", dpi = 500)
plt.show()


#Predict on 2015 - 2022 SSPs
SelSSP_imputed = DFSel_Pr_ERA_Corr_imputD
def scale_with_training_scalers(df, scalers):
    df_scaled = df.copy()
    for col in df.columns:
        if col in scalers:
            df_scaled[col] = scalers[col].transform(df[[col]])
        else:
            df_scaled[col] = df[col]
    return df_scaled

def create_lstm_sequences_input(input_df, window_size=12):
    X = []
    for i in range(len(input_df) - window_size):
        X_seq = input_df.iloc[i:i+window_size].values
        X.append(X_seq)
    return np.array(X)

Selpredictions = {}

for ssp_key, df in SelSSP_imputed.items():
    print(f"Processing {ssp_key}...")
    
    scalers = all_scalers[list(all_scalers.keys())[0]]
    df_scaled = scale_with_training_scalers(df, scalers)
    
    X_inputSel = create_lstm_sequences_input(df_scaled, window_size=12)
    y_pred_scaledSel = LSTM_PrcBest_04162025.predict(X_inputSel)
    
    y_pred_dfSel = pd.DataFrame(y_pred_scaledSel, columns=df.columns)
    for col in y_pred_dfSel.columns:
        if col in scalers:
            y_pred_dfSel[col] = scalers[col].inverse_transform(y_pred_dfSel[[col]])
    
    # Store
    Selpredictions[ssp_key] = y_pred_dfSel

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from hydroeval import evaluator, nse, pbias, kge

Selpredictions_rmvd = Selpredictions

# Define key mapping between DFSel_Pr_Org_imputD and Selpredictions
key_mapping = {
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

# Function to safely calculate Pearson correlation
def safe_pearsonr(y_true, y_pred):
    if len(set(y_true)) > 1 and len(set(y_pred)) > 1:  # Ensure variation in both arrays
        return pearsonr(y_true, y_pred)
    else:
        return (float('nan'), float('nan'))  # Return NaN if there's no variation

# Function to compute metrics
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
    y_true_df = DFSel_Pr_Org_imputD[original_key].iloc[12:].reset_index(drop=True)
    y_pred_df = Selpredictions_rmvd[prediction_key].reset_index(drop=True)
    y_pred_df = y_pred_df.clip(lower=0)

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
Selssp_metrics_df = pd.DataFrame(ssp_metrics).round(2)
#Selssp_metrics_df.to_excel(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\3_SelEval (2015_2022)\Selssp_metrics_dfLSTM.xlsx", index = False)
print("Metrics for LSTM-QDM SSP saved")
