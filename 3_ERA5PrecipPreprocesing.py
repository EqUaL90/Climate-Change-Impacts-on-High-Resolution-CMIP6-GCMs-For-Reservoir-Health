# F because I needed to consider number of days for monthly totals

def extract_precipitation_timeseries(nc_path, b):
    GlERA = xr.open_dataset(nc_path)
    GlERA["valid_time"] = pd.to_datetime(GlERA["valid_time"].values, unit="s")
    GlERA = GlERA.rename({"valid_time": "time"})
    GlERA["Precip"] = GlERA["tp"] * 1000  # Convert from meters to millimeters
    GlERA["Precip"].attrs["units"] = "mm/month"
    GlERA = GlERA.rio.write_crs("EPSG:4326")

    extracted_data = []
    for time_step in GlERA["time"].values:
        precip_values = []
        
        for _, station in b.iterrows():
            station_coords = (station.geometry.y, station.geometry.x)
            precip_value = GlERA.sel(longitude=station_coords[1], latitude=station_coords[0], time=time_step, method="nearest")["Precip"]
  
            if precip_value.size == 1:
                precip_values.append(precip_value.values.item())
            else:
                precip_values.append(precip_value.values[0])

        year_month = pd.to_datetime(time_step).strftime('%Y-%m')

        days_in_month = pd.to_datetime(year_month + "-01").days_in_month           # Get the number of days in the month
        precip_values = np.array(precip_values) * days_in_month     # Convert from monthly averages to monthly totals by multiplying by the number of days in the month

        station_ids = b["FID"].values
        df = pd.DataFrame({"FID": station_ids, "precipitation": precip_values})
        df["Date"] = pd.to_datetime(year_month)

        extracted_data.append(df)

    final_dfD = pd.concat(extracted_data)
    final_dfD.set_index('Date', inplace=True)
    final_dfD_pivot = final_dfD.pivot(columns='FID', values='precipitation')
    final_dfD_pivot.columns = [f'P{i+1}' for i in range(final_dfD_pivot.shape[1])]

    return final_dfD_pivot


import os

def batch_process_precipitation_timeseries(nc_folder_path, b):
    precipitation_data = {}

    for year in range(1980, 2025):
        nc_filename = f"ERAPrcp_{year}.nc"
        nc_path = os.path.join(nc_folder_path, nc_filename)
        
        if os.path.exists(nc_path):
            print(f"Processing {nc_filename}...")
            precipitation_data[year] = extract_precipitation_timeseries(nc_path, b)
        else:
            print(f"File {nc_filename} not found in the specified folder.")
    
    return precipitation_data

nc_folder_path = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\3ERAFiles\MonthlyPrecip\NewNigEraFile\ERAPrcpNigeria1980_2024"
precipitation_data = batch_process_precipitation_timeseries(nc_folder_path, b)

import pandas as pd

def ConcatP(precipitation_data):
    combined_df = pd.concat(precipitation_data.values(), axis=0)
    combined_df.reset_index(drop=False, inplace=True)
    return combined_df

combined_precip_df = ConcatP(precipitation_data).round(2)

def SelHist(df, start_date, end_date):
    df['Date'] = pd.to_datetime(df['Date'])
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return filtered_df
ERA5Precip = SelHist(combined_precip_df, '1990-01-01', '2022-12-31')
ERA5Precip['Date'] = pd.to_datetime(ERA5Precip['Date'])
ERA5Precip.set_index('Date', inplace=True)
ERA5PrecipM  = ERA5Precip.loc["1990-01-01":"2022-12-31"]
ERA5PrecipM = ERA5PrecipM.resample("ME").sum()
ERA5PrecipY = ERA5PrecipM.resample("YE").sum()
print(ERA5PrecipM.describe())
print(ERA5PrecipM.tail())
print(ERA5PrecipY.tail())
