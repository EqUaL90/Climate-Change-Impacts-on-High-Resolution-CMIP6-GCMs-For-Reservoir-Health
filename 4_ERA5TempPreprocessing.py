def extract_temperature_timeseries(nc_path, b):
    GlTempERA = xr.open_dataset(nc_path)
    GlTempERA["valid_time"] = pd.to_datetime(GlTempERA["valid_time"].values, unit="s")
    GlTempERA =GlTempERA.rename({"valid_time": "time"})
    GlTempERA["Temp"] = GlTempERA["t2m"]-273.15
    GlTempERA["Temp"].attrs["units"] = '°C'

    extracted_data = []
    for time_step in GlTempERA["time"].values:
        temp_values = []
        
        for _, station in b.iterrows():
            station_coords = (station.geometry.y, station.geometry.x)
            temp_value = GlTempERA.sel(longitude=station_coords[1], latitude=station_coords[0], time=time_step, method="nearest")["Temp"]
  
            if temp_value.size == 1:
                temp_values.append(temp_value.values.item())
            else:
                temp_values.append(temp_value.values[0])

        year_month = pd.to_datetime(time_step).strftime('%Y-%m')

        station_ids = b["FID"].values
        df = pd.DataFrame({"FID": station_ids, "temperature": temp_values})
        df["Date"] = pd.to_datetime(year_month)

        extracted_data.append(df)

    final_dfD = pd.concat(extracted_data)
    final_dfD.set_index('Date', inplace=True)
    final_dfD_pivot = final_dfD.pivot(columns='FID', values='temperature')
    final_dfD_pivot.columns = [f'T{i+1}' for i in range(final_dfD_pivot.shape[1])]

    return final_dfD_pivot


import os

def batch_process_temperature_timeseries(nc_folder_path, b):
    temperature_data = {}

    for year in range(1980, 2025):
        nc_filename = f"ERATotTem_{year}.nc"
        nc_path = os.path.join(nc_folder_path, nc_filename)
        
        if os.path.exists(nc_path):
            print(f"Processing {nc_filename}...")
            temperature_data[year] = extract_temperature_timeseries(nc_path, b)
        else:
            print(f"File {nc_filename} not found in the specified folder.")
    
    return temperature_data

nc_folder_path = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\3ERAFiles\MonthlyPrecip\NewNigEraFile\ERATempNigeria1980_2024"
temperature_data = batch_process_temperature_timeseries(nc_folder_path, b)

def ConcatP(temperature_data):
    combined_df = pd.concat(temperature_data.values(), axis=0)
    combined_df.reset_index(drop=False, inplace=True)
    return combined_df
combined_temp_df = ConcatP(temperature_data).round(2)

def SelHist(df, start_date, end_date):
    df['Date'] = pd.to_datetime(df['Date'])
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return filtered_df
ERA5Temp = SelHist(combined_temp_df,'1990-01-01', '2022-12-31')
ERA5Temp['Date'] = pd.to_datetime(ERA5Temp['Date'])
ERA5Temp.set_index('Date', inplace=True)
ERA5TempM = ERA5Temp.loc["1990-01-01":"2022-12-31"]
ERA5TempM = ERA5TempM.resample("ME").mean()
ERA5TempY = ERA5TempM.resample("YE").mean()
print(ERA5TempM.describe())
print(ERA5TempM.tail())
print(ERA5TempY.tail())
