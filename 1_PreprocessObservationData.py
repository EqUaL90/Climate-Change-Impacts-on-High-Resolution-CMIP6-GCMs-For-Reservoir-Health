ObsPrTmp = pd.read_csv(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Observed\NigerPrecipTemp.csv",
                       index_col="Date", parse_dates=True, na_values=-9999)
ObsPrTmp.index = pd.to_datetime(ObsPrTmp.index)
ObsPrTmp = ObsPrTmp.loc["1990-01-01" : "2022-12-31"]
ObsTemp = pd.concat([ObsPrTmp['NigerTav (oC)']] * 269, axis=1)
ObsTemp.columns = [f'T{i}' for i in range(1, 270)]

ObsTempM = ObsTemp.resample('ME').mean().round(2)
ObsTempY = ObsTempM.resample('YE').mean().round(2)
ObsPrc = pd.concat([ObsPrTmp['NigerPrecip (mm/day)']]* 269, axis=1)
ObsPrc.columns = [f'P{i}' for i in range(1, 270)]
ObsPrcM = ObsPrc.resample('ME').sum().round(2)
ObsPrcY = ObsPrcM.resample('YE').sum().round(2)

print(ObsPrTmp.shape, ObsTempM.shape, ObsTempM.tail(2))
print(ObsTempM.tail(2))
print(ObsPrcY.tail(2))
