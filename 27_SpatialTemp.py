# BASELINE PRECP. Orginal GCM data
mean_ACC = DFACCESS_CM2histMnthPrec1990_2014_regriddedF.mean(axis=1)
mean_CAN =DFCanESM5histMnthPrec1990_2014_regriddedF.mean(axis=1)
mean_GFDL = DFGFDL_ESM4histMnthPrec1990_2014_regriddedF.mean(axis=1)

# BASELINE TEMP. Orginal GCM data
mean_ACCTp = DFACCESS_CM2histMnthTemp1990_2014_regriddedF.mean(axis=1)
mean_CANTp =DFCanESM5histMnthTemp1990_2014_regriddedF.mean(axis=1)
mean_GFDLTp = DFGFDL_ESM4histMnthTemp1990_2014_regriddedF.mean(axis=1)

#BasDFTemp

#EnsembleMean of 3 baseline GCMs
EnsBasTempDF = (DFACCESS_CM2histMnthTemp1990_2014_regriddedF +
    DFCanESM5histMnthTemp1990_2014_regriddedF +
    DFGFDL_ESM4histMnthTemp1990_2014_regriddedF
) / 3

# annual
EnsBasTempDF_yr = EnsBasTempDF.resample('YE').mean()

EnsBasTempDF_yr.index =EnsBasTempDF_yr.index.year
EnsBasTempDF_yr = EnsBasTempDF_yr.round(2)

EnsBasTempDF_yrM= EnsBasTempDF_yr.mean(axis=0)
EnsBasTempDF_yrM +=4.2
df_temp = EnsBasTempDF_yrM.reset_index()
df_temp.columns = ['Station', 'MeanAnnualTemp']
df_temp['FID'] = df_temp['Station'].str.extract(r'T(\d+)').astype(int)
gdf_merged = pd.merge(df_temp, b, on='FID')
gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry='geometry')
MeanBasTemp = gdf_merged[['Station', 'FID', 'MeanAnnualTemp', 'geometry']]
MeanBasTemp

#Basin Mean Annual Temp
# Use in paper (IDW gives perefect values)
Tppredictions_by_sspAdj_26_80 = {
    k: (v).mask((v) == 0)
    for k, v in DF_Tp_target_named_imputD.items()
}
# --- 2. Compute Ensemble Mean for Each SSP ---
def compute_ensemble_mean(ssp_label):
    return pd.concat([
        v for k, v in Tppredictions_by_sspAdj_26_80.items()
        if f"{ssp_label}_26_80" in k
    ]).groupby(level=0).mean()

Tpmean_ensemble_by_ssp = {
    ssp: compute_ensemble_mean(ssp) for ssp in ['SSP126', 'SSP245', 'SSP585']
}

# --- 3. Annual Mean per SSP Period ---
periods = {
    'near': ("2026-01-01", "2050-12-31"),
    'far': ("2051-01-01", "2080-12-31")
}

Tpmean_by_ssp_period_annual = {}
for ssp, df in Tpmean_ensemble_by_ssp.items():
    df.index = pd.to_datetime(df.index)
    annual = df.resample('YE').mean()
    for label, (start, end) in periods.items():
        Tpmean_by_ssp_period_annual[f"{ssp}_{label}"] = annual.loc[start:end].mean(axis=0)

# --- 4. Convert Series to GeoDataFrame ---
def series_to_gdf(series, base_gdf):
    df = series.rename_axis('Station').reset_index(name='Mean_Temp')
    df['FID'] = df['Station'].str.extract(r'T(\d+)').astype(int)
    merged = df.merge(base_gdf[['FID', 'geometry']], on='FID', how='left')
    return gpd.GeoDataFrame(merged, geometry='geometry', crs=base_gdf.crs)

b['FID'] = b['FID'].astype(int)
Tpmean_gdfs = {key: series_to_gdf(series, b) for key, series in Tpmean_by_ssp_period_annual.items()}
for gdf in Tpmean_gdfs.values():
    gdf['Mean_Temp'] = gdf['Mean_Temp'].round(2)

# --- 5. Prepare Boundary and Grid ---
target_crs = "EPSG:32632"
niger5 = niger5.to_crs(target_crs)
boundary_poly = unary_union(niger5.geometry)

minx, miny, maxx, maxy = niger5.total_bounds
res, width, height = 1000, int((maxx - minx) / 1000), int((maxy - miny) / 1000)
transform = from_origin(minx, maxy, res, res)

mask = rasterize([(boundary_poly, 1)], out_shape=(height, width), transform=transform, fill=0, dtype='uint8')

# --- 6. IDW Interpolation Function ---
def idw_fill(raster, power=2):
    ys, xs = np.where(~np.isnan(raster))
    vals = raster[ys, xs]
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    flat_grid = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    pts = np.vstack((xs, ys)).T
    d = cdist(flat_grid, pts)
    d[d == 0] = np.nan
    w = 1 / (d ** power)
    w /= np.nansum(w, axis=1)[:, None]
    return np.dot(w, vals).reshape((height, width))

# --- 7. Create Raster From GDFs ---
def create_rasters_from_gdfs_clipped(gdf_dict, boundary_gdf, resolution=1000):
    boundary_gdf = boundary_gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    width, height = int((maxx - minx) / resolution), int((maxy - miny) / resolution)
    transform = from_origin(minx, maxy, resolution, resolution)
    boundary_mask = rasterize(
        [(geom, 1) for geom in boundary_gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    rasters = {}
    for key, gdf in gdf_dict.items():
        gdf = gdf.to_crs(target_crs)
        gdf = gdf[gdf.geometry.within(boundary_gdf.union_all())]
        shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf['Mean_Temp']) if val > 0]
        raster = rasterize(shapes, out_shape=(height, width), fill=np.nan, transform=transform, dtype='float32')
        raster[boundary_mask == 0] = np.nan
        rasters[f"RasTp_{key}"] = raster
    return rasters, transform, target_crs

rasters_dict, transform, target_crs = create_rasters_from_gdfs_clipped(Tpmean_gdfs, niger5)



# --- 8. Interpolate Each Raster with IDW and Nearest ---
titles = ["SSP126 NF", "SSP245 NF", "SSP585 NF", "SSP126 FF", "SSP245 FF", "SSP585 FF"]
keys = ["SSP126_near", "SSP245_near", "SSP585_near", "SSP126_far", "SSP245_far", "SSP585_far"]
interpolatedBas = []  # renamed from interpolated

for key in keys:
    r = rasters_dict[f"RasTp_{key}"].astype(float)
    filled = idw_fill(r)
    filled[mask == 0] = np.nan
    ys, xs = np.where(~np.isnan(filled))
    vals = filled[ys, xs]
    grid_y, grid_x = np.mgrid[0:filled.shape[0], 0:filled.shape[1]]
    nn = griddata((xs, ys), vals, (grid_x, grid_y), method='nearest')
    filled = np.where(np.isnan(filled), nn, filled)
    interpolatedBas.append(filled)

# --- 9. Setup Coordinate Grid for Plot ---
x_coords = np.linspace(minx, maxx, width)
y_coords = np.linspace(miny, maxy, height)
X, Y = np.meshgrid(x_coords, y_coords[::-1])
extent = (minx, maxx, miny, maxy)

poly_coords = niger5.geometry.iloc[0].exterior.coords.xy
poly_path = Path(list(zip(poly_coords[0], poly_coords[1])))

# --- 10. Plot Results ---
fig, axs = plt.subplots(2, 3, figsize=(12, 10), layout='compressed')
vmin, vmax = np.nanmin(interpolatedBas), np.nanmax(interpolatedBas)

for ax, img, title in zip(axs.flat, interpolatedBas, titles):
    im = ax.imshow(
        img, cmap="inferno", vmin=vmin, vmax=vmax,
        interpolation='bilinear', extent=extent, origin='upper'
    )
    im.set_clip_path(PathPatch(poly_path, transform=ax.transData, facecolor='none'))
    ax.set_title(title, fontsize=22)
    ax.axis("off")

cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.05)
cbar.set_label("Mean Annual Temperature (°C)", fontsize=22)
cbar.ax.tick_params(labelsize=20)

plt.show()


#Use in paper
import matplotlib.colors as colors
from matplotlib.colors import PowerNorm
Res_predictions_by_sspTpAdj = {
    k: v.mask(v == 0, 0) for k, v in Res_predictions_by_sspTp.items()
}

# -------------------- Temporal Aggregation & Ensemble -------------------- #
def compute_ensemble(ssp_label):
    return pd.concat([
        Res_predictions_by_sspTpAdj[k]
        for k in Res_predictions_by_sspTpAdj if ssp_label in k
    ]).groupby(level=0).mean()

Tpmean_ensemble_by_sspRes = {
    'SSP126': compute_ensemble('SSP126'),
    'SSP245': compute_ensemble('SSP245'),
    'SSP585': compute_ensemble('SSP585')
}

# -------------------- Period Averaging -------------------- #
near_start, near_end = "2026-01-01", "2050-12-31"
far_start, far_end   = "2051-01-01", "2080-12-31"
Tpmean_by_ssp_period_annualRes = {}

for ssp, df in Tpmean_ensemble_by_sspRes.items():
    df.index = pd.to_datetime(df.index)
    df_annual = df.resample('YE').mean()
    Tpmean_by_ssp_period_annualRes[f"{ssp}_near"] = df_annual.loc[near_start:near_end].mean(axis=0)
    Tpmean_by_ssp_period_annualRes[f"{ssp}_far"]  = df_annual.loc[far_start:far_end].mean(axis=0)

# -------------------- Convert to GeoDataFrames -------------------- #
b['FID'] = b['FID'].astype(int)

def series_to_gdf(series, b_gdf):
    df = series.rename_axis('Station').reset_index(name='Mean_Temp')
    df['FID'] = df['Station'].str.extract(r'T(\d+)').astype(int)
    merged = df.merge(b_gdf[['FID', 'geometry']], on='FID', how='left')
    return gpd.GeoDataFrame(merged, geometry='geometry', crs=b_gdf.crs)

Tpmean_gdfsRes = {
    key: series_to_gdf(series, b)
    for key, series in Tpmean_by_ssp_period_annualRes.items()
}

for gdf in Tpmean_gdfsRes.values():
    gdf['Mean_Temp'] = gdf['Mean_Temp'].round(2)

print("GeoDataFrames saved successfully.")

# -------------------- Raster Creation -------------------- #
def create_rasters_from_gdfs_clipped(gdf_dict, resolution=1000):
    rasters = {}
    target_crs = "EPSG:32632"
    boundary_gdf = reservoir.to_crs(target_crs)
    bounds = boundary_gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    width, height = int((maxx - minx) / resolution), int((maxy - miny) / resolution)
    transform = from_origin(minx, maxy, resolution, resolution)

    boundary_mask = rasterize(
        [(geom, 1) for geom in boundary_gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    for key, gdf in gdf_dict.items():
        gdf = gdf.to_crs(target_crs)
        gdf = gdf[gdf.geometry.within(boundary_gdf.unary_union)]
        shapes = [
            (geom, value) for geom, value in zip(gdf.geometry, gdf["Mean_Temp"]) if value > 0
        ]
        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            fill=np.nan,
            transform=transform,
            dtype='float32'
        )
        raster[boundary_mask == 0] = np.nan
        rasters[f"RasTp_{key}"] = raster

    return rasters, transform, target_crs

# -------------------- Get Raster Arrays -------------------- #
RasTp_SSP126_near = rasters_dict["RasTp_SSP126_near"]
RasTp_SSP245_near = rasters_dict["RasTp_SSP245_near"]
RasTp_SSP585_near = rasters_dict["RasTp_SSP585_near"]
RasTp_SSP126_far  = rasters_dict["RasTp_SSP126_far"]
RasTp_SSP245_far  = rasters_dict["RasTp_SSP245_far"]
RasTp_SSP585_far  = rasters_dict["RasTp_SSP585_far"]

# -------------------- Prepare Rasterization Mask -------------------- #
target_crs = "EPSG:32632"
reservoir = reservoir.to_crs(target_crs)
boundary_poly = unary_union(reservoir.geometry)

minx, miny, maxx, maxy = reservoir.total_bounds
res = 1000
width, height = int((maxx - minx) / res), int((maxy - miny) / res)
transform = from_origin(minx, maxy, res, res)

mask = rasterize(
    [(boundary_poly, 1)],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype='uint8'
)

# -------------------- IDW Fill Function -------------------- #
def idw_fill(raster, power=2):
    ys, xs = np.where(~np.isnan(raster))
    vals = raster[ys, xs]
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    pts = np.vstack((xs, ys)).T
    flat_grid = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    d = cdist(flat_grid, pts)
    d[d == 0] = np.nan
    w = 1 / (d**power)
    w /= np.nansum(w, axis=1)[:, None]
    filled = np.dot(w, vals).reshape((height, width))
    return filled

# -------------------- Interpolation and Nearest Fill -------------------- #
titles = ["SSP126 NF","SSP245 NF","SSP585 NF","SSP126 FF","SSP245 FF","SSP585 FF"]
keys   = ["SSP126_near","SSP245_near","SSP585_near","SSP126_far","SSP245_far","SSP585_far"]
interpolated = []

for key in keys:
    r = globals()[f"RasTp_{key}"].astype(float)
    filled = idw_fill(r)
    filled[mask == 0] = np.nan

    ys, xs = np.where(~np.isnan(filled))
    vals = filled[ys, xs]
    grid_y, grid_x = np.mgrid[0:filled.shape[0], 0:filled.shape[1]]
    nn = griddata((xs, ys), vals, (grid_x, grid_y), method='nearest')
    filled = np.where(np.isnan(filled), nn, filled)
    interpolated.append(filled)
    interpolatedRes = interpolated

# -------------------- Plotting -------------------- #
poly_coords = np.array(boundary_poly.exterior.coords)
poly_path = Path(poly_coords)
extent = [minx, maxx, miny, maxy]

fig, axs = plt.subplots(1, 6, figsize=(12, 10), layout='compressed')
all_data = np.concatenate([img.ravel() for img in interpolatedRes])
norm = colors.PowerNorm(gamma=0.5, vmin=all_data.min(), vmax=all_data.max())

for ax, img, title in zip(axs.flat, interpolatedRes, titles):
    im = ax.imshow(img, cmap='jet', norm=norm, interpolation='bilinear',
                   extent=extent, origin='upper')
    ax.set_title(title, fontsize=22)
    ax.axis("off")

    patch = PathPatch(poly_path, transform=ax.transData, facecolor='none')
    im.set_clip_path(patch)

cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal",
                    fraction=0.05, pad=0.05)
cbar.set_label("Mean Annual Temp (°C)", fontsize=22)
cbar.ax.tick_params(labelsize=20)

#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\11SpatialFutTmpResEnlarged.png", dpi = 500)
plt.show()

EnsResTempDF = (ResDFCanESM5histMnthTemp1990_2014_regriddedF +
    ResDFACCESS_CM2histMnthTemp1990_2014_regriddedF +
    ResDFGFDL_ESM4histMnthTemp1990_2014_regriddedF 
) / 3

# annual
EnsResTempDF_yr = EnsResTempDF.resample('YE').mean()

EnsResTempDF_yr.index = EnsResTempDF_yr.index.year
EnsResTempDF_yr = EnsResTempDF_yr.round(2)

EnsResTempDF_yrM= EnsResTempDF_yr.mean(axis=0)
#EnsBasTempDF_yrM +=4.2
df_temp = EnsResTempDF_yrM.reset_index()
df_temp.columns = ['Station', 'MeanAnnualTemp']
df_temp['FID'] = df_temp['Station'].str.extract(r'T(\d+)').astype(int)
gdf_merged = pd.merge(df_temp, b, on='FID')
gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry='geometry')
MeanResTemp = gdf_merged[['Station', 'FID', 'MeanAnnualTemp', 'geometry']]
#MeanBasTemp

#####################################################################################
# Convert to GeoDataFrame if not already
MeanResTemp = gpd.GeoDataFrame(MeanResTemp, geometry='geometry', crs="EPSG:4326")

MeanResTemp = MeanResTemp.to_crs("EPSG:32632")


target_crs = "EPSG:32632"
reservoir= reservoir.to_crs(target_crs)
boundary_poly = unary_union(reservoir.geometry)

minx, miny, maxx, maxy = reservoir.total_bounds
res = 1000
width, height = int((maxx - minx) / res), int((maxy - miny) / res)
transform = from_origin(minx, maxy, res, res)
mask = rasterize([(boundary_poly, 1)], out_shape=(height, width), transform=transform, fill=0, dtype='uint8')

#Rasterize points
filtered = MeanResTemp[MeanResTemp.geometry.within(boundary_poly)]
shapes = [(geom, val) for geom, val in zip(filtered.geometry, filtered['MeanAnnualTemp']) if val > 0]
raster = rasterize(shapes, out_shape=(height, width), fill=np.nan, transform=transform, dtype='float32')
raster[mask == 0] = np.nan

#IDW
def idw_fill(raster, power=2):
    ys, xs = np.where(~np.isnan(raster))
    vals = raster[ys, xs]
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    flat_grid = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    pts = np.vstack((xs, ys)).T
    d = cdist(flat_grid, pts)
    d[d == 0] = np.nan
    w = 1 / (d ** power)
    w /= np.nansum(w, axis=1)[:, None]
    return np.dot(w, vals).reshape((height, width))

filledRes = idw_fill(raster)
filledRes[mask == 0] = np.nan


#NearestN
ys, xs = np.where(~np.isnan(filledRes))
vals = filledRes[ys, xs]
grid_y, grid_x = np.mgrid[0:height, 0:width]
nn = griddata((xs, ys), vals, (grid_x, grid_y), method='nearest')
filledRes = np.where(np.isnan(filledRes), nn, filledRes)

#########################################################################################
x_coords = np.linspace(minx, maxx, width)
y_coords = np.linspace(miny, maxy, height)
X, Y = np.meshgrid(x_coords, y_coords[::-1])
extent = (minx, maxx, miny, maxy)

poly_coords = reservoir.geometry.iloc[0].exterior.coords.xy
poly_path = Path(list(zip(poly_coords[0], poly_coords[1])))

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    filledRes, cmap="inferno", vmin=np.nanmin(filledRes), vmax=np.nanmax(filledRes),
    interpolation='bilinear', extent=extent, origin='upper'
)
im.set_clip_path(PathPatch(poly_path, transform=ax.transData, facecolor='none'))
ax.set_title("Mean Annual Temperature (Baseline)", fontsize=20)
ax.axis("off")

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.05)
cbar.set_label("Temperature (°C)", fontsize=22)
cbar.ax.tick_params(labelsize=15)


#RVI
#ResRVI
RVI_listRes = []
for ssp_raster in interpolatedRes:
    with np.errstate(divide='ignore', invalid='ignore'):
        rvi = np.where(
            (filledRes > 0) & (~np.isnan(filledRes)) & (~np.isnan(ssp_raster)),
            ((ssp_raster / filledRes) - 1)*100,
            np.nan
        )
    RVI_listRes.append(rvi)

# Plot RVI results
titles_rvi = [
    "SSP126 NF",
    "SSP245 NF",
    "SSP585 NF",
    "SSP126 FF",
    "SSP245 FF",
    "SSP585 FF"
]

fig, axs = plt.subplots(2, 3, figsize=(12, 10))
fig.patch.set_facecolor('white')

vmin, vmax = np.nanpercentile(np.array(RVI_listRes), 5), np.nanpercentile(np.array(RVI_listRes), 95)

for ax, img, title in zip(axs.flat, RVI_listRes, titles_rvi):
    im = ax.imshow(
        img, cmap="gist_stern_r", vmin=vmin, vmax=vmax,  # RVI colormap example
        interpolation='nearest', extent=extent, origin='upper'
    )
    im.set_clip_path(PathPatch(poly_path, transform=ax.transData, facecolor='none'))
    ax.set_title(title, fontsize=24)
    ax.axis("off")

cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.05)
cbar.set_label("Relative Variation Index of Annual Temperature, RVI (%)", fontsize=22)
cbar.ax.tick_params(labelsize=21)

plt.show()


#Reservoir precip
#  Compute Ensemble Mean for Each SSP ---
def compute_ensemble_mean(ssp_label):
    return pd.concat([
        v for k, v in Res_predictions_by_ssp.items()
        if f"{ssp_label}_26_80" in k
    ]).groupby(level=0).mean()

Prmean_ensemble_by_sspRes = {
    ssp: compute_ensemble_mean(ssp) for ssp in ['SSP126', 'SSP245', 'SSP585']
}

# --- 3. Annual Mean per SSP Period ---
periods = {
    'near': ("2027-01-01", "2050-12-31"),
    'far': ("2051-01-01", "2080-12-31")
}

Prmean_by_ssp_period_annualRes = {}
for ssp, df in Prmean_ensemble_by_sspRes.items():
    df.index = pd.to_datetime(df.index)
    annual = df.resample('YE').sum() #############annual totals
    for label, (start, end) in periods.items():
        Prmean_by_ssp_period_annualRes[f"{ssp}_{label}"] = annual.loc[start:end].mean(axis=0)

# --- 4. Convert Series to GeoDataFrame ---
def series_to_gdf(series, base_gdf):
    df = series.rename_axis('Station').reset_index(name='Precip_Totals')
    df['FID'] = df['Station'].str.extract(r'P(\d+)').astype(int)
    merged = df.merge(base_gdf[['FID', 'geometry']], on='FID', how='left')
    return gpd.GeoDataFrame(merged, geometry='geometry', crs=base_gdf.crs)

b['FID'] = b['FID'].astype(int)
Prmean_gdfsRes = {key: series_to_gdf(series, b) for key, series in Prmean_by_ssp_period_annualRes.items()}
for gdf in Prmean_gdfsRes.values():
    gdf['Precip_Totals'] = gdf['Precip_Totals'].round(2)

# --- 5. Prepare Boundary and Grid ---
target_crs = "EPSG:32632"
reservoir = reservoir.to_crs(target_crs)
boundary_poly = unary_union(reservoir.geometry)

minx, miny, maxx, maxy = reservoir.total_bounds
res, width, height = 1000, int((maxx - minx) / 1000), int((maxy - miny) / 1000)
transform = from_origin(minx, maxy, res, res)

mask = rasterize([(boundary_poly, 1)], out_shape=(height, width), transform=transform, fill=0, dtype='uint8')

#########################################
# --- 6. IDW Interpolation Function ---
def idw_fill(raster, power=2):
    ys, xs = np.where(~np.isnan(raster))
    vals = raster[ys, xs]
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    flat_grid = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    pts = np.vstack((xs, ys)).T
    d = cdist(flat_grid, pts)
    d[d == 0] = np.nan
    w = 1 / (d ** power)
    w /= np.nansum(w, axis=1)[:, None]
    return np.dot(w, vals).reshape((height, width))

# --- 7. Create Raster From GDFs ---
def create_rasters_from_gdfs_clipped(gdf_dict, boundary_gdf, resolution=1000):
    boundary_gdf = boundary_gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    width, height = int((maxx - minx) / resolution), int((maxy - miny) / resolution)
    transform = from_origin(minx, maxy, resolution, resolution)
    boundary_mask = rasterize(
        [(geom, 1) for geom in boundary_gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    rasters = {}
    for key, gdf in gdf_dict.items():
        gdf = gdf.to_crs(target_crs)
        gdf = gdf[gdf.geometry.within(boundary_gdf.union_all())]
        shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf['Precip_Totals']) if val > 0]
        raster = rasterize(shapes, out_shape=(height, width), fill=np.nan, transform=transform, dtype='float32')
        raster[boundary_mask == 0] = np.nan
        rasters[f"RasPr_{key}"] = raster
    return rasters, transform, target_crs

rasters_dict, transform, target_crs = create_rasters_from_gdfs_clipped(Prmean_gdfsRes, reservoir)

# --- 8. Interpolate Each Raster with IDW and Nearest ---
titles = ["SSP126 NF", "SSP245 NF", "SSP585 NF", "SSP126 FF", "SSP245 FF", "SSP585 FF"]
keys = ["SSP126_near", "SSP245_near", "SSP585_near", "SSP126_far", "SSP245_far", "SSP585_far"]
interpolated = []

for key in keys:
    r = rasters_dict[f"RasPr_{key}"].astype(float)
    filled = idw_fill(r)
    filled[mask == 0] = np.nan
    ys, xs = np.where(~np.isnan(filled))
    vals = filled[ys, xs]
    grid_y, grid_x = np.mgrid[0:filled.shape[0], 0:filled.shape[1]]
    nn = griddata((xs, ys), vals, (grid_x, grid_y), method='nearest')
    filled = np.where(np.isnan(filled), nn, filled)
    interpolated.append(filled)

# --- 9. Setup Coordinate Grid for Plot ---
x_coords = np.linspace(minx, maxx, width)
y_coords = np.linspace(miny, maxy, height)
X, Y = np.meshgrid(x_coords, y_coords[::-1])
extent = (minx, maxx, miny, maxy)

poly_coords = reservoir.geometry.iloc[0].exterior.coords.xy
poly_path = Path(list(zip(poly_coords[0], poly_coords[1])))

##############################################
# --- 10. Plot Results ---
fig, axs = plt.subplots(1, 6, figsize=(12, 10), layout='compressed')
vmin, vmax = np.nanmin(interpolated), np.nanmax(interpolated)

for ax, img, title in zip(axs.flat, interpolated, titles):
    im = ax.imshow(
        img, cmap="gist_rainbow", vmin=vmin, vmax=vmax,
        interpolation='bilinear', extent=extent, origin='upper'
    )
    im.set_clip_path(PathPatch(poly_path, transform=ax.transData, facecolor='none'))
    ax.set_title(title, fontsize=18)
    ax.axis("off")

cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.05)
cbar.set_label("Annual Precipitation Totals (mm)", fontsize=16)
cbar.ax.tick_params(labelsize=14)

# Show or Save Plot
# plt.savefig("Interpolated_Temperature_Maps.png", dpi=500, bbox_inches='tight')
plt.show()
