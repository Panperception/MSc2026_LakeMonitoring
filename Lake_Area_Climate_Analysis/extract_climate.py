import os, ee

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project="lake-study-2026")

# Data Source
ERA5_CLIMATE = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
               .select(['total_precipitation_sum', 'potential_evaporation_sum',
                'snow_depth', 'snowmelt_sum', 'temperature_2m']);

years = list(range(2000, 2027))

# Annual summer climate extraction per lake 
def getAnnualSummerClimate(lake_info):
    lake_id = lake_info[0]
    lake_coords = lake_info[1]
    lake_region = ee.Geometry.Rectangle(lake_coords)
    lat_center = (lake_coords[1] + lake_coords[3]) / 2
    feature_list = []

    for year in years:
        if lat_center >= 0:
            hemi = 'north'
            startDate = ee.Date.fromYMD(year, 6, 1)
        else:
            hemi = 'south'
            startDate = ee.Date.fromYMD(year - 1, 12, 1)
        endDate = startDate.advance(3, 'month')

        # ERA5 monthly aggregates over the summer; take mean across months
        climate = ERA5_CLIMATE.filterDate(startDate, endDate);
        climateMean = climate.mean();
        climateSum = climate.sum();

        # Extract variable means over buffered geometry (units converted below)
        precip = climateSum.select('total_precipitation_sum').reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=lake_region, maxPixels=1e13
                ).get('total_precipitation_sum')

        evap = climateSum.select('potential_evaporation_sum').reduceRegion(
                  reducer=ee.Reducer.mean(), geometry=lake_region, maxPixels=1e13
              ).get('potential_evaporation_sum')

        snow_depth = climateMean.select('snow_depth').reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=lake_region, maxPixels=1e13
                    ).get('snow_depth')

        snow_melt = climateSum.select('snowmelt_sum').reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=lake_region, maxPixels=1e13
                    ).get('snowmelt_sum')

        temp_K = climateMean.select('temperature_2m').reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=lake_region, maxPixels=1e13
                ).get('temperature_2m')

        temp_C = ee.Number(temp_K).subtract(273.15) # Kelvin → Celsius

        feature = ee.Feature(None,
                {'lake': lake_id,
                'hemisphere': hemi,
                'year': year,
                'temp_C': temp_C,
                'precip_mm': ee.Number(precip).multiply(1000),         # m → mm
                'evap_mm': ee.Number(evap).multiply(-1000),            # m → mm
                'snow_melt_mm': ee.Number(snow_melt).multiply(1000),   # m → mm
                'snow_depth_cm': ee.Number(snow_depth).multiply(100)   # m → cm
                })
        feature_list.append(feature)

    return ee.FeatureCollection(feature_list).filter(ee.Filter.notNull(['lake']))


# Batch run for all lakes 
allRows = ee.FeatureCollection([])
for lake in lake_info:
    allRows = allRows.merge(getAnnualSummerClimate(lake))

print('Total number of records:', allRows.size().getInfo());
print(allRows.limit(10).getInfo());

# Export to Google Drive
task = ee.batch.Export.table.toDrive(
    collection=allRows,
    description="Lake_Climate_AnnualSummer",
    folder=os.path.basename(base_dir),
    fileNamePrefix="lake_climate",
    fileFormat="CSV"
)
task.start()
print(f"Lake climate information saved at: {base_dir}/lake_climate.csv") 