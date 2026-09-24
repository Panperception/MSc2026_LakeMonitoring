# Earth Engine-driven batch export of monthly history (10 lakes per batch) + CSV index
import ee, os, time, csv, math
from tqdm import tqdm

# Authenticate and initialise Earth Engine (authorisation required on first run)
ee.Authenticate()
ee.Initialize(project='lake-study-2026')

# Export configuration and seasonal month sets
START_YEAR = 2000
END_YEAR = 2021
MONTHS_NORTH = [6, 7, 8]   # boreal summer
MONTHS_SOUTH = [12, 1, 2]  # austral summer (crosses year boundary)
#EXPORT_SCALE = 30
PIXEL_SIZE = 1024
LAKES_PER_BATCH = 10
SLEEP_BETWEEN_TASKS = 0.5
SLEEP_BETWEEN_BATCHES = 30
EXPORT_FOLDER = os.path.basename(base_dir)
INDEX_CSV = "gsw_export_index.csv"

# Create the index CSV; submit Drive export tasks in batches
with open(INDEX_CSV, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["lake_name", "year", "month", "filename", "folder"])

    for batch_start in range(0, lake_count, LAKES_PER_BATCH):
        batch = lake_info[batch_start:batch_start + LAKES_PER_BATCH]
        print(f"Executing batch {batch_start//LAKES_PER_BATCH + 1}...")

        for lake_id, lake_coords in tqdm(batch):
            print(f"\n• {lake_id} : {lake_coords}")
            lake_region = ee.Geometry.Rectangle(lake_coords)
            lat_center = (lake_coords[1] + lake_coords[3]) / 2
            months = MONTHS_NORTH if lat_center >= 0 else MONTHS_SOUTH

            # Calculate dynamic optimal resolution
            # Extract structural corner points
            p_bottom_left  = ee.Geometry.Point([lake_coords[0], lake_coords[1]]) # [min_lng, min_lat]
            p_bottom_right = ee.Geometry.Point([lake_coords[2], lake_coords[1]]) # [max_lng, min_lat]
            p_top_right    = ee.Geometry.Point([lake_coords[2], lake_coords[3]]) # [max_lng, max_lat]
            # Calculate distance to compute width and height of bounding box
            width  = p_bottom_left.distance(right=p_bottom_right).getInfo()
            height = p_bottom_right.distance(right=p_top_right).getInfo()
            # Calculate optimal scale to fit lake inside the target pixel grid
            optimal_scale = math.ceil(max(width, height) / PIXEL_SIZE)
            optimal_scale = max(10, optimal_scale)      # Clamp to native sensor resolution limit
            print(f"Optimal scale = {optimal_scale}m for {PIXEL_SIZE}px image")

            for year in range(START_YEAR, END_YEAR + 1):
                for month in months:
                    # For southern hemisphere, Dec/Jan/Feb are attributed to the following calendar year
                    ym_year = year if not (lat_center < 0 and month < 3) else year + 1
                    date_str = f"{ym_year}-{month:02d}-01"

                    image = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory") \
                        .filterDate(date_str, ee.Date(date_str).advance(1, 'month')) \
                        .mosaic().select('water')
          
                    filename = f"GSW_MonthlyHistory_{lake_id}_{str(ym_year)[-2:]}_{month:02d}_img"

                    task = ee.batch.Export.image.toDrive(
                        image=image,
                        region=lake_region,
                        folder=EXPORT_FOLDER,
                        description=filename,
                        fileNamePrefix=filename,
                        scale=optimal_scale,
                        maxPixels=1e13
                    )
                    task.start()
                    writer.writerow([lake_id, ym_year, month, filename + ".tif", EXPORT_FOLDER])
                    print(f"Task submitted: {EXPORT_FOLDER}/{filename}")
                    time.sleep(SLEEP_BETWEEN_TASKS)

        print(f"Batch {batch_start//LAKES_PER_BATCH + 1} done, waiting {SLEEP_BETWEEN_BATCHES}s...\n")
        time.sleep(SLEEP_BETWEEN_BATCHES)

print("All tasks submitted. Check Earth Engine Task panel for progress.\nExported images will be automatically saved to your Google Drive → 'GSW_MonthlyHistory/<lake_name>/<year>/'\n📄 Index CSV file generated: gsw_export_index.csv")