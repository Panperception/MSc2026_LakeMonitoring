# Minimal runtime dependencies for geemap/Earth Engine in Colab
#!pip install earthengine-api geemap -q

# Core imports (Earth Engine I/O, scheduling, CSV logging, Colab file ops)
import ee, geemap, gc
import time, datetime
import os, math, re
import numpy as np
import pandas as pd
import shutil, rasterio
from tqdm import tqdm
from google.colab import files
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Authenticate and initialise Earth Engine (first run will prompt authorisation)
ee.Authenticate()
ee.Initialize(project='lake-study-2026')  # Replace with your project ID

#lake_info = candidate_lakes.getInfo()['features']

# Target summer windows; years and months to iterate (boreal by default; austral handled in get_daily_dates)
years = list(range(2000, 2027))
months = [6, 7, 8]
PIXEL_SIZE = 1024     # Set target pixel size

# Generate all dates for a given (year, base_month) with hemisphere adjustment for southern lakes
def get_daily_dates(lat, base_year, base_month):
    if lat < 0:
        base_month = (base_month + 6 - 1) % 12 + 1
        if base_month < 6:
            base_year += 1
    start = datetime.date(base_year, base_month, 1)
    end = datetime.date(base_year + int(base_month == 12), (base_month % 12) + 1, 1)
    return [start + datetime.timedelta(days=i) for i in range((end - start).days)]

# Median composite helper with scale-to-8bit display; returns None if the collection is empty
def get_median_image(collection, bands, scale_factor):
    if collection.size().getInfo() == 0:
        return None
    return collection.median().select(bands).divide(scale_factor).multiply(255).clamp(0, 255).uint8()

# Accumulate per-date status for a final CSV log
all_results = []
lake_scale = {}
cwd_dir = os.getcwd()
lake_images_dir = os.path.join(cwd_dir, 'lake_images')
os.makedirs(lake_images_dir, exist_ok=True)

for lake_id, lake_coords in lake_info:
    lake_region = ee.Geometry.Rectangle(lake_coords)
    print(f"• {lake_id} : {lake_coords}")

    base_folder = lake_id.replace(" ", "_")
    base_dirpath = os.path.join(lake_images_dir, base_folder)
    os.makedirs(base_dirpath, exist_ok=True)

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
    lake_scale[lake_id] = [width, height]

    # Estimate latitude centre from lake corrdinates to choose seasonal window
    lat_center = (lake_coords[1] + lake_coords[3]) / 2

    for year in years:
        for base_month in months:
            date_list = get_daily_dates(lat_center, year, base_month)
            if not date_list: continue

            actual_year = date_list[0].year
            actual_month = date_list[0].month
            month_folder = os.path.join(base_dirpath, f"{lake_id.replace(' ', '_')}-{str(actual_year)[-2:]}-{actual_month:02d}")
            os.makedirs(month_folder, exist_ok=True)

            for date in date_list:
                start = date.strftime("%Y-%m-%d")
                end = (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                print(f"📅 {lake_id} — {start}")

                # Daily collections with simple cloud thresholds (S2/L8) and MODIS fallback
                s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                    .filterBounds(lake_region).filterDate(start, end) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
                l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                    .filterBounds(lake_region).filterDate(start, end) \
                    .filter(ee.Filter.lt('CLOUD_COVER', 60))
                mod = ee.ImageCollection("MODIS/061/MOD09GA") \
                    .filterBounds(lake_region).filterDate(start, end)

                # Convert to 8-bit display composites
                s2_img = get_median_image(s2, ['B4', 'B3', 'B2'], 3000)
                l8_img = get_median_image(l8, ['SR_B4', 'SR_B3', 'SR_B2'], 10000)
                mod_img = get_median_image(mod, ['sur_refl_b01', 'sur_refl_b04', 'sur_refl_b03'], 5000)

                # Fuse by priority (S2 → L8 → MODIS) using unmask where available
                fused = None
                if s2_img:
                    fused = s2_img
                    if l8_img: fused = fused.unmask(l8_img)
                    if mod_img: fused = fused.unmask(mod_img)
                elif l8_img:
                    fused = l8_img
                    if mod_img: fused = fused.unmask(mod_img)
                elif mod_img:
                    fused = mod_img

                if not fused:
                    all_results.append((lake_id, start, "⚠️ No image"))
                    continue

                # Save daily GeoTIFF into the month subfolder
                tif_name = f"{lake_id.replace(' ', '_')}_{date.strftime('%Y_%m_%d')}.tif"
                tif_path = os.path.join(month_folder, tif_name)

                try:
                    geemap.download_ee_image(
                        image=fused,
                        filename=tif_path,
                        region=lake_region,
                        scale=optimal_scale,
                        crs='EPSG:4326'
                    )
                    all_results.append((lake_id, start, "Success"))
                except Exception as e:
                    all_results.append((lake_id, start, f"Download failed: {str(e)}"))

    # Zip the current lake folder and trigger a download to local
    zip_name = f"{base_folder}.zip"
    shutil.make_archive(base_dirpath, 'zip', base_dirpath)
    files.download(os.path.join(lake_images_dir,zip_name))

# Write a simple run log to CSV and show the first few entries
df_result = pd.DataFrame(all_results, columns=["Lake", "Date", "Status"])
log_name = "multi_lake_download_log.csv"
df_result.to_csv(log_name, index=False)
print("\nDownload complete, first 10 logs:")
df_result.head(10)

print("\nAll lakes processed")


# Configuration: input/output roots and parallel/batch settings
input_root = lake_images_dir
output_root = os.path.join(base_dir,'data_images')
os.makedirs(output_root, exist_ok=True)

batch_size = 200
max_workers = 2

# Image scoring heuristic:
# - reject frames with NaN/invalid shapes or very dark/low-contrast centres
# - penalise bright grey (cloud-like) and saturated white pixels
# - lower score is better; return (-score, reason)
def image_score(img):
    # Basic dimensional checks (expects at least 3 bands in CHW order)
    if img.shape[0] < 3 or img.shape[1] == 0 or img.shape[2] == 0:
        return -1, "❌ Insufficient channels or zero dimension"

    # Reject frames with NaN values
    if np.isnan(img).any():
        return -1, "❌ Image contains NaN"

    # Split channels and compute brightness
    r = img[0].astype(np.float32)
    g = img[1].astype(np.float32)
    b = img[2].astype(np.float32)
    brightness = (r + g + b) / 3
    h, w = brightness.shape

    # Central window statistics for basic quality control
    ch, cw = h // 4, w // 4
    center_r = r[ch:3 * ch, cw:3 * cw]
    center_g = g[ch:3 * ch, cw:3 * cw]
    center_b = b[ch:3 * ch, cw:3 * cw]
    center_brightness = (center_r + center_g + center_b) / 3

    center_mean = np.mean(center_brightness)
    center_std = np.std(center_brightness)

    # Cloud proxy: bright and near-grey pixels
    grayish = (np.abs(r - g) < 25) & (np.abs(r - b) < 25) & (np.abs(g - b) < 25)
    cloud_mask = (brightness > 220) & grayish
    cloud_ratio = np.sum(cloud_mask) / brightness.size

    # Proportions of saturated white and near-black pixels
    white_ratio = np.sum(brightness > 245) / brightness.size
    black_ratio = np.sum(brightness < 10) / brightness.size

    # Hard filters for centre quality and overall exposure
    if center_mean < 30:
        return -1, "Center brightness too low"
    if center_std < 5:
        return -1, "Center region contrast too low"
    if black_ratio > 0.2:
        return -1, "Too many black pixels"
    if np.max(brightness) < 50:
        return -1, "Overall brightness too low"

    # Aggregate score: clouds weighted more than specular whites
    score = cloud_ratio + 0.5 * white_ratio
    return -score, "Qualified"

# Evaluate all .tif files in a folder and export the best candidate as PNG
def process_folder(folder):
    try:
        tif_files = [f for f in os.listdir(folder) if f.lower().endswith(".tif")]
        if not tif_files:
            return f"No images: {folder}"

        best_score = float("-inf")
        best_img = None
        best_reason = ""

        for tif_name in tif_files:
            tif_path = os.path.join(folder, tif_name)
            with rasterio.open(tif_path) as src:
                img = src.read()
                if img.shape[0] < 3 or img.shape[1] == 0 or img.shape[2] == 0:
                    continue

                score, reason = image_score(img)
                if score > best_score:
                    best_score = score
                    best_img = img
                    best_reason = reason

        if best_img is None or best_score == -1:
            return f"No valid image: {folder} (Reason: {best_reason})"

        # Convert first three bands to uint8 RGB and save
        rgb = best_img[:3].astype(np.uint8)
        rgb = np.transpose(rgb, (1, 2, 0))
        img_pil = Image.fromarray(rgb)

        # Expect folder names like "<lake>-<YY>-<MM>"; parse to construct filename
        folder_name = os.path.basename(folder)
        match = re.match(r"(.*)-(\d{2})-(\d{2})", folder_name)
        if match:
            lake, year, month = match.groups()
        else:
            return f"Unable to parse: {folder_name}"

        filename = f"{lake}_{year}_{month}_img.png"
        out_path = os.path.join(output_root, filename)
        img_pil.save(out_path)

        del best_img, rgb, img_pil
        gc.collect()
        return f"Saved: {filename} (Score: {best_score:.3f})"

    except Exception as e:
        return f"Error: {folder}\n{e}"

# Batch execution over subfolders of input_root using a thread pool
all_folders = [
    os.path.join(input_root, lake, year)
    for lake in os.listdir(input_root)
    if os.path.isdir(os.path.join(input_root, lake))
    for year in os.listdir(os.path.join(input_root, lake))
    if os.path.isdir(os.path.join(input_root, lake, year))
]
total = len(all_folders)
print(f"Total to process {total} folders, {batch_size} per batch")

start_all = time.time()

for i in range(0, total, batch_size):
    batch = all_folders[i:i + batch_size]
    print(f"\n🚀 Starting batch {i//batch_size + 1}, total {len(batch)}")

    start_batch = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_folder, folder): folder for folder in batch}
        for future in tqdm(as_completed(futures), total=len(futures), desc="📷 Processing"):
            result = future.result()
            print(result)
            with open("image_selection_log.txt", "a") as log:
                log.write(result + "\n")

    print(f"Current batch time: {(time.time() - start_batch):.2f} seconds")

print(f"\nAll processing completed, total time: {(time.time() - start_all)/60:.2f} minutes")