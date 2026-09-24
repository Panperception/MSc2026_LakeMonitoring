import os, shutil, glob
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

# Converts a GSW MonthlyHistory GeoTIFF to a binary mask PNG only when water exists
def convert_to_mask_png_if_water(in_path, out_path):
    with rasterio.open(in_path) as src:
        image = src.read(1)

    # Water detection for classes {2, 254}; supports integer or float encodings
    has_water = np.any((image == 254) | (image == 2) | (np.isclose(image, 2.0)))
    if not has_water:
        print(f"No water pixels, skipped: {os.path.basename(in_path)}")
        return False

    print(f"Found water pixels : {os.path.basename(in_path)}")
    mask = np.where((image == 254) | (image == 2) | (np.isclose(image, 2.0)), 255, 0).astype(np.uint8)
    Image.fromarray(mask).save(out_path)
    return True

# Iterate over exported GeoTIFFs and create PNG masks into a single target folder
converted = 0
orig_root = os.path.join(base_dir,'download GSW data')
output_root = os.path.join(base_dir,'mask_images')
os.makedirs(orig_root, exist_ok=True)
os.makedirs(output_root, exist_ok=True)

for file in os.listdir(base_dir):
    in_path = f"{base_dir}/{file}"
    base_name = os.path.splitext(file)[0].replace("GSW_MonthlyHistory_", "")
    out_path = os.path.join(output_root, base_name + ".png")
    try:
        if convert_to_mask_png_if_water(in_path, out_path):
            converted += 1
    except Exception as e:
        print(f"Failed: {file} → {e}")

print("All PNG mask images generated and saved into a single mask folder!")

for file in glob.glob(base_dir + "/**/GSW_MonthlyHistory*.tif", recursive=True):
     shutil.move(file, orig_root)