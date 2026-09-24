import os, glob
import numpy as np
import tensorflow as tf
import cv2, keras_cv
from tqdm import tqdm
import pandas as pd

lake_count = 100
years = list(range(2000, 2027))

# Paths to trained models
UNETPP_PATH = f'{base_dir}/model/unet++_model_final.keras'
DEEPLAB_PATH = f'{base_dir}/model/deeplab_model_final.keras'
SEGFORMER_PATH = f'{base_dir}/model/segformer_model_final.keras'

# Directory locations
IMG_DIR = f'{base_dir}/data_images/'
PRED_DIR = os.path.join(base_dir,'predictions')
os.makedirs(PRED_DIR, exist_ok=True)

IMG_SIZE = (256,256)
BATCH = 64  # batch size used for inference

print("IMG_DIR:", IMG_DIR)
print("Tuned Threshold:")
print(f"Unet++={t_unetpp}, Deeplab={t_deeplab}, Segformer={t_segformer}, Ensemble={t_ensemble}")
print("Ensemble Weights:")
print(f"Unet++={wUpp}, Deeplab={wDL}, Segformer={wSF}")

# Load images; resize to IMG_SIZE, normalise to [0,1]
def load_data(img_dir, mask_dir, img_size=(256,256)):
    files = sorted(glob.glob(os.path.join(img_dir, '*')))
    X, y = [], []
    for p in tqdm(files, desc=f'Loading {os.path.basename(os.path.normpath(img_dir))}'):
        name = os.path.basename(p)
        im = cv2.imread(p)
        if im is None: continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, img_size).astype(np.float32)/255.0
        X.append(im)
    X = np.asarray(X, np.float32)
    return X, files

# Load cached predictions if available; otherwise run inference and cache
def get_or_predict(model_path, X, cache_name):
    npy = os.path.join(PRED_DIR, cache_name + ".npy")
    if os.path.exists(npy):
        return np.load(npy)
    custom_maps = {**keras_cv.models.__dict__, **keras_cv.layers.__dict__}
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_maps)
    P = model.predict(X, batch_size=BATCH, verbose=0)
    if P.shape[-1] == 1:
        P = P[..., 0].astype(np.float32)
    elif P.shape[-1] == 2:
        P = P[..., 1].astype(np.float32)
    np.save(npy, P)
    return P

# Load image set once
x_data, files = load_data(IMG_DIR, IMG_SIZE)

# Obtain per-pixel probabilities for each model (from cache or inference)
p_unetpp = get_or_predict(UNETPP_PATH,  x_data, 'pred_unetpp')
p_deeplab = get_or_predict(DEEPLAB_PATH,  x_data, 'pred_deeplab')
p_segformer = get_or_predict(SEGFORMER_PATH,  x_data, 'pred_segformer')

# Obtain ensemble prediction
p_ensemble = wUpp*p_unetpp + wDL*p_deeplab + wSF*p_segformer
y_ensemble = (p_ensemble > t_ensemble).astype(np.uint8)
y_ens_dict = {os.path.basename(f): pred for f, pred in zip(files, y_ensemble)}

rows = []
# Loop over all lakes
for i in range(1, lake_count + 1):
    lake_id = f"lake_{i}"
    width, height = lake_scale[lake_id][0], lake_scale[lake_id][1]
    # Loop over range of years
    for year in sorted(years):
        year_id = str(year)[-2:]
        img_prefix = f"{lake_id}_{year_id}"  
        lake_images = [f for f in os.listdir(IMG_DIR) if f.startswith(img_prefix) and f.endswith((".png", ".jpg"))]
        if len(lake_images) == 0:
          continue
        
        # Combine monthly predictions using pixel-wise majority voting to obtain annual lake binary mask 
        monthly_pred = [y_ens_dict[f] for f in lake_images]
        if len(lake_images) == 3:
            yearly_pred = (np.sum(monthly_pred, axis=0) >= 2).astype(np.uint8)
        elif len(lake_images) == 2:
            yearly_pred = (np.sum(monthly_pred, axis=0) >= 1).astype(np.uint8)
        else:
            yearly_pred = monthly_pred[0]

        # Find contours within predicted lake region
        contours, _ = cv2.findContours(yearly_pred.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # Extract largest lake contour
        if len(contours) > 0:
            lake_contour = max(contours, key=cv2.contourArea)
            lake_pixels = cv2.contourArea(lake_contour)
        else:
            lake_pixels = 0
        
        # Lake Area = Number of lake(water) pixels x Area covered by one pixel
        total_pixels = yearly_pred.shape[0] * yearly_pred.shape[1]
        rect_area = width * height
        pixel_area = rect_area / total_pixels
        lake_area = lake_pixels * pixel_area
        lake_area = round(lake_area / 1e6, 4)

        rows.append([lake_id, year, lake_area])
        print(f"{lake_id} {year} : {lake_area} km2")

# Create dataframe and save as CSV file
df = pd.DataFrame(rows, columns=["lake","year","area_km2"])
csv_file = f"{base_dir}/lake_area.csv"
df.to_csv(csv_file, index=False)
print(f"Lake area information saved at: {csv_file}")