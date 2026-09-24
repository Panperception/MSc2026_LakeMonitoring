import os, glob, zipfile, time, json, random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Canonical locations for test data and cached predictions
TEST_IMG_DIR  = f"{base_dir}/dataset/test/images/"
TEST_MASK_DIR = f"{base_dir}/dataset/test/masks/"
UNETPP_NPY    = f"{base_dir}/predictions/pred_unetpp_test.npy"
DEEPLAB_NPY   = f"{base_dir}/predictions/pred_deeplab_test.npy"
SEGFORMER_NPY = f"{base_dir}/predictions/pred_segformer_test.npy"

plot_dir = os.path.join(base_dir,'plots')
os.makedirs(plot_dir, exist_ok=True)

print("IMG_DIR:", TEST_IMG_DIR)
print("MSK_DIR:", TEST_MASK_DIR)
print("NPY:", UNETPP_NPY, DEEPLAB_NPY, SEGFORMER_NPY)

def read_img_rgb(path, size=None):
    # Read BGR image, convert to RGB, optional resize, normalise to [0,1]
    img = cv2.imread(path);
    if img is None: raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size and img.shape[:2] != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32)/255.)

def read_mask_bin(path, size=None):
    # Read grayscale mask, optional resize, then binarise (0/1) at threshold 127
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(path)
    if size and m.shape[:2] != size:
        m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return (m>127).astype(np.uint8)

def load_pairs(img_dir, msk_dir):
    # Match images to masks by filename stem; support common extensions
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    pairs = []
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = None
        for ext in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"):
            p = os.path.join(msk_dir, stem+ext)
            if os.path.exists(p): mp = p; break
        if mp: pairs.append((ip, mp))
    if not pairs: raise RuntimeError("No (image, mask) pairs found; check directories and filenames.")
    return pairs

pairs = load_pairs(TEST_IMG_DIR, TEST_MASK_DIR)

# Load cached per-pixel probabilities for each model; assert alignment with pairs
pred_unetpp  = np.load(UNETPP_NPY).squeeze()
pred_deeplab = np.load(DEEPLAB_NPY).squeeze()
pred_segformer = np.load(SEGFORMER_NPY).squeeze()
N = len(pairs)
assert pred_unetpp.shape[0]==pred_deeplab.shape[0]==pred_segformer.shape[0]==N, "NPY counts must match number of samples"

# Choose representative samples for qualitative comparison
#idxs = [6, 14, 60, 94, 124] if N>=3 else [0]
#idxs = [8, 12, 65, 99, 111] if N>=3 else [0]
#idxs = [9, 24, 56, 78, 99, 123, 155, 175, 200, 219, 247, 248, 270, 296, 333, 358, 390] if N>=3 else [0]
#idxs = [410, 455, 496, 517, 544, 587, 622, 657, 690, 734] if N>=3 else [0]
#fig, axes = plt.subplots(len(idxs), 6, figsize=(24, 5*len(idxs)))
#if len(idxs)==1: axes = np.array([axes])
H,W = pred_unetpp.shape[1], pred_unetpp.shape[2]

dict_lakes = {}
for idx, (ip, mp) in enumerate(pairs):
    name = '_'.join(os.path.basename(ip).split('_')[:2])
    if name not in dict_lakes:
        dict_lakes[name] = []
    dict_lakes[name].append(idx)
#lake_indices = list(dict_lakes.items())[16:20]
indices = [16, 95]
lake_indices = [list(dict_lakes.items())[i] for i in indices]
fig, axes = plt.subplots(len(lake_indices), 6, figsize=(24, 5*len(lake_indices)))

print("Tuned Threshold:")
print(f"Unet++={t_unetpp}, Deeplab={t_deeplab}, Segformer={t_segformer}, Ensemble={t_ensemble}")
print("Ensemble Weights:")
print(f"Unet++={wUpp}, Deeplab={wDL}, Segformer={wSF}")

#for r, idx in enumerate(idxs):
for r, (lake, indices) in enumerate(lake_indices):
    idx = indices[1]
    ip, mp = pairs[idx]
    img = read_img_rgb(ip, (H,W))
    gt  = read_mask_bin(mp, (H,W))
    pUpp, pDL, pSF = pred_unetpp[idx], pred_deeplab[idx], pred_segformer[idx]
    pE = wUpp * pUpp + wDL * pDL + wSF * pSF
    mUpp = (pUpp>=t_unetpp).astype(np.uint8)
    mDL = (pDL>=t_deeplab).astype(np.uint8)
    mSF = (pSF>=t_segformer).astype(np.uint8)
    mE = (pE >= t_ensemble).astype(np.uint8)

    axes[r,0].imshow(img);            axes[r,0].set_title("Image");    axes[r,0].axis("off")
    axes[r,1].imshow(gt, cmap="gray");axes[r,1].set_title("GSW mask"); axes[r,1].axis("off")
    axes[r,2].imshow(mUpp, cmap="gray");axes[r,2].set_title("U-Net++");  axes[r,2].axis("off")
    axes[r,3].imshow(mDL, cmap="gray");axes[r,3].set_title("Deeplab");  axes[r,3].axis("off")
    axes[r,4].imshow(mSF, cmap="gray");axes[r,4].set_title("Segformer");  axes[r,4].axis("off")
    axes[r,5].imshow(mE, cmap="gray");axes[r,5].set_title(f"Ensemble\nw=({wUpp},{wDL},{wSF})"); axes[r,5].axis("off")

plt.tight_layout()
plt.savefig(f"{plot_dir}/Fig_lake_segmentation.png", dpi=300)
plt.show()
plt.close(fig)
print(f"Saved: {plot_dir}/Fig_lake_segmentation.png")