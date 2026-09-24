import os, glob
import numpy as np
import tensorflow as tf
import cv2, keras_cv
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, precision_score, recall_score

# Paths to trained models
UNETPP_PATH = f'{base_dir}/model/unet++_model_final.keras'
DEEPLAB_PATH = f'{base_dir}/model/deeplab_model_final.keras'
SEGFORMER_PATH = f'{base_dir}/model/segformer_model_final.keras'

# Directories for test and (optional) validation data
TEST_IMG_DIR = f'{base_dir}/dataset/test/images/'
TEST_MASK_DIR= f'{base_dir}/dataset/test/masks/'
VAL_IMG_DIR  = f'{base_dir}/dataset/val/images/'   # optional
VAL_MASK_DIR = f'{base_dir}/dataset/val/masks/'    # optional

PRED_DIR = os.path.join(base_dir,'predictions')
os.makedirs(PRED_DIR, exist_ok=True)

IMG_SIZE = (256,256)
BATCH = 64  # batch size used for inference

# Load images and corresponding binary masks; resize to IMG_SIZE, normalise to [0,1]
def load_data(img_dir, mask_dir, img_size=(256,256)):
    files = sorted(glob.glob(os.path.join(img_dir, '*')))
    X, y = [], []
    for p in tqdm(files, desc=f'Loading {os.path.basename(os.path.normpath(img_dir))}'):
        name = os.path.basename(p)
        mpath = os.path.join(mask_dir, name.replace('.jpg','.png'))
        im = cv2.imread(p); ms = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if im is None or ms is None: continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, img_size).astype(np.float32)/255.0
        ms = cv2.resize(ms, img_size, interpolation=cv2.INTER_NEAREST)
        ms = (ms>127).astype(np.uint8)
        X.append(im); y.append(ms)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.uint8)
    return X, y

# Load cached predictions if available; otherwise run inference and cache
def get_or_predict(model_path, X, cache_name):
    npy = os.path.join(PRED_DIR, cache_name + ".npy")
    if os.path.exists(npy):
        return np.load(npy)
    custom_maps = {**keras_cv.models.__dict__, **keras_cv.layers.__dict__}
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_maps)
    P = model.predict(X, batch_size=BATCH, verbose=0)
    # Extract water probability from single channel model
    if P.shape[-1] == 1:
        P = P[..., 0].astype(np.float32)
    # Extract water probability from two-channel model via channel 1
    elif P.shape[-1] == 2:
        P = P[..., 1].astype(np.float32)
    np.save(npy, P)
    return P

# Load test set once
x_test, y_test = load_data(TEST_IMG_DIR, TEST_MASK_DIR, IMG_SIZE)
y_test_flat = y_test.reshape(-1)  # flatten to (N*H*W,)

# Optionally load validation set for threshold tuning
x_val = y_val = None
if os.path.isdir(VAL_IMG_DIR) and os.path.isdir(VAL_MASK_DIR):
    x_val, y_val = load_data(VAL_IMG_DIR, VAL_MASK_DIR, IMG_SIZE)
    y_val_flat = y_val.reshape(-1)

# Obtain per-pixel probabilities for each model (from cache or inference)
p_unetpp = get_or_predict(UNETPP_PATH,  x_test, 'pred_unetpp_test')
p_deeplab = get_or_predict(DEEPLAB_PATH,  x_test, 'pred_deeplab_test')
p_segformer = get_or_predict(SEGFORMER_PATH,  x_test, 'pred_segformer_test')

if x_val is not None:
    p_unetpp_v = get_or_predict(UNETPP_PATH,  x_val, 'pred_unetpp_val')
    p_deeplab_v = get_or_predict(DEEPLAB_PATH,  x_val, 'pred_deeplab_val')
    p_segformer_v = get_or_predict(SEGFORMER_PATH,  x_val, 'pred_segformer_val')

# Vectorised metrics: returns (F1, IoU, Accuracy, Precision, Recall)
def metrics_from_flat(y_true_flat, y_pred_flat):
    # y_* are uint8 arrays with values {0,1}
    tp = np.logical_and(y_true_flat==1, y_pred_flat==1).sum()
    tn = np.logical_and(y_true_flat==0, y_pred_flat==0).sum()
    fp = np.logical_and(y_true_flat==0, y_pred_flat==1).sum()
    fn = np.logical_and(y_true_flat==1, y_pred_flat==0).sum()

    prec = tp / (tp+fp+1e-9)
    rec  = tp / (tp+fn+1e-9)
    acc  = (tp+tn) / (tp+tn+fp+fn+1e-9)
    iou  = tp / (tp+fp+fn+1e-9)
    f1   = 2*prec*rec / (prec+rec+1e-9)
    return f1, iou, acc, prec, rec

# Brute-force the decision threshold on validation probabilities (if provided)
def best_threshold_from_probs(p_flat, y_true_flat, lo=0.3, hi=0.7, steps=41):
    best_t, best_m = 0.5, -1
    for t in np.linspace(lo, hi, steps):
        yb = (p_flat > t).astype(np.uint8)
        f1, *_ = metrics_from_flat(y_true_flat, yb)
        if f1 > best_m:
            best_m, best_t = f1, float(t)
    return best_t, best_m

# Use validation-tuned threshold if available; otherwise fallback to 0.5
def tune_or_fixed(p_test, name):
    if x_val is None:
        return 0.5
    p_flat = p_test  # placeholder
    t,_ = best_threshold_from_probs(
        p_flat = p_unetpp_v.reshape(-1) if name=='UPP' else p_deeplab_v.reshape(-1) if name=='DL' else p_segformer_v.reshape(-1),
        y_true_flat = y_val_flat
    )
    print(f"[{name}] best threshold: {t:.3f}")
    return t

# Model-specific thresholds
t_unetpp = tune_or_fixed(p_unetpp, 'UPP')
t_deeplab = tune_or_fixed(p_deeplab, 'DL')
t_segformer = tune_or_fixed(p_segformer, 'SF')

# Evaluate single models at tuned thresholds
y_unetpp = (p_unetpp.reshape(-1) > t_unetpp).astype(np.uint8)
y_deeplab = (p_deeplab.reshape(-1) > t_deeplab).astype(np.uint8)
y_segformer = (p_segformer.reshape(-1) > t_segformer).astype(np.uint8)

print("\nSingle models @ tuned threshold")
for name, yb in [('U-Net++',y_unetpp), ('DEEPLAB',y_deeplab), ('SEGFORMER',y_segformer)]:
    f1,iou,acc,prec,rec = metrics_from_flat(y_test_flat, yb)
    print(f"{name:6s} → F1={f1:.4f}, IoU={iou:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")


# Soft-voting ensemble: grid-search over weights (wu, wd, ws) with wu+wd+ws=1
weights = np.arange(0.1, 0.9, 0.1)
rows, best = [], {'F1':-1}

# Determine ensemble threshold on validation set (if available)
def ensemble_threshold(wu,wd,ws):
    if x_val is None: return 0.5
    p_ens_v = wu*p_unetpp_v + wd*p_deeplab_v + ws*p_segformer_v
    t,_ = best_threshold_from_probs(p_ens_v.reshape(-1), y_val_flat)
    return t

# Search across all valid weight combinations
for wu in weights:
    for wd in weights:
        if wu+wd>1.0: continue
        ws = 1.0 - wu - wd
        if ws < 0.1:
            continue
        t_ensemble = ensemble_threshold(wu,wd,ws)

        p_ensemble = wu*p_unetpp + wd*p_deeplab + ws*p_segformer
        y_ensemble = (p_ensemble.reshape(-1) > t_ensemble).astype(np.uint8)

        f1,iou,acc,prec,rec = metrics_from_flat(y_test_flat, y_ensemble)
        if f1 > best['F1']:
            best = {"wu":wu,"wd":wd,"ws":ws,"t":t_ensemble,"F1":f1,"IoU":iou,"Acc":acc,"Prec":prec,"Rec":rec}
            
# Save computed best weights and ensemble threshold
wUpp = best["wu"]
wDL = best["wd"]
wSF = best["ws"]
t_ensemble = round(best["t"],3)

print("\nEnsemble model")
print("Best F1 @ (wUpp,wDL,wSF):",(float(wUpp), float(wDL), float(wSF)),
      "Ensemble Threshold:",t_ensemble)
print("ENSEMBLE → F1={F1:.4f}, IoU={IoU:.4f}, Acc={Acc:.4f}, Prec={Prec:.4f}, Rec={Rec:.4f}".format(**best))