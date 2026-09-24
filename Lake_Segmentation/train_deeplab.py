# Train DeepLab model as two-class segmentation model
# This section implements a DeepLab model with metrics for evaluation and training callbacks
import os, random
import cv2, glob
import numpy as np
import tensorflow as tf
import keras_cv
from tqdm import tqdm
from keras_unet_collection import models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Set fixed random seeds to ensure reproducible training results
SEED = 7
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Define image and mask directories for training and validation
train_img_path = f'{base_dir}/dataset/train/images/'
train_mask_path = f'{base_dir}/dataset/train/masks/'
val_img_path   = f'{base_dir}/dataset/val/images/'
val_mask_path  = f'{base_dir}/dataset/val/masks/'

IMG_SIZE = (256, 256)

# Load and preprocess image-mask pairs
# Convert images to RGB, resize, normalize; binarize masks
def load_data(img_dir, mask_dir, img_size=(256, 256)):
    images, masks = [], []
    img_files = sorted(glob.glob(os.path.join(img_dir, '*')))
    for img_path in tqdm(img_files, desc=f'Loading {os.path.basename(os.path.normpath(img_dir))}'):
        fname = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, fname.replace('.jpg', '.png'))

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = (img / 255.0).astype(np.float32)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.int32)      # Convert grayscale mask to binary integer mask (0: background & 1: water) 

        images.append(img)
        masks.append(mask)

    # Convert binary mask to shape (H,W,2) using 2-channel one-Hot encoding
    return np.asarray(images, dtype=np.float32), tf.one_hot(np.array(masks), depth=2).numpy()
    
# Load datasets into memory
x_train, y_train = load_data(train_img_path, train_mask_path, IMG_SIZE)
x_val,   y_val   = load_data(val_img_path,   val_mask_path,   IMG_SIZE)
print('Train:', x_train.shape, y_train.shape, ' Val:', x_val.shape, y_val.shape)

# Define evaluation metrics: Intersection over Union (IoU) and Dice coefficient
def water_iou_metric(y_true, y_pred, smooth=1e-6):
    # Convert one-hot mask (H,W,2) -> class label (H,W) 0/1
    y_true_class = tf.argmax(y_true, axis=-1)

    # Predict class label from prediction probabilities
    y_pred_class = tf.argmax(y_pred, axis=-1)

    # Extract only water class
    y_true_water = tf.cast(y_true_class == 1, tf.float32)
    y_pred_water = tf.cast(y_pred_class == 1, tf.float32)

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_water * y_pred_water, axis=[1,2])
    union = (tf.reduce_sum(y_true_water, axis=[1,2]) + tf.reduce_sum(y_pred_water, axis=[1,2]) - intersection)
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

def water_dice_coef(y_true, y_pred, smooth=1e-6):
    # Extract ground truth water channel
    y_true_water = y_true[..., 1]

    # Extract predicted water probability
    y_pred_water = y_pred[..., 1]

    intersection = tf.reduce_sum(y_true_water * y_pred_water, axis=[1,2])
    denominator = (tf.reduce_sum(y_true_water, axis=[1,2]) + tf.reduce_sum(y_pred_water, axis=[1,2]))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)

def dice_loss(y_true,y_pred):
    return 1 - water_dice_coef(y_true,y_pred)

# Total loss = Weighted CE loss +  Dice loss
def total_loss(y_true, y_pred):
    # Categorical cross entropy loss
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)

    # Define class weights
    class_weights = tf.constant([0.3, 2.0])
    pixel_weights = tf.reduce_sum(y_true * class_weights, axis=-1)

    # Weighted categorical cross entropy loss
    weighted_cce = tf.reduce_mean(cce * pixel_weights)

    return weighted_cce + dice_loss(y_true, y_pred)

# Instantiate and compile DeepLabV3+ model
backbone = keras_cv.models.ResNet50V2Backbone.from_preset(
    "resnet50_v2_imagenet"
)
backbone.trainable = True

model = keras_cv.models.segmentation.DeepLabV3Plus(
    backbone=backbone,
    num_classes=2
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=total_loss,
    metrics=['accuracy', water_iou_metric, water_dice_coef]
)

# Define callbacks: save best model, reduce learning rate on plateau, early stopping
ckpt_path = f'{base_dir}/model/deeplab_model_best.keras'
callbacks = [
    ModelCheckpoint(ckpt_path, monitor='val_loss', mode='min',
                    save_best_only=True, save_weights_only=False, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', mode='min',
                      factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', mode='min',
                  patience=15, restore_best_weights=True, verbose=1)
]

# Train the model on training data with validation monitoring
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=16,
    epochs=100,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# Save the final model to disk
final_path = f'{base_dir}/model/deeplab_model_final.keras'
model.save(final_path)
print(f"Best checkpoint: {ckpt_path}\nFinal model: {final_path}")