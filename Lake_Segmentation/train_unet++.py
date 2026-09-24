# Train U-Net++ model as Binary segmentation model with Sigmoid
# This section implements a U-Net++ model with metrics for evaluation and training callbacks
import os, random
import cv2, glob
import numpy as np
import tensorflow as tf
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
        mask = (mask > 127).astype(np.float32)[..., None]   # Convert grayscale mask to binary float mask (0.0: background & 1.0: water) with shape (H, W, 1) 

        images.append(img)
        masks.append(mask)

    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.float32)

# Load datasets into memory
x_train, y_train = load_data(train_img_path, train_mask_path, IMG_SIZE)
x_val,   y_val   = load_data(val_img_path,   val_mask_path,   IMG_SIZE)
print('Train:', x_train.shape, y_train.shape, ' Val:', x_val.shape, y_val.shape)

# Define evaluation metrics: Intersection over Union (IoU) and Dice coefficient
def water_iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    union = tf.reduce_sum(y_true, axis=[1,2]) + tf.reduce_sum(y_pred, axis=[1,2]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

def water_dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    denominator = (tf.reduce_sum(y_true, axis=[1,2]) + tf.reduce_sum(y_pred, axis=[1,2]))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)

# Define loss fucntions: Dice loss, BCE loss and Total loss
def dice_loss(y_true,y_pred):
    return 1 - water_dice_coef(y_true,y_pred)

def total_loss(y_true,y_pred):
    # Combine binary cross-entropy and dice loss
    bce = tf.keras.losses.binary_crossentropy(y_true,y_pred)
    return bce + dice_loss(y_true,y_pred)

# Instantiate and compile U-Net++ model
model = models.unet_plus_2d(
    input_size=(IMG_SIZE[0], IMG_SIZE[1], 3),  
    filter_num=[64, 128, 256, 512, 1024],   # Convolution filters at each Encoder           
    n_labels=1,         # Binary segmentation                                    
    stack_num_down=2,   # Convolution blocks at each Encoder                           
    stack_num_up=2,     # Convolution blocks at each Decoder                           
    activation='ReLU',  # Internal activation function after each Convolution layer                             
    output_activation='Sigmoid',  # Output activation function                 
    freeze_backbone=False,                       
    weights=None,                      
    deep_supervision=False                         
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=total_loss,
    metrics=['accuracy', water_iou_metric, water_dice_coef]
)

# Define callbacks: save best model, reduce learning rate on plateau, early stopping
ckpt_path = f'{base_dir}/model/unet++_model_best.keras'
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
final_path = f'{base_dir}/model/unet++_model_final.keras'
model.save(final_path)
print(f"Best checkpoint: {ckpt_path}\nFinal model: {final_path}")