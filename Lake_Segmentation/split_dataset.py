import os
import shutil
import random

# Define dataset paths and desired split ratios
image_dir = f'{base_dir}/data_images'
mask_dir = f'{base_dir}/mask_images'
output_base = f'{base_dir}/dataset/'
splits = ['train', 'val', 'test']
split_ratio = {'train': 0.7, 'val': 0.1, 'test': 0.2}

# Create output folders for each split and type (images/masks)
for split in splits:
    os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, 'masks'), exist_ok=True)

# Get all image filenames and randomly shuffle them
all_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
random.shuffle(all_files)
total = len(all_files)
print(f"Total Images: {total}")
print(f"Total Masks: {len(sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))]))}")

# Split filenames according to the specified ratio
train_end = int(split_ratio['train'] * total)
val_end = train_end + int(split_ratio['val'] * total)
split_files = {
    'train': all_files[:train_end],
    'val': all_files[train_end:val_end],
    'test': all_files[val_end:]
}

# Copy image and corresponding mask files into respective folders
for split, files in split_files.items():
    for f in files:
        img_src = os.path.join(image_dir, f)
        mask_name = f.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_src = os.path.join(mask_dir, mask_name)
        if not os.path.exists(img_src):
            print(f"Missing image: {f}")
            continue
        if not os.path.exists(mask_src):
            print(f"Missing mask: {mask_name}")
            continue
        shutil.copy(img_src, os.path.join(output_base, split, 'images', f))
        shutil.copy(mask_src, os.path.join(output_base, split, 'masks', mask_name))

# Print summary of split counts
print("Dataset split completed, total:")
for k, v in split_files.items():
    print(f"{k}: {len(v)} images")