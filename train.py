import os
import shutil
import random
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths to your dataset
image_dir = './JPEGImages'
label_dir = './labels/all'

# Paths where images and labels will be moved
train_image_dir = './seg/images/train'
val_image_dir = './seg/images/val'
train_label_dir = './seg/labels/train'
val_label_dir = './seg/labels/val'

# Get list of all images with the correct extensions
images = [f for f in os.listdir(image_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]

# Ensure there are images in the directory
if not images:
    raise ValueError("No images found in the specified directory. Please check the path and file extensions.")

# Select a subset of the data (for example, 10% of the data)
subset_size = int(0.3 * len(images))
subset_images = random.sample(images, subset_size)

# Split into train and val sets
train_images, val_images = train_test_split(subset_images, test_size=0.2, random_state=42)

# Create directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

def copy_file(src, dst):
    shutil.copy(src, dst)

def process_images_and_labels(images, image_dir, label_dir, image_dest_dir, label_dest_dir):
    futures = []
    with ThreadPoolExecutor() as executor:
        for img in images:
            img_src = os.path.join(image_dir, img)
            img_dst = os.path.join(image_dest_dir, img)
            futures.append(executor.submit(copy_file, img_src, img_dst))

            base_name = os.path.splitext(img)[0]
            label_src = os.path.join(label_dir, base_name + '.txt')
            label_dst = os.path.join(label_dest_dir, base_name + '.txt')
            futures.append(executor.submit(copy_file, label_src, label_dst))

        for future in as_completed(futures):
            future.result()  # to re-raise any exceptions if they occurred

# Process train images and labels
process_images_and_labels(train_images, image_dir, label_dir, train_image_dir, train_label_dir)

# Process val images and labels
process_images_and_labels(val_images, image_dir, label_dir, val_image_dir, val_label_dir)

print("Random subset of images and labels have been successfully split and moved.")
