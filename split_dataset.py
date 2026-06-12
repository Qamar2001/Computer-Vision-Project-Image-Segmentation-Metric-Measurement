import os
import json
import random
import shutil

# =====================================================================
# CONFIGURATION: CHANGE THIS PATH TO WHERE YOUR ORIGINAL 116 IMAGES ARE
# =====================================================================
ORIGINAL_IMAGES_FOLDER = "C:\\Users\\qamar\\Downloads\\Project phone-20260612T171332Z-3-001\\Project phone" 

JSON_FILE_PATH = "./instances_Train.json"
OUTPUT_BASE = "./dataset"

def setup_splits():
    if not os.path.exists(JSON_FILE_PATH):
        print(f"Error: Cannot find {JSON_FILE_PATH} in the current directory.")
        return
    
    if not os.path.exists(ORIGINAL_IMAGES_FOLDER):
        print(f"Error: The path to your original images does not exist: {ORIGINAL_IMAGES_FOLDER}")
        print("Please open split_dataset.py and update line 9 with your actual folder path.")
        return

    with open(JSON_FILE_PATH, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # Shuffle dataset for clean variance
    random.seed(42)
    random.shuffle(images)

    total = len(images)
    train_count = int(total * 0.70)
    val_count = int(total * 0.20)
    
    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]

    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}

    print(f"Starting split... Total images found in JSON: {total}")

    for split_name, split_images in splits.items():
        # Setup clean directory paths
        split_img_dir = os.path.join(OUTPUT_BASE, split_name, "images")
        os.makedirs(split_img_dir, exist_ok=True)
        
        image_ids = {img['id'] for img in split_images}
        split_annotations = [ann for ann in annotations if ann['image_id'] in image_ids]
        
        split_coco = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "categories": coco_data.get("categories", []),
            "images": split_images,
            "annotations": split_annotations
        }
        
        # Save independent mini-COCO json files per split
        ann_dir = os.path.join(OUTPUT_BASE, split_name, "labels")
        os.makedirs(ann_dir, exist_ok=True)
        with open(os.path.join(ann_dir, "labels.json"), 'w') as f:
            json.dump(split_coco, f, indent=4)

        # Map and copy physical images over from your folder
        copied_count = 0
        for img in split_images:
            src_path = os.path.join(ORIGINAL_IMAGES_FOLDER, img['file_name'])
            dst_path = os.path.join(split_img_dir, img['file_name'])
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                copied_count += 1
            else:
                print(f"Warning: Could not find image file {img['file_name']} in your source folder.")
                
        print(f"  -> Created '{split_name}' split: Saved labels.json and successfully copied {copied_count} images.")
                
    print("\nSuccess! Your dataset folder is completely structured and ready.")

if __name__ == "__main__":
    setup_splits()