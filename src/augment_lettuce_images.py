import os
import cv2
import albumentations as A

# Paths
fresh_folder = "/Users/karthiknaik/Documents/Gen AI/M.Tech/lettuce_contamination_project/data/fresh_water"
contaminated_folder = "/Users/karthiknaik/Documents/Gen AI/M.Tech/lettuce_contamination_project/data/contaminated"
aug_fresh_folder = "/Users/karthiknaik/Documents/Gen AI/M.Tech/lettuce_contamination_project/data/aug_fresh_water"
aug_contaminated_folder = "/Users/karthiknaik/Documents/Gen AI/M.Tech/lettuce_contamination_project/data/aug_contaminated"

os.makedirs(aug_fresh_folder, exist_ok=True)
os.makedirs(aug_contaminated_folder, exist_ok=True)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.GaussNoise(p=0.2)
])

def augment_images(src_folder, dest_folder, n_aug=3):
    for fname in os.listdir(src_folder):
        img_path = os.path.join(src_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        base_name = os.path.splitext(fname)[0]
        # Save original
        cv2.imwrite(os.path.join(dest_folder, fname), img)
        # Save augmented images
        for i in range(n_aug):
            augmented = transform(image=img)['image']
            aug_name = f"{base_name}_aug{i+1}.jpg"
            cv2.imwrite(os.path.join(dest_folder, aug_name), augmented)

augment_images(fresh_folder, aug_fresh_folder)
augment_images(contaminated_folder, aug_contaminated_folder)

print("✅ Data augmentation complete! Augmented images saved.") 