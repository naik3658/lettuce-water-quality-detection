import os
import shutil

# Paths (update if your project folder is elsewhere)
source_base = "/Users/karthiknaik/Downloads/Lettuce disease datasets"
healthy_folder = os.path.join(source_base, "Healthy")  # Update if the folder name is different
unhealthy_folder = os.path.join(source_base, "unhealthy")

# Your project data folders (update if your project is elsewhere)
project_data = "/Users/karthiknaik/Documents/Gen AI/M.Tech/lettuce_contamination_project/data"
fresh_water_dest = os.path.join(project_data, "fresh_water")
contaminated_dest = os.path.join(project_data, "contaminated")

# Create destination folders if they don't exist
os.makedirs(fresh_water_dest, exist_ok=True)
os.makedirs(contaminated_dest, exist_ok=True)

# Move/copy healthy images
for fname in os.listdir(healthy_folder):
    src = os.path.join(healthy_folder, fname)
    dst = os.path.join(fresh_water_dest, fname)
    if os.path.isfile(src):
        shutil.copy2(src, dst)

# Move/copy all images from both subfolders in unhealthy
for subfolder in os.listdir(unhealthy_folder):
    subfolder_path = os.path.join(unhealthy_folder, subfolder)
    if os.path.isdir(subfolder_path):
        for fname in os.listdir(subfolder_path):
            src = os.path.join(subfolder_path, fname)
            dst = os.path.join(contaminated_dest, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

print("✅ Images have been organized into fresh_water and contaminated folders.") 