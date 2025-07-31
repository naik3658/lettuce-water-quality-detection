import cv2
import matplotlib.pyplot as plt
import os

# Use absolute paths for reliability
fresh_folder = "/Users/karthiknaik/Documents/Gen AI/M.Tech/lettuce_contamination_project/data/fresh_water"
contaminated_folder = "/Users/karthiknaik/Documents/Gen AI/M.Tech/lettuce_contamination_project/data/contaminated"

# Get a sample image from each folder
fresh_img_path = os.path.join(fresh_folder, os.listdir(fresh_folder)[0])
contaminated_img_path = os.path.join(contaminated_folder, os.listdir(contaminated_folder)[0])

# Load images
fresh_img = cv2.imread(fresh_img_path)
contaminated_img = cv2.imread(contaminated_img_path)

# Convert BGR to RGB for matplotlib
fresh_img = cv2.cvtColor(fresh_img, cv2.COLOR_BGR2RGB)
contaminated_img = cv2.cvtColor(contaminated_img, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(fresh_img)
plt.title("Fresh Water Lettuce")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(contaminated_img)
plt.title("Contaminated Lettuce")
plt.axis('off')

plt.show() 