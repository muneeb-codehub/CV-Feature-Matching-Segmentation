import cv2
import numpy as np
import os

# Create masks directory if it doesn't exist
os.makedirs("masks", exist_ok=True)

# Get all images from images folder
image_files = [f for f in os.listdir("images") if f.endswith(('.jpg', '.jpeg', '.png'))]

print(f"Generating masks for {len(image_files)} images...")

for img_file in image_files:
    # Read image
    img_path = os.path.join("images", img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Could not read {img_file}")
        continue
    
    # Generate mask using edge detection and morphological operations
    # This creates a binary segmentation mask
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Otsu's thresholding for automatic threshold selection
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Save mask with same filename
    mask_path = os.path.join("masks", img_file)
    cv2.imwrite(mask_path, mask)
    print(f"✅ Generated mask: {mask_path}")

print(f"\n✅ Successfully generated {len(image_files)} masks!")
print(f"Masks saved in: masks/")
