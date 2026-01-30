import cv2
import matplotlib.pyplot as plt
from itertools import combinations
import os

# Create results directory
os.makedirs("results", exist_ok=True)

# --- Helper function to show and save matches ---
def show_match(img1, img2, matches_img, method, pair_name, save_name):
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{method} Feature Matching - {pair_name}")
    plt.axis("off")
    save_path = os.path.join("results", save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"   💾 Saved: {save_path}")
    plt.close()

# --- Detect and match features ---
def detect_and_match(img1_path, img2_path, method="SIFT"):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print(f"❌ Error: Could not load {img1_path} or {img2_path}")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Choose detector
    if method == "SIFT":
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method == "BRISK":
        detector = cv2.BRISK_create()
        norm_type = cv2.NORM_HAMMING
    elif method == "ORB":
        detector = cv2.ORB_create()
        norm_type = cv2.NORM_HAMMING
    else:
        print("❌ Invalid method name!")
        return

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top 20 matches
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

    pair_name = f"{img1_path} & {img2_path}"
    save_name = f"{method}_{img1_path.replace('.jpg', '')}_{img2_path.replace('.jpg', '')}.png"
    print(f"✅ {method}: Found {len(matches)} matches between {pair_name}")
    show_match(img1, img2, matched_img, method, pair_name, save_name)

# --- Run for all pairs of your images ---
images = ["1.jpg", "2.jpg", "3.jpg"]

for img1, img2 in combinations(images, 2):
    print(f"\n=== Comparing {img1} and {img2} ===")
    for method in ["SIFT", "BRISK", "ORB"]:
        detect_and_match(img1, img2, method)
