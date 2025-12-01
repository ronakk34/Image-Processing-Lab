import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import urllib.request

# Create directory if it doesn't exist
os.makedirs("images_dataset", exist_ok=True)

# ONLINE IMAGE URL LIST - Corrected URLs and syntax
image_urls = [
    "https://picsum.photos/300/300?random=1",
    "https://picsum.photos/300/300?random=2",
    "https://picsum.photos/300/300?random=3",
    "https://picsum.photos/300/300?random=4",
    "https://picsum.photos/300/300?random=5",
    "https://picsum.photos/300/300?random=6",
    "https://picsum.photos/300/300?random=7",
    "https://picsum.photos/300/300?random=8",
    "https://picsum.photos/300/300?random=9",
    "https://picsum.photos/300/300?random=10"
]

# Download images
for i, url in enumerate(image_urls):
    save_path = f"images_dataset/img_{i+1}.jpg"
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded {url} to {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Download query image
query_url = "https://picsum.photos/300/300?random=11" # A new random image for query
query_path = "query.jpg"
try:
    urllib.request.urlretrieve(query_url, query_path)
    print(f"Downloaded query image from {query_url} to {query_path}")
except Exception as e:
    print(f"Error downloading query image: {e}")


def extract_features(image_path):
    # Read image in grayscale for ORB
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None

    orb = cv2.ORB_create(500) # Create ORB detector with 500 features
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None or len(descriptors) == 0:
        # Return a zero vector if no descriptors are found
        return np.zeros((1, 32)) # ORB descriptors are typically 32-dimensional
    return descriptors.mean(axis=0).reshape(1, -1) # Mean of descriptors as feature vector

# Get list of image files in the dataset directory
image_files = [f for f in os.listdir("images_dataset") if f.endswith(('.jpg', '.png'))]

features = []  # Stores feature vectors for all dataset images
paths = []     # Stores paths of all images (used later for displaying matches)

for file in image_files:
    img_path = os.path.join("images_dataset", file)
    f = extract_features(img_path)
    if f is not None:
        features.append(f)
        paths.append(img_path)

if not features:
    print("No features extracted from dataset images. Cannot proceed.")
    exit()

features = np.vstack(features)

# Extract feature for the query image
query_feature = extract_features(query_path)

if query_feature is None:
    print("Could not extract features from query image. Cannot proceed.")
    exit()

# Compute cosine similarity between query feature and dataset features
similarity_scores = cosine_similarity(query_feature, features)[0]

# Get top-k similar images
top_k = 5
# Sort in descending order and get top_k indices
top_idx = similarity_scores.argsort()[::-1][:top_k]

plt.figure(figsize=(15, 8))

# Display the query image
plt.subplot(2, top_k, 1)
query_img = cv2.imread(query_path)
if query_img is not None:
    plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    plt.title("Query Image")
    plt.axis("off")
else:
    plt.title("Query Image (Failed to load)")
    plt.axis("off")

# Display the top-k most similar images
print(f"\nTop {top_k} most similar images:")
for i, idx in enumerate(top_idx):
    img_to_display = cv2.imread(paths[idx])
    if img_to_display is not None:
        plt.subplot(2, top_k, top_k + i + 1) # Position in the second row
        plt.imshow(cv2.cvtColor(img_to_display, cv2.COLOR_BGR2RGB))
        plt.title(f"Match {i+1} (Score: {similarity_scores[idx]:.2f})")
        plt.axis("off")
    else:
        print(f"Warning: Could not load image {paths[idx]}")

plt.tight_layout()
plt.show()
