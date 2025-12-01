import cv2
import numpy as np
import requests
from io import BytesIO # Corrected from Bytes10
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to download image from URL
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10) # Added = and reasonable timeout
        response.raise_for_status() # Raise an exception for HTTP errors
        img_array = np.frombuffer(response.content, np.uint8) # Added =
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # Added =
        if img is None:
            print("Could not decode:", url)
        return img # Corrected typo 'ing' to 'img'
    except Exception as e:
        print("Error loading", url, ":", e)
        return None

# Function: Compute color histogram
def calculate_color_histogram(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image_rgb], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]) # Added = and corrected list syntax
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# ONLINE IMAGE URL LIST
image_urls = [
    "https://picsum.photos/300/300?random=1", # Corrected last part of URL
    "https://picsum.photos/300/300?random=2",
    "https://picsum.photos/300/300?random=3",
    "https://picsum.photos/300/300?random=4",
    "https://picsum.photos/300/300?random=5", # Corrected typo 'S' to '5'
    "https://picsum.photos/300/300?random=6",
    "https://picsum.photos/300/300?random=7",
    "https://picsum.photos/300/300?random=8",
    "https://picsum.photos/300/300?random=9",
    "https://picsum.photos/300/300?random=10"
]

# Removed extra malformed URL:
# "https://picsum.photos/300/300?random-11*"

feature_list = []
loaded_images = []

# Download and Extract features
for url in image_urls:
    print("Downloading:", url)
    img = load_image_from_url(url) # Added =
    if img is None:
        continue
    loaded_images.append(img)
    features = calculate_color_histogram(img) # Added =
    feature_list.append(features)

# Validation
if len(feature_list) < 2:
    raise ValueError("Not enough images loaded to perform clustering.")

data_matrix = np.array(feature_list)

# K-MEANS Clustering
K = 3 # Added =
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto') # Added = and n_init parameter
kmeans.fit(data_matrix)
labels = kmeans.labels_ # Added =
print("\n Clustering Complete!")

# Display images cluster-wise
for cluster_id in range(K):
    print("\n=== Cluster", cluster_id, "===") # Added closing '='
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id] # Added = and ==
    for i in cluster_indices[:3]: # show first 3 images
        plt.imshow(cv2.cvtColor(loaded_images[i], cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Cluster {cluster_id}") # Corrected f-string syntax
        plt.show()