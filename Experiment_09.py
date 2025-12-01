import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#Load Image
img = cv2.imread("/content/modiji.jpg")
# Check if image was loaded correctly
if img is None:
    print("Error: Image not found. Please check the path.")
    # Using a placeholder image if the path is invalid for demonstration purposes
    # In a real scenario, you might want to exit or handle this more robustly.
    img = np.zeros((200, 300, 3), dtype=np.uint8) # Create a black image as a fallback
    cv2.putText(img, "Image not found", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#Reshape image for K-Means
pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

#Apply K-Means Clustering
K = 4 # number of segments
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
labels = kmeans.fit_predict(pixel_values)
#Replace pixel colors with cluster centers
centers = np.uint8(kmeans.cluster_centers_)
segmented_data = centers[labels]
segmented_image = segmented_data.reshape(img_rgb.shape)

#Show Original vs Segmented
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis("off")
plt.subplot(1,2,2)
plt.title("Segmented Image (K-Means)")
plt.imshow(segmented_image)
plt.axis("off")
plt.show()