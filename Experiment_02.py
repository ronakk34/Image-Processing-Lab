import numpy as np
import cv2
import matplotlib.pyplot as plt


#img path

img1 = cv2.imread("C://Image Processing Lab//NonDemented.jpg")
img2 = cv2.imread("C://Image Processing Lab//ModerateDemented.jpg")

# Resize second image to match first
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Convert to grayscale
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_resized,cv2.COLOR_BGR2GRAY)

# Compute SSIM (score + diff map)
score,diff = ssim(gray1, gray2, full = True)
print(f"Similarity Score:{score+100:.2f}%")

# Normalize diff for display
diff = (diff * 255).astype("uint8")

# Show images + diffrence map
plt.figure(figsize = (12,5))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
plt.title("Image 1 (Normal)")
plt.axis("Off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY))
plt.axis("Off")

plt.subplot(1,3,3)
plt.imshow(diff, cmap ="gray")
plt.title(f"Difference Map\nSimilarity: {score+100:.2f}%")

plt.tight_layout()
plt.show()