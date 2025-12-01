import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two overlapping images
img1 = cv2.imread("/content/melodiji.jpg")
img2 = cv2.imread("/content/modiji.jpg")

# Check if images are loaded successfully
if img1 is None or img2 is None:
    print("Error: Could not load one or both images. Please check paths.")
    # Exit or handle error appropriately
    exit()

# Ensure img2 is 3-channel for blending with panorama
# If img2 is grayscale (2D array or 3D with 1 channel), convert it to BGR
if len(img2.shape) == 2 or (len(img2.shape) == 3 and img2.shape[2] == 1):
    img2_for_blending = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
else:
    img2_for_blending = img2

# Convert to grayscale for feature detection (original img1 and img2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and descriptors (using ORB)
orb = cv2.ORB_create(2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Check if descriptors are found
if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
    print("Error: Not enough keypoints/descriptors found for one or both images.")
    exit()

# Match features using BFMatcher
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)

# Sort matches by quality
matches = sorted(matches, key=lambda x: x.distance)

# Use top 50 matches (or fewer if less than 50 are available)
good_matches = matches[:50]

# Check if enough good matches are found
if len(good_matches) < 4:
    print("Error: Not enough good matches to compute Homography.")
    exit()

# Extract matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute Homography
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print("Homography Matrix:\n", H)

# Check if Homography was found
if H is None:
    print("Error: Could not compute Homography. Panorama stitching might not be possible.")
    exit()

# Get dimensions of img2 for blending (h2, w2)
h2, w2 = img2_for_blending.shape[:2]

# Determine the dimensions of the final panoramic image based on the warped image 1 and image 2
# Get the corners of the first image
h1, w1 = img1.shape[:2]
corners1 = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)

# Transform the corners using the homography
transformed_corners = cv2.perspectiveTransform(corners1, H)

# Get the corners of the second image (at its original position)
corners2 = np.float32([[0, 0], [w2 - 1, 0], [w2 - 1, h2 - 1], [0, h2 - 1]]).reshape(-1, 1, 2)

# Combine all corners to find the bounding box for the panorama
all_corners = np.concatenate((transformed_corners, corners2), axis=0)

# Calculate integer min/max coordinates to cover all pixels
min_x_coord = np.min(all_corners[:, 0, 0])
max_x_coord = np.max(all_corners[:, 0, 0])
min_y_coord = np.min(all_corners[:, 0, 1])
max_y_coord = np.max(all_corners[:, 0, 1])

# Use floor and ceil to correctly define the bounding box for integer pixel coordinates
min_x_int = int(np.floor(min_x_coord))
max_x_int = int(np.ceil(max_x_coord))
min_y_int = int(np.floor(min_y_coord))
max_y_int = int(np.ceil(max_y_coord))

# Calculate panorama dimensions (width, height) to cover all pixels
panorama_width = max_x_int - min_x_int
panorama_height = max_y_int - min_y_int

# Create a translation matrix to shift the panorama so that min_x_int, min_y_int become (0,0)
translation_dist = [-min_x_int, -min_y_int]
H_translate = np.array([[1, 0, translation_dist[0]],
                        [0, 1, translation_dist[1]],
                        [0, 0, 1]], dtype=np.float32)

# Apply the translation to the homography
H_final = H_translate @ H

# Create a canvas large enough to hold the stitched panorama
# Note: cv2.warpPerspective dsize is (width, height)
panorama = cv2.warpPerspective(img1, H_final, (panorama_width, panorama_height))

# Place the second image onto the panorama (blending might be needed for seamlessness)
# To blend, we'll simply overlay img2 where it belongs.

# Calculate the region on the panorama where img2 will be placed
# Ensure the slice dimensions perfectly match img2_for_blending
panorama_target_y_start = translation_dist[1]
panorama_target_y_end = translation_dist[1] + h2
pano_target_x_start = translation_dist[0]
pano_target_x_end = translation_dist[0] + w2

# Ensure the slice indices do not exceed panorama bounds to prevent errors if calculation is slightly off
pano_slice_y_start = max(0, panorama_target_y_start)
pano_slice_y_end = min(panorama_height, panorama_target_y_end)
pano_slice_x_start = max(0, pano_target_x_start)
pano_slice_x_end = min(panorama_width, pano_target_x_end)

# Get the actual region from panorama (may be smaller if clipped)
pano_region = panorama[pano_slice_y_start:pano_slice_y_end, pano_slice_x_start:pano_slice_x_end]

# Get the corresponding region from img2_for_blending
img2_region = img2_for_blending[pano_slice_y_start - panorama_target_y_start:
                                 pano_slice_y_end - panorama_target_y_start,
                                 pano_slice_x_start - pano_target_x_start:
                                 pano_slice_x_end - pano_target_x_start]

# Only blend if both regions are not empty and have matching sizes
if pano_region.shape == img2_region.shape and pano_region.size > 0:
    blended_region = cv2.addWeighted(pano_region,
                                     0.5,
                                     img2_region,
                                     0.5,
                                     0)
    panorama[pano_slice_y_start:pano_slice_y_end, pano_slice_x_start:pano_slice_x_end] = blended_region
else:
    # If sizes don't match or regions are empty, try direct overlay if possible
    # This part can be refined for more sophisticated blending if needed
    # For now, let's just directly place if blending fails due to size issue after clipping
    panorama[pano_slice_y_start:pano_slice_y_end, pano_slice_x_start:pano_slice_x_end] = img2_region


# Display results
plt.figure(figsize=(15,8))

plt.subplot(1,3,1)
plt.title("Image 1")
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Image 2")
plt.imshow(cv2.cvtColor(img2_for_blending, cv2.COLOR_BGR2RGB)) # Use the 3-channel version for display
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Panorama Output")
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()