import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting

# Define 3D points for a simple house-like structure
pts = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Base square
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Top square (forming a prism)
    [0.5, 0.5, 1.5]                              # Roof peak
])

# Define a simple projection matrix (for orthographic projection to X-Y plane)
# This takes the X and Y coordinates directly for 2D projection
P = np.array([[1, 0, 0],
              [0, 1, 0]])

# Project 3D points to 2D
img_projected = pts @ P.T # This performs matrix multiplication: (N, 3) @ (3, 2) -> (N, 2)

# Visualization
fig = plt.figure(figsize=(12, 6))

# 2D Camera Image (Projection)
plt.subplot(1, 2, 1)
plt.scatter(img_projected[:, 0], img_projected[:, 1])
plt.title("Camera Image (2D Projection)")
# Label points by their index for clarity
for i, p in enumerate(img_projected):
    plt.text(p[0] + 0.05, p[1] + 0.05, str(i)) # Offset text slightly for readability
plt.xlabel("X (projected)")
plt.ylabel("Y (projected)")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal

# 3D House View
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='blue', s=50)
ax.set_title('Original House (3D)')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Drawing lines to connect the house points (edges) for better visualization
# Indices for points: 0-3 for base, 4-7 for top, 8 for roof peak
edges = [
    (0,1),(1,2),(2,3),(3,0), # Base edges
    (4,5),(5,6),(6,7),(7,4), # Top edges
    (0,4),(1,5),(2,6),(3,7), # Vertical edges
    (4,8),(5,8),(6,8),(7,8)  # Roof peak connections
]

for e in edges:
    ax.plot(*zip(pts[e[0]], pts[e[1]]), color='black')

plt.tight_layout()
plt.show()