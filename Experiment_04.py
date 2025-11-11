import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define camera intrinsic parameters
fx, fy = 800, 800
cx, cy = 320, 240
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# 2. Define camera extrinsics
R = np.eye(3)
t = np.array([[0, 0, 0]]).T

# 3. Define 3D Points (cube)
cube_points = np.array([
    [-1, -1, 4], [1, -1, 4], [1, 1, 4], [-1, 1, 4],  # front face
    [-1, -1, 6], [1, -1, 6], [1, 1, 6], [-1, 1, 6]   # back face
])

# 4. Project 3D -> 2D
proj_points = (K @ (R @ cube_points.T + t)).T
proj_points = proj_points[:, :2] / proj_points[:, 2:3]

# 5. Visualization
fig = plt.figure(figsize=(12, 6))

# (a) 3D Scene View
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2], c='blue', s=50)
ax1.set_title("3D Scene (Camera Looking Towards Z-axis)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

edges = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]

for e in edges:
    ax1.plot(*zip(cube_points[e[0]], cube_points[e[1]]), color='black')

ax1.scatter(0, 0, 0, c='red', s=100, label='Camera')
ax1.legend()

# (b) 2D Image Projection
ax2 = fig.add_subplot(122)
ax2.scatter(proj_points[:, 0], proj_points[:, 1], c='red', s=50)

for e in edges:
    ax2.plot(proj_points[[e[0], e[1]], 0],
             proj_points[[e[0], e[1]], 1],
             color='black')

ax2.set_title("Projection on 2D Image Plane")
ax2.invert_yaxis()
ax2.set_xlabel("x (pixels)")
ax2.set_ylabel("y (pixels)")

plt.show()
