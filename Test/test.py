
filepath = "C:\\Users\\Sepeh\\OneDrive\\AI\\SPCToolkit\\L001-3M.ply"
import core.point_cloud as pc
import open3d as o3d
import numpy as np

# Load the point cloud from a PLY file
pcd = o3d.io.read_point_cloud(filepath)

# Extract the points as a NumPy array
points = np.asarray(pcd.points)

# Extract the colors as a NumPy array (if available)
colors = np.asarray(pcd.colors) if pcd.has_colors() else None

# Create a PointCloud instance
point_cloud = pc.PointCloud(points, colors)
point_cloud.show()



