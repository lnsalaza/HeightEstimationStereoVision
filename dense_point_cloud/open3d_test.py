import open3d as o3d


pcd = o3d.io.read_point_cloud("../dense_point_cloud/point_clouds/new_keypoint_disparity/new_600_front_original.ply")
centroids = o3d.io.read_point_cloud("../dense_point_cloud/point_clouds/new_keypoint_disparity/new_600_front_centroids.ply")
print(pcd)
dataset = o3d.data.PLYPointCloud()

print(dataset)
office = o3d.io.read_point_cloud(dataset.path)

print(office)
o3d.visualization.draw_plotly([pcd, centroids], width=1920, height=1080, zoom=0.5)
