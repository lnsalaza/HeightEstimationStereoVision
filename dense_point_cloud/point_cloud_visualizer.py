import open3d as o3d
# OLD
# pcd = o3d.io.read_point_cloud("./point_clouds/old_calibration.ply")
# NEW 
pcd= o3d.io.read_point_cloud("./point_clouds/new_calibration.ply")



# Visualizar

viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(pcd)
# opt = viewer.get_render_option()
# opt.point_size = 100
viewer.run()


viewer.destroy_window()