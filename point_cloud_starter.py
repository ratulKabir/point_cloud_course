## IMPORT LIBRARIES
import numpy as np
import time
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

## USE http://www.open3d.org/docs/release/tutorial/Basic/

## CHALLENGE 1 - OPEN A FILE OF YOUR CHOICE AND VISUALIZE THE POINT CLOUD
# The supported extension names are: pcd, ply, xyz, xyzrgb, xyzn, pts.
path2pc = 'test_files/sdc.pcd'
pcd = o3d.io.read_point_cloud(path2pc)
def vis_pcd(pcd, vis03d = True, vispptk = False):
    if vis03d:
        o3d.visualization.draw_geometries([pcd])

    ## IF YOU HAVE PPTK INSTALLED, VISUALIZE USING PPTK
    if vispptk:
        import pptk

        v = pptk.viewer(pcd.points)
        v.set(point_size=0.005)

## CHALLENGE 2 - VOXEL GRID DOWNSAMPLING
print(f"Points before downsampling: {len(pcd.points)} ")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
print(f"Points after downsampling: {len(downpcd.points)}")# DOWNSAMPLING
# vis_pcd(downpcd)

# ## CHALLENGE 3 - SEGMENTATION
start_time = time.time()
_, inliers = pcd.segment_plane(distance_threshold=0.3,
                                   ransac_n=3, num_iterations=500)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

outlier_cloud.paint_uniform_color([1, 0, 0])
inlier_cloud.paint_uniform_color([0, 1, 1])

end_time = time.time()
print("Time taken:", end_time-start_time)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# ## CHALLENGE 4 - CLUSTERING USING DBSCAN
# labels =
# max_label= labels.max()
# print(f"point cloud has {max_label + 1} clusters")
#
# ## BONUS CHALLENGE - CLUSTERING USING KDTREE AND KNN INSTEAD
# pcd_tree =
#
# ## CHALLENGE 5 - BOUNDING BOXES IN 3D
# bounding_boxes =
#
# ## CHALLENGE 6 - VISUALIZE THE FINAL RESULTS
# list_of_visuals =
#
# ## BONUS CHALLENGE 2 - MAKE IT WORK ON A VIDEO
