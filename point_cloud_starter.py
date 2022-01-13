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
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

## CHALLENGE 4 - CLUSTERING USING DBSCAN
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.2,
                                         min_points=10,
                                         print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.455,
#                                   front=[-0.4999, -0.1659, -0.8499],
#                                   lookat=[2.1813, 2.0619, 2.0999],
#                                   up=[0.1204, -0.9852, 0.1215])

# ## BONUS CHALLENGE - CLUSTERING USING KDTREE AND KNN INSTEAD
# pcd_tree =
#
## CHALLENGE 5 - BOUNDING BOXES IN 3D
start_t = time.time()
obbs = []
indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

MAX_POINTS = 300
MIN_POINTS = 30

for i in range(0, len(indexes)):
    nb_points = len(outlier_cloud.select_by_index(indexes[i]).points)
    if (nb_points>MIN_POINTS and nb_points<MAX_POINTS):
        sub_cloud = outlier_cloud.select_by_index(indexes[i])
        obb = sub_cloud.get_axis_aligned_bounding_box() # there is also oriented aligned bbox, instead of axis_aligned
        obb.color = (0, 0, 1)
        obbs.append(obb)
print("Number of bounding boxes: ", len(obbs))

list_of_visuals = []
list_of_visuals.append(outlier_cloud)
list_of_visuals.extend(obbs)
list_of_visuals.append(inlier_cloud)

end_t = time.time()

o3d.visualization.draw_geometries((list_of_visuals))

# ## CHALLENGE 6 - VISUALIZE THE FINAL RESULTS
# list_of_visuals =
#
# ## BONUS CHALLENGE 2 - MAKE IT WORK ON A VIDEO
