
import open3d as o3d
import numpy as np
import os

import time
import copy
def keypoints_to_spheres(keypoints, spheres, r=0.004):
    # spheres = o3d.geometry.TriangleMesh()
    spheres.clear()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.0, 0.0])
    return spheres


if __name__ == '__main__':
    root = 'data/'

    print(os.listdir(root))
    root_dir = root + os.listdir(root)[-1]

    print(root_dir)

    idx_list = os.listdir(root_dir)

    print(idx_list)


    vis = o3d.visualization.Visualizer()
    vis.create_window('PCD123', width=1280, height=720)

    pointcloud = o3d.geometry.PointCloud()
    pointcloud_selected = o3d.geometry.PointCloud()
    pointcloud_selected_spheres = o3d.geometry.TriangleMesh()
    pointcloud_selected_anchor = o3d.geometry.TriangleMesh()
    line_set = o3d.geometry.LineSet()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 10.0  # note that this is scaled with respect to pixels,

    show_list = []

    for idx in ['1', '2', '3', '4', '5', '6', '7', '8']:
    # for idx in ['1']:
        pointcloud_selected_anchor.clear()

        cur_root = root_dir+'/'+idx
        print(cur_root)

        pcd = o3d.io.read_point_cloud(cur_root+'/pcd.pcd')
        points = np.load(cur_root+'/line_set.npy')
        points_selected = np.load(cur_root+'/points_selected.npy')

        p0 = points[0].copy()
        points -= p0[None]
        points_selected -= p0[:, None]


        if idx == '1':
            rot_base = points[1:, :3].T
        rot_cur = points[1:, :3].T


        R = np.dot(rot_base, np.linalg.pinv(rot_cur))

        pcd = pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd = pcd.translate((-p0[0], -p0[1], -p0[2]))
        pcd = pcd.rotate(R, center=(0, 0, 0))
        pcd = pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        print(idx, points.shape, points_selected.shape, pcd)

        lines = [[0, 1], [0, 2], [0, 3]]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # 
        points_show = points.copy()
        points_show[1:] = np.dot(R, rot_cur).T
        # print(points_show)
        line_set.points = o3d.utility.Vector3dVector(points_show)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # line_set.rotate(R, center=(0, 0, 0))

        show_list.append(copy.deepcopy(line_set))



        pcd_xyz = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        mask = (pcd_xyz[:, 0]>-0.3)*(pcd_xyz[:, 1]>-0.3)*(pcd_xyz[:, 2]>-0.3)*\
                (pcd_xyz[:, 0]<1.0)*(pcd_xyz[:, 1]<1.0)*(pcd_xyz[:, 2]<1.0)
        pcd_xyz = pcd_xyz[mask]
        pcd_colors = pcd_colors[mask]
        print(pcd_xyz.shape, pcd_colors.shape)

        pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

        pointcloud.points = pcd.points
        pointcloud.colors = pcd.colors

        pointcloud_selected.points = o3d.utility.Vector3dVector(points_selected.transpose((1,0)))
        pointcloud_selected = pointcloud_selected.rotate(R, center=(0, 0, 0))
        pointcloud_selected.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        keypoints_to_spheres(pointcloud_selected, pointcloud_selected_spheres)

        show_list.append(copy.deepcopy(pointcloud))
        show_list.append(copy.deepcopy(pointcloud_selected_spheres))

    o3d.visualization.draw_geometries(show_list)

