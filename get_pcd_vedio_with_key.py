import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
import datetime
from pathlib import Path
import time
import sys

import keyboard

import calcu_axis

first_frame_flage = True
add_frame_idx = False

def add_fame(e):
    global add_frame_idx
    add_frame_idx = True
    print('set add_frame_idx', add_frame_idx)

def exit(e):
    pipeline.stop()
    vis.destroy_window()
    sys.exit(0)

keyboard.on_press_key("a", add_fame)
keyboard.on_press_key("e", exit)


def keypoints_to_spheres(keypoints, spheres, r=0.004):
    # spheres = o3d.geometry.TriangleMesh()
    spheres.clear()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.0, 0.0])
    return spheres

def anchor_to_spheres(keypoints, spheres, r=0.006):
    # spheres = o3d.geometry.TriangleMesh()
    spheres.clear()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([0.0, 0.0, 1.0])
    return spheres


def uv2xyz(uv, depth_image, pinhole_camera_intrinsic, depth_scale = 1000.0):
    depth_selected = depth_image[uv[:, 1], uv[:, 0]]
    intrinsic_matrix = pinhole_camera_intrinsic.intrinsic_matrix
    fx, cx, fy, cy = intrinsic_matrix[0][0], intrinsic_matrix[0][2], intrinsic_matrix[1][1], intrinsic_matrix[1][2]
    
    D = depth_selected
    z = D / depth_scale
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    points_selected = np.array([x, y, z])
    return points_selected, depth_selected


if __name__ == '__main__':

    save_root = 'data/'+str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    print(save_root)
    exp_dir = Path(save_root)
    exp_dir.mkdir(exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    print(dir(config))

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True

            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
    print('found_rgb', found_rgb)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline_profile.get_stream(rs.stream.depth) 
    intr = profile.as_video_stream_profile().get_intrinsics() 
    intr_coeffs = np.array(intr.coeffs)

    print(intr)
    print(dir(intr))
    print(dir(profile.as_video_stream_profile()))

    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1] ])
    print(intr_matrix)

    print(device_product_line)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 15)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # Start streaming
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters_create()

    # cv::aruco::GridBoard::create(5, 7, 0.04, 0.01, dictionary);
    board = cv2.aruco.GridBoard_create(5, 7, 0.04, 0.01, arucoDict)
    pcds = []

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 100  # note that this is scaled with respect to pixels,

    vis = o3d.visualization.Visualizer()
    vis.create_window('PCD123', width=1280, height=720)

    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.7, 0.7, 0.7])

    geom_added = False
    pointcloud = o3d.geometry.PointCloud()
    pointcloud_selected = o3d.geometry.PointCloud()
    pointcloud_selected_spheres = o3d.geometry.TriangleMesh()
    pointcloud_selected_anchor = o3d.geometry.TriangleMesh()
    line_set = o3d.geometry.LineSet()

    save_idx = 0
    timestamp = time.time()

    try:
        idx = 0
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_image

            print(depth_image.shape, color_image.shape)

            depth_colormap = depth_image

            color_image = np.asarray(color_image[:, :, [2,1,0]], order='C')


            profile = frames.get_profile()
            intrinsics = profile.as_video_stream_profile().get_intrinsics()

            pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, 
                                        intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

            color_raw = o3d.geometry.Image(color_image)
            depth_raw = o3d.geometry.Image(depth_image)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                convert_rgb_to_intensity=False)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
            # print(dir(pinhole_camera_intrinsic))
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pointcloud.points = pcd.points
            pointcloud.colors = pcd.colors

            (corners, ids, rejected) = cv2.aruco.detectMarkers(color_image, arucoDict,
                                    parameters=arucoParams)
            rvec, tvec = None, None
            success, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, intr_matrix, intr_coeffs, rvec, tvec)

            
            pointcloud_selected_anchor.clear()
            pointcloud_selected_spheres.clear()
            line_set.clear()


            if len(corners) != 0:
                uv = []
                for i in corners:
                    uv.append(i[0, [0]])
                uv = np.concatenate(uv).astype(np.int32)

                points_selected, depth_selected = uv2xyz(uv, depth_image, pinhole_camera_intrinsic)

                vec, p0 = calcu_axis.solve(points_selected, ids[:, 0])
                pointcloud_selected.points = o3d.utility.Vector3dVector(p0[None])
                pointcloud_selected.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                anchor_to_spheres(pointcloud_selected, pointcloud_selected_anchor)

                points = [p0, p0+vec[0]*0.08, p0+vec[1]*0.08, p0-vec[2]*0.08]
                lines = [[0, 1], [0, 2], [0, 3]]
                colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                pointcloud_selected.points = o3d.utility.Vector3dVector(points_selected.transpose((1,0)))
                pointcloud_selected.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                keypoints_to_spheres(pointcloud_selected, pointcloud_selected_spheres)


            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1)

            if geom_added == False:
                vis.add_geometry(pointcloud)
                # if len(corners) != 0:
                vis.add_geometry(pointcloud_selected_spheres)
                # if success>0:
                vis.add_geometry(pointcloud_selected_anchor)
                vis.add_geometry(mesh_frame)
                vis.add_geometry(line_set)
                geom_added = True

            idx += 1
            vis.update_geometry(pointcloud)
            
            vis.update_geometry(mesh_frame)
            if len(corners) != 0:
                vis.update_geometry(pointcloud_selected_spheres)
                vis.update_geometry(line_set)
            if success>0:
                vis.update_geometry(pointcloud_selected_anchor)
            vis.poll_events()
            vis.update_renderer()

            if add_frame_idx is True:

                save_idx += 1
                print("save the idx: ", save_idx)

                cur_dir = exp_dir.joinpath('%d'%save_idx)
                cur_dir.mkdir(exist_ok=True)

                # print(vec.shape, p0.shape, pcd, points_selected.shape)

                o3d.io.write_point_cloud(str(cur_dir)+'/pcd.pcd', pcd)
                np.save(str(cur_dir)+'/line_set.npy', np.array(points))
                np.save(str(cur_dir)+'/points_selected.npy', points_selected)
                add_frame_idx = False

    finally:
        # Stop streaming
        pipeline.stop()
        vis.destroy_window()