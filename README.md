# RobotSense

![](https://github.com/Prevalenter/RobotSense/blob/main/data/img.jpg)

硬件设备：Intel RealSense 435d

系统平台：Winodws

开发语言：python

依赖模块：*opencv*-python 4.5.1, pyrealsense 2.50.0.3812, keyboard 0.13.5,Open3d 0.15.1



### 文件组织

get_pcd_vedio_with_key .py 获取多角度的点云数据，每次按下按键'a'时保存，数据保存在data文件夹下

concatenation.py 将多角度的点云数据进行拼接

calcu_axis.py 根据检测出的二维码坐标计算三维坐标轴

