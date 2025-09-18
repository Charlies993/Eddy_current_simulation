This project is used for automation simulation of eddy current testing in NDT.
此项目用于无损检测领域中的涡流检测自动化仿真。

Already achieved feature: 
1. 3D model
   automated building planar spiral coils, rectangular coil, TMR sensor, litz wire; planar testpiece with random size defect;
3. advanced feature
   2.1 Supports the construction of array coils, where each individual coil element can be of a different type.
   2.2 Supports arbitrary path scanning (specimen in motion, coil stationary).
5. post processing
   Supports exporting induced voltage data in CSV format.

已实现功能：
1. 3D模型构建
    自动生成平面螺纹线圈，平面矩形线圈，TMR线圈，利兹线；平板试件及其任意尺寸缺陷；
2. 高级功能：
   2.1 支持构建阵列线圈，每个元素线圈种类可不相同。
   2.2 支持任意路径扫描。（平板试件运动，线圈静止）
3. 后处理
   支持感应线圈感应电压导出为csv格式。

Please see example.py to learn how to use.
请参阅example.py了解使用示例。

Before using this project, please install ANSYS EM software at version 2024R1, and install pyaedt python package at version 0.15
see https://aedt.docs.pyansys.com/version/stable/Getting_started/Installation.html
使用前请提前安装好Ansys EM 2024R1版本，及python包版本为0.15的pyaedt.
参阅 https://aedt.docs.pyansys.com/version/stable/Getting_started/Installation.html
