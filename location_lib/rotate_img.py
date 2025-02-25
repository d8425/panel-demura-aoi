import cv2
import numpy as np


def rotate_around_point(image, angle_rad, point):
    (h, w) = image.shape[:2]

    # 旋转角度（度）
    angle_deg = np.degrees(angle_rad)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(point, angle_deg, 1.0)

    # 计算新的图像边界
    cos_angle = np.abs(M[0, 0])
    sin_angle = np.abs(M[0, 1])

    new_w = int((h * sin_angle) + (w * cos_angle))
    new_h = int((h * cos_angle) + (w * sin_angle))

    M[0, 2] += (new_w / 2) - point[0]
    M[1, 2] += (new_h / 2) - point[1]

    # 进行旋转
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h))

    return rotated_image
