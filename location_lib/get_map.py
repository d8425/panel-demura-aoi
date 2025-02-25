import pdb

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from scipy import interpolate
from numba import jit
import pyfftw.interfaces.numpy_fft as fftw
from datetime import datetime
# from uitls import *

def _get_mapping(img):
    # if cp.cuda.runtime.getDeviceCount():
        # GPU
        # img_gpu = cp.asarray(img)
        # fft_img = cp.fft.fft2(img_gpu)
    # else:
        # CPU
    img = img[int(img.shape[0]/2)-500:int(img.shape[0]/2)+500, int(img.shape[1]/2)-500:int(img.shape[1]/2)+500]
    fft_img = fftw.fft2(img)

    fft_img_shift = abs(np.fft.fftshift(fft_img))

    # find max point, in a stable area
    row_len, col_len = fft_img_shift.shape
    roi = fft_img_shift[int(row_len/4):int(3*row_len/4),int(3*col_len/4):col_len] # 右边
    # roi = fft_img_shift[int(3*row_len/4):int(row_len),int(col_len/4):int(3*col_len/4)] # 下边
    [max_row, max_col] = np.where(roi == np.max(roi))
    # roi point, and they should be transform to original point
    max_row = max_row+row_len/4  # 右
    max_col = max_col+3*col_len/4  # 右
    # max_row = max_row+3*row_len/4 # 下
    # max_col = max_col+col_len/4 # 下

    # calculate the distance from the point to center point
    distance = ((int(max_row)-row_len/2)**2+(int(max_col)-col_len/2)**2)**0.5

    # panel Slope, (rad)
    slope = np.arctan(((int(max_row)-row_len/2)/(int(max_col)-col_len/2))) # 右
    # slope = np.arctan(((int(max_col) - col_len / 2)/(int(max_row) - row_len / 2))) # 下
    # frequency & mapping, cosA = col/dis(target distance)
    dis = col_len/np.cos(slope)
    fre_pix = dis/distance  # 横向上取点，该fre_pix即为mapping

    return fre_pix, slope

def _get_map(mapping,slope,res,pixel_type):

    # inital map
    map = np.zeros([*res,2]).astype(np.float32)
    delta_r = mapping * np.sin(slope)
    delta_c = mapping * np.cos(slope)

    # map_x: row
    map[0,...,0] = np.linspace(0,delta_r*res[1],num=res[1])
    for ridx in range(res[0]):
        map[ridx, ..., 0] = map[0,...,0]+ridx*delta_c

    # map_y: col
    if pixel_type == '1': # RGGB:R&B
        delta_c = 2*delta_c
    map[0,...,1] = np.linspace(0,delta_c*res[1],num=res[1])
    for ridx in range(res[0]):
        map[ridx, ..., 1] = map[0,...,1]+ridx*delta_r

    # 偶数行偏移，如右偏移则+，左偏则-，上偏则-，下偏则+
    map[1:-1:2, :, 0] = map[1:-1:2, :, 0] - delta_r / 2
    map[1:-1:2, :, 1] = map[1:-1:2, :, 1] - delta_c / 2
    return map


###################################
@jit(nopython=True, cache=True)
def process_col_loc_right_edge2(next_point, step_s, block_size_s, maps_new_point_x, maps_new_point_y, step, new_img):
    # 确保 next_point 是 NumPy 数组
    # next_point = np.array(next_point, dtype=np.int64)

    # 计算 ROI（感兴趣区域）
    next_point = np.array([next_point[0], int(next_point[1] - step_s / 2)], dtype=np.int64)
    x_min = next_point[0] - block_size_s
    x_max = next_point[0] + block_size_s + 1
    y_min = next_point[1] - block_size_s
    y_max = next_point[1] + block_size_s + 1

    # 确保 ROI 在图像边界内
    # if x_min < 0 or x_max > new_img.shape[0] or y_min < 0 or y_max > new_img.shape[1]:
    #     return next_point, maps_new_point_x, maps_new_point_y, np.array([]), np.array([])

    next_roi = new_img[x_min:x_max, y_min:y_max]

    # 找到非零元素的坐标
    [x, y] = next_roi.nonzero()
    if len(x) == 0:
        return list([0,0]), maps_new_point_x, maps_new_point_y, 0, 0
    x = x[0]
    y = y[0]
    # 更新 x 和 y
    x = next_point[0] + x - block_size_s
    y = next_point[1] + y - block_size_s

    # 找到非零元素的坐标
    # [x0, y0] = np.nonzero(next_roi != 0)


    # 更新 x 和 y
    # x = next_point[0] + x0 - block_size_s
    # y = next_point[1] + y0 - block_size_s

    # 创建numpy数组
    # next_point = np.empty(2, dtype=np.float64)
    next_point = [x, y]
    # next_point = np.array([x,y])
    # next_point = next_point.flatten()
    # if len(x0) > 0:  # 缩短step如果搜索到，则改变定位的位置
    maps_new_point_y += step / 2

    return next_point, maps_new_point_x, maps_new_point_y, x, y


@jit(nopython=True, cache=True)
def process_col_loc_left_edge2(next_point, step_s, block_size_s, maps_new_point_x, maps_new_point_y, step, new_img):
    # 确保 next_point 是 NumPy 数组
    # next_point = np.array(next_point, dtype=np.int64)

    # 计算新的位置
    next_point = np.array([next_point[0], int(next_point[1] + step_s / 2)], dtype=np.int64)

    # 计算 ROI（感兴趣区域）
    x_min = next_point[0] - block_size_s
    x_max = next_point[0] + block_size_s + 1
    y_min = next_point[1] - block_size_s
    y_max = next_point[1] + block_size_s + 1

    # 确保 ROI 在图像边界内
    # if x_min < 0 or x_max > new_img.shape[0] or y_min < 0 or y_max > new_img.shape[1]:
    #     return next_point, maps_new_point_x, maps_new_point_y, np.array([]), np.array([])

    next_roi = new_img[x_min:x_max, y_min:y_max]

    # 找到非零元素的坐标
    [x, y] = next_roi.nonzero()
    if len(x) == 0:
        return list([0,0]), maps_new_point_x, maps_new_point_y, 0, 0
    x = x[0]
    y = y[0]
    # 更新 x 和 y
    x = next_point[0] + x - block_size_s
    y = next_point[1] + y - block_size_s

    next_point = [x, y]

    # if len(x) > 0:  # 缩短step如果搜索到，则改变定位的位置
    maps_new_point_y -= step / 2
    return next_point, maps_new_point_x, maps_new_point_y, x, y


@jit(nopython=True, cache=True)
def process_col_loc_right(next_point, step_s, block_size_s, maps_new_point_x, maps_new_point_y, step, new_img):
    # inital
    next_roi = new_img[next_point[0] - block_size_s:next_point[0] + block_size_s + 1,
               next_point[1] - block_size_s:next_point[1] + block_size_s + 1]
    [x, y] = next_roi.nonzero()
    if len(x) == 0:
        return list([0,0]), maps_new_point_x, maps_new_point_y, 0, 0
    x = x[0]
    y = y[0]
    x, y = next_point[0] + x - block_size_s, next_point[1] + y - block_size_s
    maps_new_point_x = maps_new_point_x
    maps_new_point_y = maps_new_point_y + step
    # iter next_point
    next_point1 = [x, y]

    next_point2 = [int(next_point1[0]), int(next_point1[1] + step_s)]
    return next_point2, maps_new_point_x, maps_new_point_y, x, y


@jit(nopython=True, cache=True)
def process_col_loc_left(next_point, step_s, block_size_s, maps_new_point_x, maps_new_point_y, step, new_img):
    next_roi = new_img[next_point[0] - block_size_s:next_point[0] + block_size_s + 1,
               next_point[1] - block_size_s:next_point[1] + block_size_s + 1]
    [x, y] = next_roi.nonzero()
    if len(x) == 0:
        return list([0,0]), maps_new_point_x, maps_new_point_y, 0, 0
    x = x[0]
    y = y[0]
    x, y = next_point[0] + x - block_size_s, next_point[1] + y - block_size_s
    maps_new_point_x = maps_new_point_x
    maps_new_point_y = maps_new_point_y - step
    # iter next_point
    next_point1 = [x, y]
    # if len(x) != 0:
    # next_point2 = [int(next_point1[0][0]), int(next_point1[1][0] - step_s)]
    next_point2 = [int(next_point1[0]), int(next_point1[1] - step_s)]
    # else:
    #     next_point2 = next_point1
    return next_point2, maps_new_point_x, maps_new_point_y, x, y

#####################################
@jit(nopython=True, cache=True)
def _mean_filter(img, kernel_size):
    pad = kernel_size // 2
    h, w = img.shape
    output = np.zeros_like(img)

    # 遍历每个像素
    for i in range(h):
        for j in range(w):
            # 累积内核区域内的像素值
            pixel_sum = 0.0
            count = 0

            # 遍历内核
            for ki in range(-pad, pad + 1):
                for kj in range(-pad, pad + 1):
                    ni = i + ki
                    nj = j + kj

                    # 检查是否在边界内
                    if 0 <= ni < h and 0 <= nj < w:
                        pixel_sum += img[ni, nj]
                        count += 1

            # 计算均值
            output[i, j] = pixel_sum / count

    return output

# @jit(nopython=True, cache=True)
# def _simple_blur(img, kernel_size):
#     pad = kernel_size // 2
#     # 创建输出数组
#     output = np.zeros_like(img)
#     h, w = img.shape
#
#     # 手动进行边界填充
#     for i in range(h):
#         for j in range(w):
#             # 计算均值
#             sum_val = 0.0
#             count = 0
#
#             for k in range(-pad, pad + 1):
#                 for l in range(-pad, pad + 1):
#                     ni, nj = i + k, j + l
#                     if 0 <= ni < h and 0 <= nj < w:  # 确保在边界内
#                         sum_val += img[ni, nj]
#                         count += 1
#
#             output[i, j] = sum_val / count  # 计算均值
#
#     return output


@jit(nopython=True, cache=True)
def _deblur_img(img, loc_threshold):
    # img_blur = cv2.blur(img, [51, 51])
    img_blur = _mean_filter(img, 51)
    img_deblur = img - img_blur * 0.5
    img_deblur = np.where(img_deblur < loc_threshold, 0, 255)  # 先验位
    return img_deblur


def _find_a_pixel(img, loc_threshold):  # 原始版，可用
    img_deblur = _deblur_img(img, loc_threshold)
    _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(img_deblur), connectivity=4)

    # 创建一个新的图像，初始化为全黑
    new_img = np.zeros_like(img_deblur, dtype=np.uint8)

    # 遍历每个连通域 - 加入连通域 - 方法2：
    centroids = centroids.astype(int)
    # 保证坐标在图像边界内
    h, w = new_img.shape
    centroids[:, 0] = np.clip(centroids[:, 0], 0, w - 1)
    centroids[:, 1] = np.clip(centroids[:, 1], 0, h - 1)
    new_img[centroids[:, 1], centroids[:, 0]] = 255
    return new_img

def _find_all_pixel(maps, mapping, img, slope, kernel, pixel_type, res, loc_threshold):
    s_t = datetime.now()
    new_img = _find_a_pixel(img, int(loc_threshold))
    f_time_find_first_pixel = datetime.now() - s_t
    print('location: get real map, find first pixel' + ', time:'+str(f_time_find_first_pixel))
    # if len(new_img!=0) < 0.8 *

    # 寻找第一个像素，最上边中间某一像素 - method-2
    img_r, img_c = new_img.shape
    mid_roi = new_img[:, int(img_c/2-100):int(img_c/2+100)]
    mid_x, mid_y = np.where(mid_roi != 0)
    mid_x_y = mid_x + mid_y
    min_mid_x_y = np.where(mid_x_y == np.min(mid_x_y))
    if np.size(min_mid_x_y)>1:
        min_mid_x_y = min_mid_x_y[0][0]
    sx, sy = int(mid_x[min_mid_x_y]), int(mid_y[min_mid_x_y])
    sy = int(img_c/2-100)+sy

    # 创建一个矩阵，放置该定位信息
    maps_new = np.float32(np.zeros([maps.shape[0]*4,maps.shape[1]*4,2]))

    maps_new = _ergodic_all_pixels(maps_new, new_img, sx, sy, maps, mapping)

    # 构建map
    x_map, y_map = np.where(maps_new[...,0]!=0)
    x_map_min = np.min(x_map)
    x_map_max = np.max(x_map)
    y_map_min = np.min(y_map)
    y_map_max = np.max(y_map)
    maps_new1 = maps_new[x_map_min:x_map_max+1,y_map_min:y_map_max+1]
    return maps_new1

@jit(nopython=True, cache=True)
def _ergodic_all_pixels(maps_new, new_img, sx, sy, maps, mapping):
    # put the first point into the mid of map
    center_point_x = maps_new.shape[0] / 2
    center_point_y = maps_new.shape[1] / 2
    maps_new[int(center_point_x), int(center_point_y), 0] = sy
    maps_new[int(center_point_x), int(center_point_y), 1] = sx
    maps_new_point_x_right = maps_new_point_x_left = int(center_point_x)
    maps_new_point_y_right = maps_new_point_y_left = int(center_point_y)

    step = 4  # has to be even, >=2
    # if pixel_type == '0':
    #     step = step/2
    block_size = 3  # has to be odd>=3
    next_point_right = next_point_left = start_point = [sx, sy]  # next_point used for iter, start_point, used to record every row start point
    block_size_s = int(block_size / 2)
    step_s = round(step * mapping)
    flag_all = 0
    flag_col = 0  # row search flag
    dead_line = maps.shape[0]
    start_point_list_x = []
    start_point_list_y = []
    x_right = x_left = y_right = y_left = 1
    is_edge_right = 0  # edge search finsh flag
    is_edge_left = 0  # edge search finsh flag
    r, c = maps.shape[0:2]
    for _ in range(r*c):

        # from left to right search
        # if x_right != 0 or x_left != 0 and is_edge_right == 0:
        if x_right != 0 and is_edge_right == 0:
            next_point_right_record = next_point_right  # record the last point

            next_point_right, maps_new_point_x_right, maps_new_point_y_right, x_right, y_right = process_col_loc_right(
                next_point_right, step_s, block_size_s, maps_new_point_x_right, maps_new_point_y_right, step, new_img)
            # if isinstance(next_point_right, list):
            #     print('1')
            # else:
            #     print('0')
            if len(next_point_right) == 0 or next_point_right[0] == 0:  # edge search
                next_point_right, maps_new_point_x_right, maps_new_point_y_right, x_right, y_right = process_col_loc_right_edge2(
                    next_point_right_record, step_s, block_size_s, maps_new_point_x_right, maps_new_point_y_right, step,
                    new_img)
                # if isinstance(next_point_right, list):
                #     print('1')
                # else:
                #     print('0')
                is_edge_right = 1
        # else:
        #     print(x_right)


        # if len(x_left) != 0 and is_edge_left == 0:
        if x_left != 0 and is_edge_left == 0:
            next_point_left_record = next_point_left
            next_point_left, maps_new_point_x_left, maps_new_point_y_left, x_left, y_left = process_col_loc_left(
                next_point_left, step_s, block_size_s, maps_new_point_x_left, maps_new_point_y_left, step, new_img)
            if len(next_point_left) == 0 or next_point_left[0] == 0:
                next_point_left, maps_new_point_x_left, maps_new_point_y_left, x_left, y_left = process_col_loc_left_edge2(
                    next_point_left_record, step_s, block_size_s, maps_new_point_x_left, maps_new_point_y_left, step,
                    new_img)
                is_edge_left = 1

        flag_col += 1
        # if len(x_right)>1, select the first
        # x_right = np.array([x_right[0]])
        # y_right = np.array([y_right[0]])
        # x_left = np.array([x_left[0]])
        # y_left = np.array([y_left[0]])

        # if len(x_right) == 1 or len(x_right) == 2:
        if x_right != 0:
            # next pixel location
            # maps_new[int(maps_new_point_x_right), int(maps_new_point_y_right - 2 * step), 0] = y_right[0]  # 2*step
            # maps_new[int(maps_new_point_x_right), int(maps_new_point_y_right - 2 * step), 1] = x_right[0]
            maps_new[int(maps_new_point_x_right), int(maps_new_point_y_right - 2 * step), 0] = int(y_right)  # 2*step
            maps_new[int(maps_new_point_x_right), int(maps_new_point_y_right - 2 * step), 1] = int(x_right)

        # if len(x_left) == 1 or len(x_left) == 2:
        if x_left != 0:
            # maps_new[int(maps_new_point_x_left), int(maps_new_point_y_left), 0] = y_left[0]
            # maps_new[int(maps_new_point_x_left), int(maps_new_point_y_left), 1] = x_left[0]
            maps_new[int(maps_new_point_x_left), int(maps_new_point_y_left), 0] = int(y_left)
            maps_new[int(maps_new_point_x_left), int(maps_new_point_y_left), 1] = int(x_left)

        # if (x_right == 0 and len(x_left) == 0) or is_edge_right == is_edge_left == 1: # 向下方进行索引
        # if x_right == 0:
        #     print('1')
        #     if len(x_left) == 0:
        #         print('1')
        if is_edge_right == 1 and is_edge_left == 1:
            is_edge_right = is_edge_left = 0
            flag_col = 0
            start_point_list_x.append(start_point[0])
            start_point_list_y.append(start_point[1])
            start_point[0] = start_point[0] + step_s

            if new_img[start_point[0], start_point[1]] == 0:
                next_start_roi = new_img[start_point[0] - block_size_s:start_point[0] + block_size_s + 1,
                                 start_point[1] - block_size_s:start_point[1] + block_size_s + 1]
                [x_start, y_start] = np.where(next_start_roi != 0)
                x_start, y_start = start_point[0] + x_start - block_size_s, start_point[1] + y_start - block_size_s

                if len(x_start) == 0 and len(y_start) == 0:
                    start_point1 = start_point
                    start_point1[0] = start_point1[0] + step_s
                    next_start_roi1 = new_img[start_point1[0] - block_size_s:start_point1[0] + block_size_s + 1,
                                      start_point1[1] - block_size_s:start_point1[1] + block_size_s + 1]
                    [x_start1, y_start1] = np.where(next_start_roi1 != 0)
                    x_start1, y_start1 = start_point1[0] + x_start1 - block_size_s, start_point1[
                        1] + y_start1 - block_size_s
                    if len(x_start1) == len(y_start1) == 0:
                        break
                    else:
                        start_point = start_point
                else:
                    start_point = [int(x_start[0]), int(y_start[0])]

            next_point_right = next_point_left = start_point
            maps_new_point_x_right = maps_new_point_x_right + step
            maps_new_point_y_right = center_point_y
            maps_new_point_x_left = maps_new_point_x_left + step
            maps_new_point_y_left = center_point_y
            x_right = x_left = y_right = y_left = 1
            flag_all += 1
            if flag_all >= dead_line:
                print('loc finsh')
                break
    return maps_new


@jit(nopython=True, cache=True)
def _move_to_brightness(maps, img):
    # move the maps to the brightness point
    search_size = 3
    search_size_edge = int(search_size / 2)
    maps_r, maps_c = maps.shape[0:2]

    for r in range(maps_r):
        for c in range(maps_c):
            if maps[r, c, 0] != 0 or maps[r, c, 1] != 0:
                # Calculate the boundaries of the region of interest (ROI)
                x_start = max(round(maps[r, c, 0] - search_size_edge), 0)
                x_end = min(round(maps[r, c, 0] + search_size_edge + 1), img.shape[1])
                y_start = max(round(maps[r, c, 1] - search_size_edge), 0)
                y_end = min(round(maps[r, c, 1] + search_size_edge + 1), img.shape[0])

                # Extract ROI with boundary checks
                roi = img[y_start:y_end, x_start:x_end]

                if roi.size > 0:  # Ensure roi is not empty
                    roi_max = roi.max()

                    if roi.shape[0] > 0 and roi[search_size_edge - y_start, search_size_edge - x_start] != roi_max:
                        shift = np.where(roi == roi_max)
                        if shift[0].size > 0 and shift[1].size > 0:  # Ensure np.where found some indices
                            shift_x = int(shift[1][0]) - search_size_edge
                            shift_y = int(shift[0][0]) - search_size_edge
                            maps[r, c, 0] = int(maps[r, c, 0] + shift_x)
                            maps[r, c, 1] = int(maps[r, c, 1] + shift_y)

    return maps

@jit(nopython=True, cache=True)
def complete_map2(maps, map_real, pixel_type, file_idx, mapping, img):
    # 修正delta/rggb中菱形排序
    map_real_r, map_real_c = map_real[..., 0].shape
    maps_fill = np.zeros_like(maps)
    if pixel_type == '0':
        maps_fill[0:map_real_r, 0:map_real_c, :] = map_real
    elif pixel_type == '1' or pixel_type == '2':
        # maps_fill[0:map_real_r,0:map_real_c,:] = map_real[:,::2,:]
        temp_map = map_real[:,::2,:]
        maps_fill[0:temp_map.shape[0],0:temp_map.shape[1]] = temp_map

    map_real_r, map_real_c = maps_fill.shape[0:2]
    # find even-1 col
    col_list = [i for i in range(0, map_real_r, 4)]
    for r in col_list:
        row_x = maps_fill[r, :, 0]
        row_y = maps_fill[r, :, 1]
        maps_fill[r, ..., 0] = cal_one_dim_func(row_x, 0)  # 0: row ,1: col
        maps_fill[r, ..., 1] = cal_one_dim_func(row_y, 0)

    # complete the row
    for c in range(map_real_c):
        col_x = maps_fill[:, c, 0]
        col_y = maps_fill[:, c, 1]
        maps_fill[:, c, 0] = cal_one_dim_func(col_x, 1)
        maps_fill[:, c, 1] = cal_one_dim_func(col_y, 1)

    # maps shift
    if pixel_type == '1':
        maps_fill[1::2, :, 0] = maps_fill[1::2, :, 0] - mapping
    elif pixel_type == '2':
        maps_fill[1::2, :, 0] = maps_fill[1::2, :, 0] + mapping
    elif pixel_type == '0':
        maps_fill = maps_fill
    else:
        assert 0, 'wrong pixel_type (param: spatial_arrangement)'
    # maps_fill[maps_fill <= 0] = 0 # 无定位数据的像素点归纳为0
    maps_fill = maps_fill.astype(np.uint16)
    return maps_fill

@jit(nopython=True, cache=True)
def cal_one_dim_func(line, direction):
    # line[line==0] = []
    x = np.where(line != 0)
    y = list(x)
    # y = kx + b
    # left_area_y_val = np.mean(line[x[0][0:10:1]])
    # right_area_y_val = np.mean(line[x[0][-10::]])

    # min_y = sorted(set(line))[1]
    # max_y = line.max()
    # k = (max_y - min_y) / (np.max(x) - np.min(x))
    left_area_y_val = np.mean(line[x[0][0:10:1]])
    right_area_y_val = np.mean(line[x[0][-10::]])
    if len(x[0])<6:
        print('error')
    k = (right_area_y_val - left_area_y_val) / (x[0][-5]-x[0][5])
    # if k is small, k will be unstable, then k should be 0 will be stable
    if k<0.5 and k>-0.5:
        k = 0

    b = left_area_y_val - k*x[0][5]

    for idx in range(len(line)):
        # R corner area using func:
        if idx <= x[0][0]: # or idx >= x[0][-1]: # location without near points, using for left side pixels
            line[idx] = k*(idx+1) + b
        else: # high acc location, using for mid and right pixels
            # mid point interp for 3 pixels:
            if line[idx]==0:
                line[idx] = line[idx - 1] + k
                if line[idx] == 0:
                    line[idx] = line[idx - 2] + 2*k
                    if line[idx] == 0:
                        line[idx] = line[idx - 3] + 3 * k

    return line
