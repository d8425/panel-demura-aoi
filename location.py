import cv2
import matplotlib.pyplot as plt
import numpy as np
from uitls import plot_map
from datetime import datetime

from lib.location_lib import get_map, get_kernel

def _location(param, brt):
    # cfg
    root_path = param['Database']['data_path']
    loc_files = param['Database']['location_pattern']
    loc_files = loc_files.split(',')
    color_list = brt['color']
    flag = 0
    mapping = 0
    slope = 0
    for file_idx in range(len(color_list)):
        img = cv2.imread(root_path+'//'+loc_files[file_idx]+'.'+param['Database']['pattern_type'], cv2.IMREAD_GRAYSCALE)
        print('location: get image: '+loc_files[file_idx]+'.'+param['Database']['pattern_type'])
        # img rot
        if int(param['Database']['position'])-1 >= 1:
            img = np.rot90(img, k=int(param['Database']['position'])-1)


        # get mapping and slope, and use only one time fft
        if flag == 0:
            s_t = datetime.now()
            [mapping, slope] = get_map._get_mapping(img)  # panel should be in the center of image

            f_time_get_mapping = datetime.now() - s_t
            print('location: get mapping, time:'+str(f_time_get_mapping))

        # make kernel
        brt['kernel'+color_list[file_idx]] = get_kernel._get_kernel(mapping)
        print('location: get kernel')

        # create map
        pixel_type = param['Panel']['spatial_arrangement'][file_idx]
        # maps = get_map._get_map(mapping, slope, panel_res, pixel_type)
        maps = brt['map'+brt.color[file_idx]]

        # find a pixel, maps for info not renumber it
        s_t = datetime.now()
        map_real = get_map._find_all_pixel(maps, mapping, img, slope, brt['kernel'+color_list[file_idx]], pixel_type, param['Panel']['panel_res_' + color_list[file_idx]], param['Database']['location_threshold'])
        f_time_find_all_pixel = datetime.now() - s_t
        print('location: get real map'+color_list[file_idx]+', time:'+str(f_time_find_all_pixel))

        # complete the maps
        s_t = datetime.now()
        maps = get_map.complete_map2(maps, map_real, pixel_type, file_idx, mapping, img)
        f_time_complete_map = datetime.now() - s_t
        print('location: complete map' + color_list[file_idx]+', time:'+str(f_time_complete_map))

        # move the maps to the brightness pixel
        s_t = datetime.now()
        maps = get_map._move_to_brightness(maps, img)
        f_time_move_to_brightness = datetime.now() - s_t
        print('location: get brightness pixels, '+'time:'+str(f_time_move_to_brightness))

        brt['map' + color_list[file_idx]] = maps

        flag += 1
    return brt
