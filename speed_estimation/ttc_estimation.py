from scipy.linalg import norm
import pandas as pd
import numpy as np
import math

from modules.scaling_factor.scaling_factor_extraction import (
    GeometricModel,
    CameraPoint
)

import sys
sys.path.append('../Two-Dimensional-Time-To-Collision/src/')
import TwoDimTTC

def twodim_ttc_estimate(list_of_cars, geometry_model: GeometricModel, fps: int, frame_shape: tuple):
    '''
        list_of_cars: dict({'id': int,
                            'tracked_boxes': List[TrackingBox],
                            'bbox': TrackingBox,
                            'est_speed': float,
                            'direction': Direction})
    '''

    ttc_list_of_cars = []
    list_of_car_ids = []
    for t_car in list_of_cars:
        box1, box2 = t_car['tracked_boxes']
        # ignore parked cars
        meters_moved = geometry_model.get_distance_from_camera_points(
            CameraPoint(
                box1.frame_count,
                box1.center_x,
                box1.center_y,
            ),
            CameraPoint(
                box2.frame_count,
                box2.center_x,
                box2.center_y,
            ),
        )
        if meters_moved < 10:  # for 2 sec
            continue

        w_coord1 = geometry_model.get_world_point(CameraPoint(
            box1.frame_count,
            box1.center_x,
            box1.center_y,
        ))
        w_coord2 = geometry_model.get_world_point(CameraPoint(
            box2.frame_count,
            box2.center_x,
            box2.center_y,
        ))

        # prepare vectors for computing TTC
        [x, y, _] = w_coord1.coords()
        hx = w_coord2.coords()[0] - w_coord1.coords()[0]
        hy = w_coord2.coords()[1] - w_coord1.coords()[1]
        vx = hx / ((box2.frame_count - box1.frame_count) / fps)
        vy = hy / ((box2.frame_count - box1.frame_count) / fps)

        # Calculate car size
        h, w, _ = frame_shape

        x_size, y_size = (2.5, 4.5) if box1.class_id == 2 else (4, 12)
        # Vehicle moves in paralel to the x or y axis
        # if box1.center_x == box2.center_x:
        #     x_size = geometry_model.get_distance_from_camera_points(
        #         CameraPoint(
        #             box1.frame_count,
        #             box1.center_x -
        #             int(box1.width / 2) if (box1.center_x -
        #                                     int(box1.width / 2)) > 0 else 0,
        #             box1.center_y,
        #         ),
        #         CameraPoint(
        #             box1.frame_count,
        #             box1.center_x +
        #             int(box1.width / 2) if (box1.center_x +
        #                                     int(box1.width / 2)) < w else w - 1,
        #             box1.center_y,
        #         ),
        #     )
        #     y_size = geometry_model.get_distance_from_camera_points(
        #         CameraPoint(
        #             box1.frame_count,
        #             box1.center_x,
        #             box1.center_y -
        #             int(box1.height / 2) if (box1.center_x -
        #                                      int(box1.height / 2)) > 0 else 0,
        #         ),
        #         CameraPoint(
        #             box1.frame_count,
        #             box1.center_x,
        #             box1.center_y +
        #             int(box1.height / 2) if (box1.center_y +
        #                                      int(box1.height / 2)) < h else h - 1,
        #         ),
        #     )
        # elif box1.center_y == box2.center_y:
        #     x_size = geometry_model.get_distance_from_camera_points(
        #         CameraPoint(
        #             box1.frame_count,
        #             box1.center_x -
        #             int(box1.width / 2) if (box1.center_x -
        #                                     int(box1.width / 2)) > 0 else 0,
        #             box1.center_y,
        #         ),
        #         CameraPoint(
        #             box1.frame_count,
        #             box1.center_x +
        #             int(box1.width / 2) if (box1.center_x +
        #                                     int(box1.width / 2)) < w else w - 1,
        #             box1.center_y,
        #         ),
        #     )
        #     y_size = 1.5 * x_size # hard code as the object length is not shown on frame
        # else:
        #     # find the nearest point in the bounding box and propagate it along/ penpercular to velocity vector to get object dimensions
        #     # find angular coefficient between the velocity vector and horizontal line
        #     k = (box2.center_y - box1.center_y) / \
        #         (box2.center_x - box1.center_x)
        #     print('Angular Coefficient:', k)

        #     depth_map = geometry_model.depth_model.predict_depth(box1.frame_count)
        #     bb_memo_map = depth_map[box1.y_coord:(box1.y_coord + box1.height - 10), box1.x_coord:(box1.x_coord + box1.width)]
        #     y_pivot, x_pivot= np.where(bb_memo_map == bb_memo_map.max())
        #     y_pivot += box1.y_coord
        #     x_pivot += box1.x_coord

        #     # find projection of x size onto axises
        #     if k > 0:
        #         x_size = geometry_model.get_distance_from_camera_points(
        #             CameraPoint(box1.frame_count,
        #                         x_pivot,
        #                         y_pivot),
        #             CameraPoint(box1.frame_count,
        #                         box1.x_coord,
        #                         int(y_pivot - k * (x_pivot - box1.x_coord))),
        #         )
        #         y_size = geometry_model.get_distance_from_camera_points(
        #             CameraPoint(box1.frame_count,
        #                         x_pivot,
        #                         y_pivot),
        #             CameraPoint(box1.frame_count,
        #                         box1.x_coord + box1.width,
        #                         int(y_pivot - (box1.x_coord + box1.width - x_pivot) / k)),
        #         )
        #     else:
        #         x_size = geometry_model.get_distance_from_camera_points(
        #             CameraPoint(box1.frame_count,
        #                         x_pivot,
        #                         y_pivot),
        #             CameraPoint(box1.frame_count,
        #                         box1.x_coord + box1.width,
        #                         int(y_pivot - abs(k) * (box1.x_coord + box1.width - x_pivot))),
        #         )
        #         y_size = geometry_model.get_distance_from_camera_points(
        #             CameraPoint(box1.frame_count,
        #                         x_pivot,
        #                         y_pivot),
        #             CameraPoint(box1.frame_count,
        #                         box1.x_coord,
        #                         int(y_pivot - (x_pivot - box1.x_coord) / abs(k))),
        #         )

        (width, length) = (x_size, y_size) if x_size < y_size else (y_size, x_size)
        # (width, length) = (2.5, 4.5) if box1.class_id == 2 else (4, 12)

        if len(ttc_list_of_cars) == 0:
            ttc_list_of_cars = [[x, y, vx, vy, hx, hy, length, width]]
        else:
            ttc_list_of_cars.append([x, y, vx, vy, hx, hy, length, width])
        list_of_car_ids.append(t_car['id'])

    if len(ttc_list_of_cars) > 1:
        # prepare samples for computing TTC
        samples = pd.DataFrame(columns=['x_i', 'y_i', 'vx_i', 'vy_i', 'hx_i', 'hy_i', 'length_i', 'width_i',
                                        'x_j', 'y_j', 'vx_j', 'vy_j', 'hx_j', 'hy_j', 'length_j', 'width_j'])
        idx = 0
        car_ids = []
        for i in range(len(ttc_list_of_cars) - 1):
            for j in range(i + 1, len(ttc_list_of_cars)):
                samples.loc[idx] = ttc_list_of_cars[i] + ttc_list_of_cars[j]
                car_ids.append((list_of_car_ids[i], list_of_car_ids[j]))
                idx += 1

        # To return a dataframe with the input vehicle pair samples, where 2D-TTC are saved in a new column named 'TTC'
        samples = TwoDimTTC.TTC(samples, 'dataframe')

        # To return a numpy array of 2D-TTC values
        ttcs = TwoDimTTC.TTC(samples, 'values')
        ttc_with_car_ids = []
        for i in range(ttcs.shape[0]):
            if np.isfinite(ttcs[i]):
                ttc_with_car_ids.append({'ids': car_ids[i],
                                         'ttc': ttcs[i]})

        return ttc_with_car_ids
    else:
        return np.array([])
