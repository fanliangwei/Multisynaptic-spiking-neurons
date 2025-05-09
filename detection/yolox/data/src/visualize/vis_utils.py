"""
Functions to display events and boxes
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import print_function
import numpy as np
import cv2

LABELMAP = ["car", "pedestrian"]
LABELMAP_LARGE = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']

def make_binary_histo(events, img=None, width=304, height=240):
    """
    simple display function that shows negative events as blacks dots and positive as white one
    on a gray background
    args :
        - events structured numpy array
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int
        - height int
    return:
        - img numpy array, height x width x 3)
    """
    if img is None:
        img = 127 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 127
    if events.size:
        idx = (events['x'] < width) & (events['y'] < height)
        events = events[idx]
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        img[events['y'], events['x'], :] = 255 * events['p'][:, None]
    return img


def draw_bboxes(img, boxes, labelmap=LABELMAP):
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 2)
        text = '{}'.format(class_name)
        txt_color = (0, 0, 0) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 1, 1)[0]  ### 0.4
        txt_bk_color = (np.multiply(color, 0.7)).astype(np.uint8).tolist()
        cv2.rectangle(
                img,
                (pt1[0], pt1[1] + 1),
                (pt1[0] + txt_size[0] + 1, pt1[1] + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
        # cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        # cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, text, (pt1[0], pt1[1] + txt_size[1]), font, 0.4, txt_color, thickness=1)

