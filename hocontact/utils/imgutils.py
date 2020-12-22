# Copyright (c) Lixin YANG, Jiasen Li. All Rights Reserved.

import hocontact.config as cfg
import cv2
from PIL import Image
import numpy as np
import hocontact.utils.func as func

def draw_hand_skeloten(clr, uv, bone_links, colors=cfg.JOINT_COLORS):
    for i in range(len(bone_links)):
        bone = bone_links[i]
        for j in bone:
            cv2.circle(clr, tuple(uv[j]), 4, colors[i], -1)
        for j, nj in zip(bone[:-1], bone[1:]):
            cv2.line(clr, tuple(uv[j]), tuple(uv[nj]), colors[i], 2)
    return clr


def batch_with_heatmap(
        inputs,
        heatmaps,
        num_rows=2,
        parts_to_show=None,
        n_in_batch=1,
):
    # inputs = func.to_numpy(inputs * 255)  # 0~1 -> 0 ~255
    heatmaps = func.to_numpy(heatmaps)
    batch_img = []
    for n in range(min(inputs.shape[0], n_in_batch)):
        inp = inputs[n]
        batch_img.append(
            sample_with_heatmap(
                inp,
                heatmaps[n],
                num_rows=num_rows,
                parts_to_show=parts_to_show
            )
        )
    resu = np.concatenate(batch_img)
    return resu


def sample_with_heatmap(img, heatmap, num_rows=2, parts_to_show=None):
    if parts_to_show is None:
        parts_to_show = np.arange(heatmap.shape[0])  # 21

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = cv2.resize(img, (size, size))

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = cv2.resize(heatmap[part_idx], (size, size))
        out_resized = out_resized.astype(float)
        out_img = inp_small.copy() * .4
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .6

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img


def color_heatmap(x):
    color = np.zeros((x.shape[0], x.shape[1], 3))
    color[:, :, 0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:, :, 1] = gauss(x, 1, .5, .3)
    color[:, :, 2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d
