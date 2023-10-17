'''
File: /testCarSequence.py
Created Date: Who knows when.
Author: Who knows who.
Comment:
-----
Last Modified: Wednesday September 13th 2023
Modified By: Ronit Hire <rhire@andrew.cmu.edu>
-----
Copyright (c) 2023 Carnegie Mellon University
-----
'''

import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from tqdm import tqdm
from functools import partial


def add_prev_patch(i, lk_res, patch):
    rect = lk_res[i]
    pt_topleft = rect[:2]
    pt_bottomright = rect[2:4]
    patch.set_width(pt_bottomright[0] - pt_topleft[0])
    patch.set_height(pt_bottomright[1] - pt_topleft[1])
    patch.set_xy((pt_topleft[0], pt_topleft[1]))
    patch.set_width(pt_bottomright[0] - pt_topleft[0])
    patch.set_height(pt_bottomright[1] - pt_topleft[1])
    patch.set_xy((pt_topleft[0], pt_topleft[1]))
    return (patch,)

def update_fig(i, lk_res, seq, patch, im, save_ids, save_prefix):
    rect = lk_res[i]
    pt_topleft = rect[:2]
    pt_bottomright = rect[2:4]
    patch.set_width(pt_bottomright[0] - pt_topleft[0])
    patch.set_height(pt_bottomright[1] - pt_topleft[1])
    patch.set_xy((pt_topleft[0], pt_topleft[1]))
    im.set_array(seq[:, :, i])
    patch.set_width(pt_bottomright[0] - pt_topleft[0])
    patch.set_height(pt_bottomright[1] - pt_topleft[1])
    patch.set_xy((pt_topleft[0], pt_topleft[1]))
    if i in save_ids:
        plt.savefig(
            save_prefix + str(i) + ".png",
            bbox_inches='tight',
            pad_inches=0,
        )
    return (im,patch,)


def animate_tracks(seq_path, rects_path, rects_path_prev=None, save_ids=[], save_prefix='q_'):
    ##### Code for animating, debugging, and saving images.
    fig, ax = plt.subplots(1)

    lk_res = np.load(rects_path)
    seq = np.load(seq_path)

    # add old path to animation
    if rects_path_prev is not None:
        lk_res_prev = np.load(rects_path_prev)
        rect2 = lk_res_prev[0]
        pt_topleft = rect2[:2]
        pt_bottomright = rect2[2:4]
        patch_prev = patches.Rectangle(
            (pt_topleft[0], pt_topleft[1]),
            pt_bottomright[0] - pt_topleft[0],
            pt_bottomright[1] - pt_topleft[1],
            linewidth=2,
            edgecolor='b',
            facecolor='none',
        )
        ax.add_patch(patch_prev)


    It1 = seq[:, :, 0]
    rect = lk_res[0]
    pt_topleft = rect[:2]
    pt_bottomright = rect[2:4]
    patch = patches.Rectangle(
        (pt_topleft[0], pt_topleft[1]),
        pt_bottomright[0] - pt_topleft[0],
        pt_bottomright[1] - pt_topleft[1],
        linewidth=3,
        edgecolor='r',
        facecolor='none',
    )
    ax.add_patch(patch)


    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = ax.imshow(It1, cmap='gray')

    if rects_path_prev is not None:
        ani_patch = animation.FuncAnimation(
            fig,
            partial(add_prev_patch, lk_res=lk_res_prev, patch=patch_prev),
            frames=range(lk_res_prev.shape[0]),
            interval=50,
            blit=False,
            repeat=False
        )

    ani = animation.FuncAnimation(
        fig,
        partial(update_fig, lk_res=lk_res, seq=seq, patch=patch, im=im, save_ids=save_ids, save_prefix=save_prefix),
        frames=range(lk_res.shape[0]),
        interval=50,
        blit=False,
        repeat=False
    )
    plt.show()

    ## Sample code for generating output image grid
    fig, axarr = plt.subplots(1, 5)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    for i in range(5):
        axarr[i].imshow(
            plt.imread(save_prefix + str(save_ids[i]) + ".png")
        )
        axarr[i].axis('off')
        axarr[i].axis('tight')
        axarr[i].axis('image')
    plt.savefig(save_prefix + "collage" + ".png", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 3:
        print('Example usage: python plotRects.py q1.3 car')
    else:
        with open("results_config.yml", "r") as f:
            params = yaml.safe_load(f)
        os.makedirs(params["save_dir"], exist_ok=True)
        test = params[args[1]][args[2]]
        seq_path = test['seq_path']
        rect_path = test['rect_path']
        rect_path_prev = test.get('rect_path_prev', None)
        save_ids = test['plot_frames']
        save_prefix = test['save_prefix']
        animate_tracks(seq_path, rect_path, rect_path_prev, save_ids, save_prefix)
