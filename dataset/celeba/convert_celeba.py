#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : September 13, 2020
##################################################### SOURCE START #####################################################

"""
Overview:
    Convert CelebA dataset to .npy format.
    Please download the zipped images manually.

Usage:
    convert_celeba.py
    convert_celeba.py (-h | --help)
    convert_celeba.py --version

Options:
    -h, --help   Show this message.
    --version    Show version.
"""


import os

import docopt
import numpy as np
import cv2   as cv

DIRPATH = "img_align_celeba/"
IMAGE_SIZE = (128, 128)


def main(args):

    ### It is impossible to automatically download the zipped image from the official page,
    ### therefore, if the image diretory is not exists, just show a message and exit.
    if not os.path.exists(DIRPATH):
        print("Image directory not found.")
        print("Please download the zipped images from the official page of CelebA dataset, and unzip here.")
        print("Official page URL: <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>")
        exit()

    filenames = [f for f in os.listdir(DIRPATH) if f.endswith(".jpg")]
    tensor    = np.zeros((len(filenames), *IMAGE_SIZE, 3), dtype = np.uint8)

    for n, filename in enumerate(filenames):
        filepath = os.path.join(DIRPATH, filename)
        image    = cv.imread(filepath, cv.IMREAD_COLOR)
        tensor[n, :, :, :] = cv.resize(image, IMAGE_SIZE)

    np.save("celeba_align_128x128.npy", tensor)


if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    ### Run main procedure.
    main(args)

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
