#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : September 13, 2020
##################################################### SOURCE START #####################################################


import glob

import numpy as np
import cv2   as cv


def main():

    filepaths = sorted(list(glob.glob("thumbnails128x128/**/*.png", recursive=True)))
    tensor    = np.zeros((len(filepaths), 128, 128, 3), dtype = np.uint8)

    for n, filepath in enumerate(filepaths):
        img = cv.imread(filepath, cv.IMREAD_COLOR)
        tensor[n, :, :, :] = img

    np.save("ffhq_128x128.npy", tensor)


if __name__ == "__main__":
    main()

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
