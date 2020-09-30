#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : September 27, 2020
##################################################### SOURCE START #####################################################

"""
Overview:

Usage:
    demo_ffhq_colorize.py [--dim_pca <int>] [--dataset <str>] [--num_test <int>] [--num_train <int>]
    demo_ffhq_colorize.py (-h | --help)
    demo_ffhq_colorize.py --version

Options:
    --dim_pca <int>      Dimension of principal components.     [default: 1024]
    --dataset <str>      Select dataset (celeba or ffhq).       [default: celeba]
    --num_test <int>     Number of test data.                   [default: 100]
    --num_train <int>    Number of training data.               [default: 30000]
    --seed <int>         Random seed.                           [default: 111]
    -h, --help           Show this message.
    --version            Show version.
"""


import os
import time

import docopt
import numpy as np
import cv2   as cv
import skimage.metrics

from linear_image2image_translation import PCA, LinearI2I


def load_dataset(dataset_name, data_type, num):

    print("  - Loading dataset: %s" % dataset_name)
    if dataset_name == "celeba":
        img_all = np.load("dataset/celeba/celeba_align_128x128.npy")
    elif dataset_name == "ffhq":
        img_all = np.load("dataset/ffhq/ffhq_thumbnails128x128.npy")
    else:
        raise RuntimeError("load_dataset: dataset should be 'celeba' or 'ffhq'.")

    print("  - Dataset type: %s..." % data_type)
    if data_type == "train": # Use from top
        img_c = img_all[:+num, :, :, :]
    elif data_type == "test": # Use from bottom
        img_c = img_all[-num:, :, :, :]
    else:
        raise RuntimeError("load_dataset: data_type should be 'train' or 'test'.")

    print("  - Generating grayscale images...")
    img_g = np.zeros((img_c.shape[0], 128, 128), dtype = np.uint8)
    for n in range(img_c.shape[0]):
        img_g[n, :, :] = cv.cvtColor(img_c[n, :, :, :], cv.COLOR_BGR2GRAY)

    print("  - Reshape images to vectors...")
    X_c = img_c.reshape((img_c.shape[0], 3 * 128**2)).astype(np.float32) / 255.0
    X_g = img_g.reshape((img_g.shape[0], 1 * 128**2)).astype(np.float32) / 255.0

    return (X_c, X_g)


def train(X_c, X_g, pca_dim):

    print("  - Principal component analysis for color domain...")
    pca_c = PCA(n_components = pca_dim)
    if os.path.exists("pca_color.npz"):
        pca_c.load("pca_color.npz")
        Z_c = pca_c.transform(X_c)
    else:
        Z_c = pca_c.fit_transform(X_c, adjust_sign = True)
        pca_c.save("pca_color.npz")

    print("  - Principal component analysis for grayscale domain...")
    pca_g = PCA(n_components = pca_dim)
    if os.path.exists("pca_gray.npz"):
        pca_g.load("pca_gray.npz")
        Z_g = pca_g.transform(X_g)
    else:
        Z_g = pca_g.fit_transform(X_g, adjust_sign = True)
        pca_g.save("pca_gray.npz")

    print("  - Calculate linear transformation...")
    lin = LinearI2I(supervise = True)
    if os.path.exists("lin_gray_to_color.npz"):
        lin.load("lin_gray_to_color.npz")
    else:
        lin.fit(Z_g, Z_c)
        lin.save("lin_gray_to_color.npz")

    return (lin, pca_c, pca_g)


def test(dirpath, X_c, X_g, lin, pca_color, pca_gray):

    def inference(x, lin, pca_c, pca_g):
        Q    = lin.matrix_
        W_c  = pca_c.components_
        m_c  = pca_c.mean_
        W_g  = pca_g.components_
        m_g  = pca_g.mean_
        img = W_c.T @ (Q @ (W_g @ (x - m_g))) + m_c
        return 255.0 * img.reshape((128, 128, 3))

    def postproc_colorize(img_gray_input, img_color_pred):

        if len(img_gray_input.shape) < 3:
            img_gray_input = img_gray_input[:, :, np.newaxis]

        img_gray_input = img_gray_input.astype(np.float32)

        img_gray_pred = cv.cvtColor(img_color_pred, cv.COLOR_BGR2GRAY)
        img_gray_pred = img_gray_pred[:, :, np.newaxis]
        img_gray_pred  = img_gray_pred.astype(np.float32)

        img_postproc = img_gray_input * img_color_pred / (img_gray_pred + 1)
        img_postproc = np.clip(img_postproc, 0, 255)
        img_postproc = img_postproc.astype(np.uint8)

        return img_postproc

    def create_concatenated_image(color_gt, gray, color_pred):
        concat = np.zeros((128, 384, 3), dtype = np.uint8)
        concat[:,   0:128, :] = color_gt.reshape((128, 128, 3))
        concat[:, 128:256, :] = gray.reshape((128, 128, 1))
        concat[:, 256:384, :] = color_pred.reshape((128, 128, 3))
        return concat

    os.makedirs(dirpath, exist_ok = True)

    images_output = np.zeros((X_g.shape[0], 128, 128, 3), dtype = np.uint8)

    print("  - Running inference...")
    time_start = time.time()
    for n in range(X_g.shape[0]):
        img_in  = 255.0 * X_g[n, :].reshape((128, 128))
        img_out = inference(X_g[n, :], lin, pca_color, pca_gray)
        img_out = postproc_colorize(img_in, img_out)
        images_output[n, :, :, :] = img_out

    print("  - Elasped time on CPU: %.3f [msec/image]" % ((time.time() - time_start) / X_g.shape[0] * 1000))

    ssim_measures, nmse_measures = list(), list()
    for n in range(X_g.shape[0]):
        img_gt  = (255.0 * X_c[n, :]).reshape((128, 128, 3)).astype(np.uint8)
        img_out = images_output[n, :, :, :]
        ssim_measures.append(skimage.metrics.structural_similarity(img_gt, img_out, multichannel = True))
        nmse_measures.append(skimage.metrics.normalized_root_mse(img_gt, img_out))
    print("  - SSIM measure: %.3f" % np.mean(ssim_measures))
    print("  - Normalized root MSE measure: %.3f" % np.mean(nmse_measures))
 
    print("  - Saving output images...")
    for n in range(X_g.shape[0]):
        img_gt  = (255.0 * X_c[n, :]).reshape((128, 128, 3)).astype(np.uint8)
        img_in  = (255.0 * X_g[n, :]).reshape((128, 128)).astype(np.uint8)
        img_out = images_output[n, :, :, :]
        image   = create_concatenated_image(img_gt, img_in, img_out)
        cv.imwrite(os.path.join(dirpath, "image-%04d.png" % n), image)
 

def main(args):

    print(":: Dump train/test parameters...")
    for key, val in args.items(): print("  -", key, "=", val)

    print(":: Expected computational cost for inference...")
    dim_in  = 128 * 128 * 3
    dim_out = 128 * 128
    madds   = (dim_in + dim_out) * (args["--dim_pca"] + 1)
    print("  - {:,} [MADDs/image]".format(madds))

    print(":: Loading training dataset...")
    X_color, X_gray = load_dataset(args["--dataset"], "train", args["--num_train"])

    print(":: Train linear transformation...")
    lin, pca_color, pca_gray = train(X_color, X_gray, args["--dim_pca"])

    print(":: Loading training dataset...")
    X_color, X_gray = load_dataset(args["--dataset"], "test", args["--num_test"])

    print(":: Run test...")
    test("output", X_color, X_gray, lin, pca_color, pca_gray)


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
