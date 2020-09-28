#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : September 27, 2020
##################################################### SOURCE START #####################################################


import numpy as np
import cv2   as cv
import torch


class PCA:
# {{{

    def __init__(self, n_components, niter = 10):
        self.n_components = n_components
        self.niter = niter
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.device = self.devide_auto_detect()

    def devide_auto_detect(self):
        if torch.cuda.is_available(): return "cuda:%d" % torch.cuda.current_device()
        else                        : return "cpu"

    def fit(self, X_CPU, transform = False, adjust_sign = True):

        if not (hasattr(X_CPU, "shape") and len(X_CPU.shape) == 2):
            raise RuntimeError("PCA.fit: input variable should be 2-dimensional matrix.")

        X = torch.from_numpy(X_CPU).to(self.device)

        m = torch.mean(X, dim = 0)

        U, S, V = torch.pca_lowrank(X - m, self.n_components, center = False, niter = self.niter)

        Z = torch.matmul(X - m, V)

        if adjust_sign:
            s = torch.sign(torch.mean(Z**3, axis = 0))
            V = V * s[np.newaxis, :]
            Z = torch.matmul(X - m, V)

        self.mean_ = m.cpu().numpy()
        self.components_ = V.cpu().numpy().T
        self.explained_variance_ = np.square(S.cpu().numpy()) / (X_CPU.shape[0] - 1)

        if transform: return Z.cpu().numpy()
        else        : return self

    def fit_transform(self, X_CPU, *pargs, **kwargs):
        return self.fit(X_CPU, transform = True, *pargs, **kwargs)

    def inverse_transform(self, Z_CPU):
        Z = torch.from_numpy(Z_CPU).to(self.device)
        m = torch.from_numpy(self.mean_).to(self.device)
        W = torch.from_numpy(self.components_).to(self.device)
        X = torch.matmul(Z, W) + m
        return X.cpu().numpy()

    def load(self, filepath_npz):
        var = np.load(filepath_npz)
        self.mean_ = var["m"]
        self.components_ = var["W"]
        self.explained_variance_ = var["V"]

    def save(self, filepath_npz):
        np.savez(filepath_npz, m = self.mean_, W = self.components_, V = self.explained_variance_)

    def transform(self, X_CPU):
        X = torch.from_numpy(X_CPU).to(self.device)
        m = torch.from_numpy(self.mean_).to(self.device)
        W = torch.from_numpy(self.components_.T).to(self.device)
        Z = torch.matmul(X - m, W)
        return Z.cpu().numpy()

# }}}

class LinearI2I():
# {{{

    def __init__(self, supervise):
        self.supervise = supervise
        self.device = self.devide_auto_detect()
        self.matrix_ = None

    def devide_auto_detect(self):
        if torch.cuda.is_available(): return "cuda:%d" % torch.cuda.current_device()
        else                        : return "cpu"

    def fit(self, X_CPU, Y_CPU):
        X = torch.from_numpy(X_CPU).to(self.device)
        Y = torch.from_numpy(Y_CPU).to(self.device)
        if self.supervise: self.fit_supervise(X, Y)
        else             : self.fit_unsupervise(X, Y)
        return self

    def fit_supervise(self, X_GPU, Y_GPU):
        M = torch.matmul(torch.transpose(Y_GPU, 0, 1), X_GPU)
        U, S, V = torch.svd(M)
        Q = torch.matmul(U, torch.transpose(V, 0, 1))
        self.matrix_ = Q.cpu().numpy()

    def fit_unsupervise(self, X_GPU, Y_GPU):
        self.Q = torch.eye(X_GPU.shape[1])
        ### TODO(tetsuya.ishikawa): not implemented
        pass

    def load(self, filepath_npz):
        var = np.load(filepath_npz)
        self.matrix_ = var["Q"]

    def save(self, filepath_npz):
        np.savez(filepath_npz, Q = self.matrix_)

    def transform(self, X_CPU):
        X = torch.from_numpy(X_CPU).to(self.device)
        Q = torch.from_numpy(self.matrix_).to(self.device)
        Y = torch.matmul(Q, X)
        return Y.cpu().numpy()

# }}}

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
