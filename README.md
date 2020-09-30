# Linear Image2Image Transformation

Unofficial implementation of "The Surprising Effectiveness of Linear Unsupervised Image-to-Image Translation".
Features of this repository are:

* usage of core classes provided by this repository is close to the `scikit-learn`,
* this repository focus on the efficiency of the inference.

Now this code supports only colorization of the CelebA and FFHQ dataset, however, I will add more features in the future.


## Requirements

- Python 3.6.9
- docopt 0.6.2
- Numpy 1.18.4
- PyTorch 1.5.0
- scikit-image 0.17.2

It is good idea to use docker environment in order to avoide polluting your environment.
The code in this repository is executable (and actually developed) under
[this docker image](https://hub.docker.com/r/tiskw/pytorch).


## Usage

The interface of the linear transformation is almost same as the `scikit-learn` library.

```python
>>> from linear_image2image_translation import LinearI2I
>>> X_gray  = ...                         # Grayscale images (shape = [n_data, n_features])
>>> X_color = ...                         # Color images     (shape = [n_data, n_features])
>>> lin = LinearI2I()                     # Create linear transformation class instance
>>> lin.fit(X_gray, X_color)              # Training
>>> X_color_pred = lin.transform(X_gray)  # Inference
array([[ -1.59893613e+00,  -2.18870965e-01,  -4.84763930e-02,
...
```

## Demo: colorization of CelebA dataset

### Summary

| Method                  | PCA dimension | Accuracy (SSIM/NRMSE) | Multi-add operations | Inference time (CPU) |
|:-----------------------:|:-------------:|:---------------------:|:--------------------:|:--------------------:|
| Cycle GAN [1]           |     -         | 0.914 / -             | 110  [G madds/image] |   -  [msec/image]    |
| Our method (Supervised) |   512         | 0.933 / 0.139         | 33.6 [M madds/image] | 5.01 [msec/image]    |
| Our method (Supervised) |   128         | 0.933 / 0.141         |  8.4 [M madds/image] | 1.43 [msec/image]    |


### Sample test images

* Left: Original color image (ground truth)
* Center: Grayscale image (input image)
* Right: Predicted color image (inference result)

<div align="center">
  <img src="resources/celeba_colorization/image-0018.jpg" width="384" height="128" alt="Sample imaeg of CelebA colorization" /><br />
  <img src="resources/celeba_colorization/image-0043.jpg" width="384" height="128" alt="Sample imaeg of CelebA colorization" /><br />
  <img src="resources/celeba_colorization/image-0088.jpg" width="384" height="128" alt="Sample imaeg of CelebA colorization" />
</div>

### Usage

At first, please download the zipped images from
[the official web page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

```console
$ cd dataset/celeba
$ python3 convert_celeba.py
```

The file `celeba_align_128x128.npy` will be generated.
You can erase other files/directories if not necessary.

Then, run the following command:

```console
$ python3 demo_colorize.py
```

The test results will be dumped under the `output` directory.
Please see `python3 demo_colorize.py --help` for more details.


### Pre-trained weights for CelebA colorization

* PCA dimension = 128: 
  [PCA color](https://www.dropbox.com/s/4ly2nq9f5ksucgj/pca_color.npz?dl=0),
  [PCA gray](https://www.dropbox.com/s/20gls2ln6p5r9w8/pca_gray.npz?dl=0),
  [Linear transform](https://www.dropbox.com/s/7kbiom6b9og4ofc/lin_gray_to_color.npz?dl=0)


## Licence

[MIT Licence](https://opensource.org/licenses/mit-license.php)


## Reference

[1] E. Richardson and Y. Weiss, "The Surprising Effectiveness of Linear Unsupervised Image-to-Image Translation", arXiv, 2020.
[PDF](https://arxiv.org/pdf/2007.12568.pdf)


## Author

Tetsuya Ishikawa ([EMail](mailto:tiskw111@gmail.com), [Website](https://tiskw.gitlab.io/home/))
