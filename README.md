# Linear Image2Image Transformation

Unofficial implementation of "The Surprising Effectiveness of Linear Unsupervised Image-to-Image Translation".
Now this code support only colorization of the FFHQ dataset, however, I will add more features in the future.


## Requirement

- Python 3.6.9
- docopt 0.6.2
- Numpy 1.18.4
- PyTorch 1.5.0

It is good idea to use docker environment in order to avoide polluting your environment.
The code in this repository is executable (and actually developed) under
[this docker image](https://hub.docker.com/r/tiskw/pytorch).


## Usage

### Download FFHQ dataset

```console
$ cd dataset/ffhq
$ python3 download_and_convert_ffhq.py
```

The file `ffhq_thumbnails128x128.npy` will be generated.
You can erase other files/directories if not necessary.


### Training and inference

```console
$ python3 demo_hhfq_colorize.py
```

The test results will be dumped under the `output` directory.
Please see `python3 demo_hhfq_colorize.py --help` for more details.


## Example: colorization of FFHQ images

| Method     | PCA dimension | Accuracy (SSIM) | Inference time    |
|:----------:|:-------------:|:---------------:|:-----------------:|
| Supervised | 1,024         | Comming soon    | 9.92 [msec/image] |
| Supervised |   512         | Comming soon    | 5.88 [msec/image] |
| Supervised |   256         | Comming soon    | 3.68 [msec/image] |


### Pre-trained weights

* PCA dimension = 1024
  - [PCA color](https://www.dropbox.com/s/qug0c750gvl9n07/pca_ffhq_color.npz?dl=0)
  - [PCA gray](https://www.dropbox.com/s/pzafwpuzia5srmh/pca_ffhq_gray.npz?dl=0)
  - [Linear transform](https://www.dropbox.com/s/j9l5wcle7szw1sf/lin_ffhq_gray_to_color.npz?dl=0) |


## Licence

[MIT Licence](https://opensource.org/licenses/mit-license.php)


## Reference

[1] E. Richardson and Y. Weiss, "The Surprising Effectiveness of Linear Unsupervised Image-to-Image Translation", arXiv, 2020.
[PDF](https://arxiv.org/pdf/2007.12568.pdf)


## Author

Tetsuya Ishikawa ([EMail](mailto:tiskw111@gmail.com), [Website](https://tiskw.gitlab.io/home/))
