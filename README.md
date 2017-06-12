# Implementation of [Camelyon'16](https://camelyon16.grand-challenge.org/) grand challenge

This repository contains the source code for deep learning based cancer detection system, developed to identify metastatic breast cancer from digital Whole Slide Images (WSI). Using dataset of the Camelyon’16 grand challenge, developed system obtained an area under the receiver operating curve (ROC) of 92.57%, which is superior than results obtained by [the winning method of Camelyon’16 grand challenge](https://camelyon16.grand-challenge.org/results/), developed by Harvard & MIT research laboratories. 

## Requirements
  - [python 3.5.x or above](https://www.python.org/downloads/)
  - [Tensorflow 0.12.1 or above (GPU version)](https://github.com/tensorflow/tensorflow)
  - [OpenSlide](http://openslide.org/download/)
  - [sk-learn](http://scikit-learn.org/stable/)
  - [sk-image](http://scikit-image.org/docs/dev/api/skimage.html)
  - [open-cv v3.0](http://docs.opencv.org/3.1.0/d5/de5/tutorial_py_setup_in_windows.html)
  - [numPy](https://github.com/numpy/numpy), [sciPy](https://github.com/scipy/scipy)

## Modules
  - [inception](camelyon16/inception)
    - contains implementation of inception-v3 deep network.
      - defining inception-v3 architecture
      - training inception-v3
      - evaluating inception-v3
      - implementation of TF-Slim
  - [ops](camelyon16/ops)
    - contains sub-modules for performing common operations 
      - reading WSIs
      - extracting patches from WSIs
      - file ops (copy, move, delete patches)
  - [preprocess](camelyon16/preprocess)
    - contains sub-modules for data pre-processing
      - find Region of Interest (ROI) for WSIs
      - extract training patches from WSIs
      - build TF-Records for training patches
  - [postprocess](camelyon16/postprocess)
    - contains sub-modules related to post-processing
      - extract patches for heatmaps
      - building TF-Records for heatmap patches
      - building heatmaps
      - extract features from heatmaps
      - feature classifiers (SVM, Random Forest)

