# exploring-loss

UNDER CONSTRUCTION.  The files in this repository do not yet reproduce the figures in the paper.  It will be updated soon.

This repository contains the files needed to replicate the results in:
Leslie N. Smith, Nicholay Topin, "Exploring loss function topology with cyclical learning rates" arXiv preprint arXiv:1702.04283.

It is necessary to modify Caffe with the Cyclical Learning Rate (CLR) policies in order to replicate the results shown in "Exploring loss function topology with cyclical learning rates" by Leslie N. Smith and Nicholay Topin.  Instructions for modifying Caffe are included in this repository and please see "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith at https://arxiv.org/abs/1506.01186 for additional information.

This repository contains:

CifarResnet-56.prototxt - the training architecture for Resnet-56 on the Cifar10 data.

solver.prototxt - the Caffe training solver file

LRrange-solver.prototxt - the solver file for running an LR range test

clrsolver.prototxt - the solver file for running CLR with LR bounds 0.1 to 1.0 and stepsize=10000


If you encounter bugs or missing data, you can contact me at leslie.smith@nrl.navy.mil

