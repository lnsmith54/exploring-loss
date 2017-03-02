# exploring-loss


This repository contains the files needed to replicate the Figures in:
Leslie N. Smith, Nicholay Topin, "Exploring loss function topology with cyclical learning rates" arXiv preprint arXiv:1702.04283.

It is necessary to modify Caffe with the Cyclical Learning Rate (CLR) policies in order to replicate the results shown in "Exploring loss function topology with cyclical learning rates" by Leslie N. Smith and Nicholay Topin.  Instructions for modifying Caffe are included in this repository and please see "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith at https://arxiv.org/abs/1506.01186 for additional information.

This repository contains:

architectures/ - the network architectures. 

solver.prototxt - the Caffe training solver file that was used for Figure 1a.

LRrange-solver.prototxt - the solver file for running an LR range test, which was used for Figure 1b.

clrsolver.prototxt - the solver file for running CLR with LR bounds 0.1 to 0.35 or 1.0 and stepsize=10000, which was used for Figure 2a and 2b.

train.sh - an example script for running Caffe.

interpolation/ - the scripts necessary for creating and evaluatin interpolations between networks (more details in separate readme).

Of course, you must change the folder paths to match your system (i.e., the location of your Cifar-10 data).

If you encounter bugs or missing data, you can contact me at leslie.smith@nrl.navy.mil.
