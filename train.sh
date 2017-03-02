#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/github-explore/solver.prototxt  \
    --gpu=all 2>&1 | tee examples/github-explore/results/fig1a-lr14


exit
