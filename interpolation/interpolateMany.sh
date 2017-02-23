#!/usr/bin/env bash

delta=15
while [ $delta -lt 100 ];
do
  prev=5
  let "next=prev+delta"


  while [ $next -lt 105 ];
  do
    mkdir normResNet_${prev}k-${next}k
    cd normResNet_${prev}k-${next}k
    python ../interpolationEvaluation.py /home/ntopin/probingLoss/Resnet56Cifar.prototxt /home/ntopin/probingLoss/reduced_Resnet56Cifar.prototxt "/home/ntopin/fullResNet_iter_${prev}000.caffemodel" "/home/ntopin/fullResNet_iter_${next}000.caffemodel" 12 "test_loss_normalTrainResNet_${prev}k_to_${next}k" 80 |& tee train_log.txt 2>&1
    cd ..
    let "prev=prev+5"
    let "next=next+5"
  done

  let "delta=delta+5"
done

exit
