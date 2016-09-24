#!/usr/bin/env bash

# export learningRate=0.11
# export learningRate=0.1
# export learningRate=0.01
export epoch_step=1
# export batchSize=128
# export batchSize=8
# export batchSize=128
# export batchSize=32
# export epoch_step=350
#"{10,20,30,40,50,60,70,80,90,100,110,120,130,}"
# export epoch_step="{60,120,160}"
export max_epoch=1000
# export learningRateDecay=0.00005
# export learningRateDecayRatio=0.96
# export learningRateDecayRatio=0.2

export nesterov=true
# export momentum=0
# export momentum=0.6
# export momentum=0.8
export momentum=0.9
export learningRateDecay=0.0001
export learningRateDecayRatio=0.994
export learningRate=0.06
# export learningRate=0.035
# export batchSize=128
export batchSize=64

export randomcrop_type=reflection
export	weightDecay=0.0005
# export	weightDecay=0.00001
# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand


# export learningRate=0.03
# export batchSize=16
# export learningRateDecayRatio=0.996
# export learningRateDecay=0.0001
# export momentum=0.96

export save=logs/${model}_${RANDOM}${RANDOM}
mkdir -p $save
th train.lua | tee $save/log.txt
