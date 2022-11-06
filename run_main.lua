local chunk=dofile("main.lua")
chunk("-netType","resnet-dense-felix","-dataset", "mhad", "-numChannels",16 ,"-batchSize", 10, "-depth",5, "-full", 128 ,"-nEpochs", 80) 
--dofile('th main.lua') ({ netType="resnet-dense-felix",dataset="mhad",numChannels=16,batchSize=10, depth=30,full=128, nEpochs=80})
