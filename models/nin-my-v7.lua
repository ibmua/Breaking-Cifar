-- This is a modified version of NIN network in
-- https://github.com/szagoruyko/cifar.torch
-- Network-In-Network: http://arxiv.org/abs/1312.4400
-- Modifications:
--  * removed dropout
--  * added BatchNorm
--  * the last layer changed from avg-pooling to linear (works better)
require 'nn'
local utils = paths.dofile'utils.lua'
-- local cudnn = require 'cudnn'
require 'cunn'
require 'cudnn'
---
torch.setdefaulttensortype('torch.CudaTensor')
---

local function createModel(opt)
   local model = nn.Sequential()

   local function Block(...)
     local arg = {...}
     model:add(cudnn.SpatialConvolution(...):noBias())
     model:add(nn.SpatialBatchNormalization(arg[2],1e-5))
    --  model:add(nn.ELU())
    --  model:add(nn.ReLU(true))
     model:add(nn.PReLU())
     return model
   end

   Block(3,48,3,3		,1,1,1,1	,3)
   Block(48,768,3,3		,1,1,1,1	,48)
   -- Block(768,192,1,1	,1,1,0,0	,192)
   model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
   -- Block(192,1536,3,3	,1,1,1,1	,192)
   Block(768,3072,3,3	,1,1,1,1	,768)
   Block(3072,6144,3,3	,1,1,1,1	,1536)

   Block(6144,1536,1,1	,1,1,0,0	,3)
   model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())

   -- Block(192,192,3,3,1,1,1,1)
   Block(1536,768,3,3		,1,1,1,1	,768)
   -- Block(768,6144,3,3		,1,1,1,1	,768)

   -- Block(192,192,1,1)
   -- Block(6144,1536,3,3		,1,1,1,1	,768)

   -- Block(1536,1536,1,1)
   model:add(nn.SpatialAveragePooling(8,8,1,1))
   model:add(nn.View(-1):setNumInputDims(3))
   model:add(nn.Linear(768,opt and opt.num_classes or 10))

   utils.FCinit(model)
   utils.testModel(model)
   utils.MSRinit(model)
   return model
end

return createModel


-- model=nin ./scripts/train_cifar.sh
