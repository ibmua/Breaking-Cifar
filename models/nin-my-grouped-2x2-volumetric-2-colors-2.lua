-- This is a modified version of NIN network in
-- https://github.com/szagoruyko/cifar.torch
-- Network-In-Network: http://arxiv.org/abs/1312.4400
-- Modifications:
--	* removed dropout
--	* added BatchNorm
--	* the last layer changed from avg-pooling to linear (works better)
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
		model:add(nn.ReLU(true))
	 --	model:add(nn.ELU())
		return model
	end

	-- Block(3,192,5,5,1,1,2,2				,1)
	Block(3,16,		1,1,1,1,0,0				,1)
	Block(16,64,	1,1,1,1,0,0				,1)

	Block(64,64,	2,2,1,1,1,1				,1)
	Block(64,128,	2,2,1,1,0,0				,1)
	Block(128,64,	2,2,1,1,1,1				,1)
	Block(64,256,	2,2,1,1,0,0				,1)
	-- Block(192,162,1,1		,1,1,0,0	,3)
	-- Block(162,96,1,1		,1,1,0,0	,1)
	Block(256,128,1,1		,1,1,0,0	,1)
	model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
	-- Block(96,96,	3,3,1,1,1,1				,3)
	-- Block(96,192,	3,3,1,1,1,1				,3)

	Block(128,128,		2,2,1,1,1,1				,1)
	Block(128,2048,		2,2,1,1,0,0				,32)
	Block(2048,1024,	2,2,1,1,1,1				,16)
	Block(1024,1024,	2,2,1,1,0,0				,4)


	-- Block(96,192,5,5		,1,1,2,2	,3)
	-- Block(192,192,1,1		,1,1,0,0	,3)
	Block(1024,1024,1,1		,1,1,0,0	,1)
	model:add(nn.VolumetricAveragePooling(2,2,2, 2,2,2))

	Block(512,1024,2,2		,1,1,1,1	,16)
	Block(1024,2048,2,2		,1,1,0,0	,8)

	-- Block(192,192,1,1		,1,1,0,0	,1)
	Block(2048,1024,1,1		,1,1,0,0	,1)

	-- Block(192,192,1,1)
	-- model:add(nn.SpatialAveragePooling(8,8,1,1))
	model:add(nn.VolumetricAveragePooling(4,8,8, 4,1,1))
	model:add(nn.View(-1):setNumInputDims(3))
	-- model:add(nn.Linear(256,opt and opt.num_classes or 10))
	model:add(nn.Linear(256,opt and opt.num_classes or 10))

	utils.FCinit(model)
	utils.testModel(model)
	utils.MSRinit(model)

	return model
end

return createModel


-- model=nin ./scripts/train_cifar.sh
