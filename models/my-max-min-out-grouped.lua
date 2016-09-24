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
nninit = require 'nninit'


local function createModel(opt)
	local model = nn.Sequential()

	local function Block(...)
		local arg = {...}
		model:add	(
			cudnn.SpatialConvolution(...)
					:noBias()
					:init(
							'weight'
						,	nninit.orthogonal
						,	{
							dist = 'uniform',
							gain =
								{
								'lrelu'			,
								leakiness = 0.1
								}
							}
						)
					)
		model:add(nn.SpatialBatchNormalization(arg[2],1e-5))
		-- model:add(nn.ReLU(true))
	 	model:add(nn.ELU())
		return model
	end

	Block(3,48,5,5,1,1,2,2				,3)

	Block(48,192,1,1		,1,1,0,0	,48)

	Block(192,96,1,1		,1,1,0,0	,96)
	Block(96,48	,1,1		,1,1,0,0	,48)

	Block(48,48	,1,1		,1,1,0,0	,1)

	model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())

	Block(48,192,5,5		,1,1,2,2	,48)

	Block(192,768,1,1		,1,1,0,0	,192)
	Block(768,384,1,1		,1,1,0,0	,384)
	Block(384,192,1,1		,1,1,0,0	,192)

	Block(192,192,1,1		,1,1,0,0	,1)

	model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())

	Block(192,768,3,3		,1,1,1,1	,192)

	Block(768,3072	,1,1	,1,1,0,0	,768)
	Block(3072,1536	,1,1	,1,1,0,0	,1536)
	Block(1536,768	,1,1	,1,1,0,0	,768)

	Block(768,768	,1,1	,1,1,0,0	,1)

	-- Block(192,192,1,1)
	model:add(nn.SpatialAveragePooling(8,8,1,1))
	model:add(nn.View(-1):setNumInputDims(3))
	model:add(nn.Linear(768,opt and opt.num_classes or 10))

	utils.FCinit(model)
	utils.testModel(model)
	-- utils.MSRinit(model)

	return model
end

return createModel


-- model=nin ./scripts/train_cifar.sh
