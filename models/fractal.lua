require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
-- local utils = paths.dofile'utils-old.lua'
local utils = paths.dofile'utils.lua'

local Convolution = cudnn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

Gs	=	4



function	fractalBlock
			(
				depth
			,
			)


local function createModel(opt)
	assert(opt and opt.depth)
	assert(opt and opt.num_classes)
	assert(opt and opt.widen_factor)



	local function Dropout()
		return nn.Dropout(opt and opt.dropout or 0,nil,true)
	end



	local depth = opt.depth

	local blocks = {}







	local model = nn.Sequential()
	do
		assert((depth - 4) % (6 * Gs) == 0, 'depth should be 6n+4')
		local n = (depth - 4) / 6

		n	=	n / Gs	--	divided into groups

		local k = opt.widen_factor
		local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

		model:add(Convolution(3,nStages[1],	3,3,	1,1,	1,1			,1	)) -- one conv at the beginning (spatial size: 32x32)
		-- model:add(layer(wide_basic, nStages[1], nStages[2], n, 1		,nStages[1]	)) -- Stage 1 (spatial size: 32x32)
		-- model:add(layer(wide_basic, nStages[2], nStages[3], n, 2		,nStages[2]	)) -- Stage 2 (spatial size: 16x16)
		-- model:add(layer(wide_basic, nStages[3], nStages[4], n, 2		,nStages[3]	)) -- Stage 3 (spatial size: 8x8)

		model:add(layer(wide_basic, nStages[1], nStages[2], n, 1		,1	)) -- Stage 1 (spatial size: 32x32)
		model:add(layer(wide_basic, nStages[2], nStages[3], n, 2		,1	)) -- Stage 2 (spatial size: 16x16)
		model:add(layer(wide_basic, nStages[3], nStages[4], n, 2		,1	)) -- Stage 3 (spatial size: 8x8)

		model:add(SBatchNorm(nStages[4]))
		model:add(ReLU(true))
		model:add(Avg(8, 8, 1, 1))
		model:add(nn.View(	nStages[4]):setNumInputDims(3))
		model:add(nn.Linear(nStages[4], opt.num_classes))
	end

	utils.DisableBias(	model)
	utils.testModel(	model)
	utils.MSRinit(		model)
	utils.FCinit(		model)

	-- model:get(1).gradInput = nil

	return model
end

return createModel
