
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


local function createModel(opt)
	model = torch.load(
	'/home/sharpy/local-net/jupyter/lua/facebook/wide-residual-networks/logs/wide-res-restructured-pooled-actually-dot-g_248252290/model.t7'

	)
	return model
end

return createModel
