-- -- This is a modified version of NIN network in
-- -- https://github.com/szagoruyko/cifar.torch
-- -- Network-In-Network: http://arxiv.org/abs/1312.4400
-- -- Modifications:
-- --  * removed dropout
-- --  * added BatchNorm
-- --  * the last layer changed from avg-pooling to linear (works better)
-- require 'nn'
-- require 'cutorch'
-- require 'cunn'
-- require 'cudnn'
-- local utils = paths.dofile'utils.lua'
-- -- local utils = paths.dofile'utils-old.lua'
--
-- createModel = torch.load(
-- '/home/sharpy/local-net/jupyter/lua/facebook/wide-residual-networks/logs/nin-original_154497595/model_pow.t7'
-- )
--
-- return createModel




--  Wide Residual Network
--  This is an implementation of the wide residual networks described in:
--  "Wide Residual Networks", http://arxiv.org/abs/1605.07146
--  authored by Sergey Zagoruyko and Nikos Komodakis

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

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
    -- '/home/sharpy/local-net/jupyter/lua/facebook/wide-residual-networks/logs/original-theory-1g-no1x1-faster-second_3185614107/model-pow.t7'
	'/home/sharpy/local-net/jupyter/lua/facebook/wide-residual-networks/logs/nin-original_154497595/model_pow.t7'

	)
	return model
end

return createModel
