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

	local nStages = torch.Tensor{12*3 * 2}

	local model = nn.Sequential()
	model
		:add(
			Convolution	(
						3,	nStages[1],
						3,	3,
						1,	1,
						1,	1,
						3
						)
			) -- one conv at the beginning (spatial size: 32x32)

		:add(
			SBatchNorm
				(
				nStages[1]
				)
			)

		:add(
			ReLU(true)
			)

		:add(
			Avg	(
				6, 6,
				4, 4,
				1, 1
				)
			)	--	8x8 out

		:add(
			nn.View	(
					nStages[1] * 8*8
					)
					:setNumInputDims(3)
			)

		:add(
			nn.Linear
					(
						nStages[1] * 8*8
					,
						opt.num_classes *10
					)
			)

		:add(
			ReLU(true)
			)

		:add(
			nn.Linear
					(
						opt.num_classes *10
					,
						opt.num_classes
					)
			)


	utils.DisableBias(	model)
	utils.testModel(	model)
	utils.MSRinit(		model)
	utils.FCinit(		model)

	-- model:get(1).gradInput = nil

	return model
end

return createModel
