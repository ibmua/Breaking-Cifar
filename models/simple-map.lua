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

local nn = require 'nn'
local utils = paths.dofile'utils-old.lua'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)

	local model = nn.Sequential()

	model
		:add(
			nn.SpatialConvolutionMap
				(
					nn.tables.random(3, 24, 1)
				,
					3,3
				,
					1,1
				)
			)
		:add(
			nn.SpatialAveragePooling(30,30,1,1)
			)
		:add(
			nn.View(-1):setNumInputDims(3)
			)
		:add(
			nn.Linear(24,	opt.num_classes )
			)


	utils.DisableBias(model)
	utils.testModel(model)
	utils.MSRinit(model)
	utils.FCinit(model)

	-- model:get(1).gradInput = nil

	return model
end

return createModel
