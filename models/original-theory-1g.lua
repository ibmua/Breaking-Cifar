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
	assert(opt and opt.depth)
	assert(opt and opt.num_classes)
	assert(opt and opt.widen_factor)



	local function Dropout()
		return nn.Dropout(opt and opt.dropout or 0,nil,true)
	end



	local depth = opt.depth

	local blocks = {}



	local function wide_basic(nInputPlane, nOutputPlane, stride	,	groups)
		local nBottleneckPlane = nOutputPlane

		local conv_params =
			{
				{nInputPlane			,	nBottleneckPlane	*1	,	3,3	,	stride,stride,	1,1	,	nInputPlane			},
				{nBottleneckPlane	*1	,	nBottleneckPlane		,	1,1	,	1,1,			0,0	,	1					},
				{nBottleneckPlane		,	nBottleneckPlane	*1	,	3,3	,	1,1,			1,1	,	nBottleneckPlane	},
				{nBottleneckPlane	*1	,	nOutputPlane			,	1,1	,	1,1,			0,0	,	1					},
			}

		local block = nn.Sequential()
		local convs = nn.Sequential()

		print(groups,'groups')

		for i,v in ipairs(conv_params) do

			if i == 1 then
				local module = nInputPlane == nOutputPlane and convs or block

				module
					:add(
						SBatchNorm( v[1] )
						)
					:add(
						ReLU(true)
						)
			else
				-- I don't know for sure, but 2x SBatchNorm may probably interfere.
				convs
					:add(
						SBatchNorm
							(
							v[1]
							)
						)

				if	conv_params[i-1][3]	==	1	then
					-- WE DON'T RELU AFTER 3x3 kernels.
					-- to stay close to the original model.
					convs:add(ReLU(true))
					end


				if opt.dropout > 0 then
					convs:add(Dropout())
					end


			end

			print					(
										v[1],v[2]
									,
										v[3],v[4]
									,
										v[5],v[6]
									,
										v[7],v[8]
									,
										v[9]
									)

			convs:add	(
						Convolution	(
										v[1],v[2]
									,
										v[3],v[4]
									,
										v[5],v[6]
									,
										v[7],v[8]
									,
										v[9]
									)
						)
		end

		local shortcut = nInputPlane == nOutputPlane and
			nn.Identity() or
							Convolution	(
											nInputPlane,nOutputPlane
										,	1,1
										,	stride,stride
										,	0,0
										,	1
										)

		return block
				:add(
					nn.ConcatTable()
						:add(convs)
						:add(shortcut)
					)
				:add(
					nn.CAddTable(true)
					)
	end



	-- Stacking Residual Units on the same stage
	local function layer(block, nInputPlane, nOutputPlane, count, stride 	, groups)
		local s = nn.Sequential()

		s:add	(
						block(nInputPlane, nOutputPlane, stride 			, groups)
				)
		for i=2,count do
			s:add	(
						block(nOutputPlane, nOutputPlane, 1 				, groups)
					)
		end
		return s
	end



	local model = nn.Sequential()
	do
		assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
		local n = (depth - 4) / 6

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
