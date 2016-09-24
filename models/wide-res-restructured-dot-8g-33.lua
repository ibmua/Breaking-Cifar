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




local Linear, parent = torch.class('nn.NormalizedLinearNoBias', 'nn.Linear')
--[[
    This module creates a Linear layer, but with no bias component.
    In training mode, it constantly self-normalizes it's weights to
    be of unit norm.
    Authors: Mark Tygert, Soumith Chintala
]]--

function Linear:__init(inputSize, outputSize)
    parent.__init(self, inputSize, outputSize)
    self.bias:zero()
end

function Linear:updateOutput(input)
    if self.train then
        -- in training mode, renormalize the weights
        -- before every forward call
        self.weight:div(self.weight:norm())
        local scale = math.sqrt(self.weight:size(1))
        self.weight:mul(scale)
    end
    return parent.updateOutput(self, input)
end

function Linear:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    if input:dim() == 1 then
        self.gradWeight:addr(scale, gradOutput, input)
    elseif input:dim() == 2 then
        self.gradWeight:addmm(scale, gradOutput:t(), input)
    end
end





Gs	=	8 -- 40-depth -- 6x residual blocks | 2-width -> 3x residual blocks | 2xWidth x 2xGroups


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
				{nInputPlane					,	nBottleneckPlane			,	1,1	,	1,1,			0,0	,	1					},
				{nBottleneckPlane				,	nBottleneckPlane*2	*	Gs	,	3,3	,	stride,stride,	1,1	,	Gs		*	groups	},
				{nBottleneckPlane	*2	*	Gs	,	nBottleneckPlane*4	*	Gs	,	3,3	,	1,1,			1,1	,	Gs		*	groups	},
				{nBottleneckPlane	*4	*	Gs	,	nOutputPlane	*2	*	Gs	,	1,1	,	1,1,			0,0	,	Gs					},
				{nOutputPlane		*2	*	Gs	,	nOutputPlane				,	1,1	,	1,1,			0,0	,	1					},
			}

		local block = nn.Sequential()
		local convs = nn.Sequential()

		-- print(groups,'groups')

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

					:add(
						ReLU(true)
						-- nn.PReLU(v[2])
						)


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

			if	v[5]	>	1	or	v[6]	>	1	then
				convs
					:add(
						Convolution	(
										v[1],v[2]
									,
										v[3],v[4]
									,
										-- v[5],v[6]
										1,1
									,
										v[7],v[8]
									,
										v[9]
									)
						)
					:add(
						Max(2,2,2,2)
						)

			else
				convs
					:add(
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
			end

		-- convs
		-- 	:add(
		-- 		nn.VolumetricMaxPooling(Gs, 1, 1, nOutputPlane )
		-- 		)


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
					nn.CMulTable()
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
		assert((depth - 4) % (6 * Gs) == 0, 'depth should be 6n+4')
		local n = (depth - 4) / 6

		n	=	n / Gs	--	divided into groups

		local k = opt.widen_factor
		local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

		model:add(Convolution(3,nStages[1],	3,3,	1,1,	1,1			,1	)) -- one conv at the beginning (spatial size: 32x32)
		-- model:add(layer(wide_basic, nStages[1], nStages[2], n, 1		,nStages[1]	)) -- Stage 1 (spatial size: 32x32)
		-- model:add(layer(wide_basic, nStages[2], nStages[3], n, 2		,nStages[2]	)) -- Stage 2 (spatial size: 16x16)
		-- model:add(layer(wide_basic, nStages[3], nStages[4], n, 2		,nStages[3]	)) -- Stage 3 (spatial size: 8x8)

			:add(layer(wide_basic, nStages[1], nStages[2], n, 1		,2	)) -- Stage 1 (spatial size: 32x32)
			:add(layer(wide_basic, nStages[2], nStages[3], n, 2		,2	)) -- Stage 2 (spatial size: 16x16)
			:add(layer(wide_basic, nStages[3], nStages[4], n, 2		,1	)) -- Stage 3 (spatial size: 8x8)

			:add(SBatchNorm(nStages[4]))
			:add(ReLU(true))
			:add(Avg(8, 8, 1, 1))
			:add(nn.View(	nStages[4]):setNumInputDims(3))
		-- model:add(nn.Linear(nStages[4], opt.num_classes))

		model
			:add(
				-- nn.NormalizedLinearNoBias(nStages[4], opt.num_classes)
				nn.Linear(nStages[4], opt.num_classes)
				)
	end

	utils.DisableBias(	model)
	utils.testModel(	model)
	utils.MSRinit(		model)
	utils.FCinit(		model)

	-- model:get(1).gradInput = nil

	return model
end

return createModel
