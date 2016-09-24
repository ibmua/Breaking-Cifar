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

local Convolution = cudnn.SpatialConvolution


width 		= 1
depth 		= 3
sequences 	= 1


function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end


function	Convolutions(convs)

	-- print(convs)

	m	=	nn.Sequential()
	for	i	,	c	in	ipairs(convs)	do
		-- print(c)
		m	:add(
				Convolution	(
								c[1],c[2]
							,
								c[3],c[4]
							,
								c[5],c[6]
							,
								c[7],c[8]
							,
								c[9]
							)
				)
			:add(
				nn.SpatialBatchNormalization(c[2],1e-5)
				)
			:add(
				nn.ReLU(true)
				)

		end

	return m
	end


function build	(
					ins_input
				,	outs
				)

	local ins = ins_input
	local module = nn.Sequential()


	for	seq	=	1,	sequences	do

		local	conv_params	=
				-- {
				-- 	{nInputPlane					,	nBottleneckPlane	*	Gs/4	,	3,3	,	1,1,	1,1	,	1	},
				-- 	{nBottleneckPlane	*	Gs/4	,	nBottleneckPlane	*	Gs		,	3,3	,	1,1,	1,1	,	Gs	},
				-- 	{nBottleneckPlane	*	Gs		,	nOutputPlane					,	1,1	,	1,1,	0,0	,	1	},
				-- }
				-- {	0,0	,	1,1	,	1,1,	0,0	,	1	},
				{
					{
						{	ins			,	ins	* width	,	3,3	,	1,1,	1,1	,	1		}
					,	{	ins	* width	,	ins			,	3,3	,	1,1,	1,1	,	width	}
					}
				}
		-- print(conv_params)

		for i = 1 , depth do
			-- table.insert
			-- 	(
			-- 		conv_params
			-- 	,	table.copy( conv_params[1] )
			-- 	,	1
			-- 	)
			conv_params[i] =
							deepcopy( conv_params[1] )

			end

		conv_params[1][1][1] = ins


		for	i	,	c	in	ipairs(conv_params)	do

			if	i ~= 1	then
				conv_params	[ i ]
							[ 1 ]
							[ 1 ] =
										-- output of last
											conv_params	[
															i-1
														]
														[
															#conv_params[ i-1 ]
														]
														[
															2
														]
										--	+	input to first
										+	conv_params	[
															i-1
														]
														[
															1
														]
														[
															1
														]
				end

			-- conv_params[i][2] =
			-- 						ins
			-- 					+	conv_params[i][1]

			end

		print('conv_params', conv_params)


		for	i	,	c	in	ipairs(conv_params)	do
			module
				:add(
					nn.Concat(2)
						:add(
							nn.Identity()
							)
						-- :add(
							-- nn.Sequential()
						:add(
							Convolutions( c )
							)
							-- )
					)
			end


		module
			:add(
				Convolution	(
										conv_params	[ #conv_params 			]
													[ #conv_params[1] 		]
													[ 2 ]
									-- +	ins
									+
										conv_params	[ #conv_params 			]
													[ 1				 		]
													[ 1 ]
									-- ins*4
								,	outs
							,
								1,1
							-- ,
							-- 	1.1
							-- ,
							-- 	0,0
							-- ,
							-- 	1
							)
				)
			:add(
				nn.SpatialBatchNormalization( outs ,1e-5)
				)
			:add(
				nn.ReLU(true)
				)


		ins	=	outs
		end

	return	module

	end



-- pad = (ker-1)/2
--
-- recursive =
-- 	nn.Sequential()
-- 		:add(
-- 			Convolution
-- 				(
-- 					x	,	x*4
-- 				,	ker	,	ker
-- 				,	1	,	1
-- 				,	pad	,	pad
-- 				)
-- 			)
-- 		:add(
-- 			nn.Concat(2)
-- 				:add(
-- 					recurse
-- 						(
-- 							depth-1
-- 						,	x*4
-- 						,	ker
-- 						)
-- 					)
-- 				:add(
-- 					)
-- 			)
-- 		:add(
-- 			nn.SpatialBatchNormalization
-- 				(
-- 				x*4	,	1e-5
-- 				)
-- 			)
-- 		-- :add(
-- 		-- 	nn.ReLU(true)
-- 		-- 	)
-- 		:add(
-- 			nn.PReLU
-- 				(
-- 				x*4
-- 				)
-- 			)
-- 		:add(
-- 			Convolution
-- 				(
-- 					x*4	,	x*2
-- 				,	1	,	1
-- 				,	1	,	1
-- 				,	0	,	0
-- 				)
-- 			)
-- 		:add(
-- 			nn.SpatialBatchNormalization
-- 				(
-- 				x*2	,	1e-5
-- 				)
-- 			)
-- 		:add(
-- 			nn.PReLU
-- 				(
-- 				x*2
-- 				)
-- 			)
-- 		:add(
-- 			Convolution
-- 				(
-- 					x*2	,	x
-- 				,	1	,	1
-- 				,	1	,	1
-- 				,	0	,	0
-- 				)
-- 			)
-- 		:add(
-- 			nn.SpatialBatchNormalization
-- 				(
-- 				x	,	1e-5
-- 				)
-- 			)
		-- :add(
		-- 	nn.PReLU
		-- 		(
		-- 		x
		-- 		)
		-- 	)
			-- nn.ReLU(true)





local function createModel(opt)

	width		= opt.widen_factor
	depth		= opt.depth
	sequences	= opt.sequences

	local model = nn.Sequential()

	local function Block(...)
		local arg = {...}
		model:add(cudnn.SpatialConvolution(...):noBias())
		model:add(nn.SpatialBatchNormalization(arg[2],1e-5))
		model:add(nn.ReLU(true))
	 --	model:add(nn.ELU())
		return model
	end

	--32
	Block(3,	32,		3,3,	1,1,	1,1				,1)

	model
		-- :add(
		-- 	build	(
		-- 				3
		-- 			,	16
		-- 			)
		-- 	)
		-- :add(
		-- 	build	(
		-- 				16
		-- 			,	32
		-- 			)
		-- 	)



		:add(
			build	(
						32
					,	64
					)
			)


		:add(
			nn.SpatialMaxPooling(3,3,2,2):ceil()
			)
		--16

		:add(
			build	(
						64
					,	128
					)
			)

		:add(
			Convolutions(
							{
								{	128	,	128	,	3,3	,	1,1,	1,1	,	8	}
							}
						)
			)

		:add(
			nn.SpatialMaxPooling(3,3,2,2):ceil()
			)
		--8

		:add(
			build	(
						128
					,	512
					)
			)

		:add(
			Convolutions(
							{
								{	512	,	512	,	3,3	,	1,1,	1,1	,	4	}
							}
						)
			)

		:add(
			nn.VolumetricAveragePooling(2,8,8, 2,1,1)
			)
		:add(
			nn.View(-1):setNumInputDims(3)
			)
		:add(
			nn.Linear	(
							256
						-- ,	opt and opt.num_classes or 10
						,	100
						)
			)

	utils.FCinit(model)
	utils.testModel(model)
	utils.MSRinit(model)

	return model
end

return createModel


-- model=nin ./scripts/train_cifar.sh
