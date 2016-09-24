-- This is a modified version of NIN network in
-- https://github.com/szagoruyko/cifar.torch
-- Network-In-Network: http://arxiv.org/abs/1312.4400
-- Modifications:
--  * removed dropout
--  * added BatchNorm
--  * the last layer changed from avg-pooling to linear (works better)
require 'nn'
local utils = paths.dofile'utils.lua'
-- local cudnn = require 'cudnn'
require 'cunn'
require 'cudnn'
---
-- torch.setdefaulttensortype('torch.CudaTensor')
---




function least_common_divisor( a, b )
	for i=2,100000 do
		if a%i == 0  and  b%i == 0 then
			return i
		end
	end
	return 1
end


function n_groups( whether, g )
	if	(
		whether	==	1
		)	then
			-- return g
			if	g<40	then
				return	g
				end

			a	=		g
					/	least_common_divisor(g,6)

			a	=		g
					/	least_common_divisor(a,6)

			return	a
					/	least_common_divisor(a,6)

			end
	if	(
		whether	~=	0
		)	then
			return whether
			end
	return 1
end

function pool_test( whether, model )
	if	(
		whether	==	1
		)	then
			model:add
				(
				nn.SpatialMaxPooling
					( 2,2,2,2 )
				)

			return model
			end
	if	(
		whether	==	2
		)	then
			model:add
				(
				nn.SpatialMaxPooling
					( 3,3,2,2 )
				)

			return model
			end
	if	(
		whether	==	3
		)	then
			model:add
				(
				nn.SpatialFractionalMaxPooling
					( 3,3, 0.5, 0.5 )
				)

			return model
			end
	if	(
		whether	==	4
		)	then
			model:add
				(
				nn.SpatialFractionalMaxPooling
					( 4,4, 0.5, 0.5 )
				)

			return model
			end
	if	(
		whether	==	5
		)	then
			model:add
				(
				nn.SpatialMaxPooling
					( 5,5 , 4,4 )
				)

			return model
			end
	-- if	(
	-- 	whether	==	2
	-- 	)	then
	-- 		model:add
	-- 			(
	-- 			nn.SpatialAvgPooling
	-- 				( 3,3,2,2 )
	-- 			)
	--
	-- 		return model
	-- 		end

	return model
end



nninit = require 'nninit'
SBatchNorm = cudnn.SpatialBatchNormalization

	paddings	=
		{
			0
		,	0
		,	0
		,	0
		,	0
		}
	strides	=
		{
			1
		,	1
		,	1
		,	1
		,	1
		}
	linear		=
		{
		-- 256
		0
		}
	grouped		=
		{
			0
		,	1
		,	1
		,	1
		,	0
		}


	features =
		{
		3
		--	32x32
		,
		-- 24
		36
		--	15x15
		,
		-- 1296
		-- 768
		1296
		--	7x7
		,
		-- 20736
		10368
		-- 6144
		-- 128
		--	5x5
		,
		-- 82944
		-- 1024
		10
		--	1x1
		}
	kernels		=
		{
			1
		,	3
		,	3
		,	4
		,	1
		}

	pool_after	=
		{
			0
		,	2
		,	2
		,	0
		,	5
		}
classes = {'airplane', 'automobile', 'bird', 'cat',
					'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}




local function createModel(opt)
   local model = nn.Sequential()


			resolution_after =
				{
				32
				}
				for i=2,#features do

					resolution_after[i]	=
						math.floor
								(	(
										resolution_after[i-1]
									+	2* paddings[i]
									-	kernels[i]
									)
										/	strides[i]
								)
									+ 1

					if	pool_after[i] ~=	0	then
						if	pool_after[i] ==	5	then
							resolution_after[i]
								=
									math.floor
											(
												resolution_after[i]
											/
												4
											)
						elseif
							pool_after[i] ==	2	then
							resolution_after[i]
								=
									math.floor
											(
												(resolution_after[i]-1)
											/
												2
											)
						else
							resolution_after[i]
								=
									math.floor
											(
												resolution_after[i]
											/
												2
											)
							end
						end

				end
			-- model:add(nn.SpatialConvolution(3,		, 5, 5))


			ith	=	1

				model	=	pool_test( pool_after[ith] , model)


			ith	=	2

				gr	=	n_groups( grouped[ith] , features[ith-1] )

				model
					:add(
						cudnn.SpatialConvolution(
							features[ith-1]
							,
							features[ith]
							,
							kernels[ith]			,	kernels[ith]
							,
							strides[ith]			,	strides[ith]
							,
							paddings[ith]			,	paddings[ith]
							,
							gr
							)
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
					:add(
						SBatchNorm
							(
							features[ith]
							)
						)

				model	=	pool_test( pool_after[ith] , model)

				model:add
					(
					nn.PReLU()
					)
			-- stage 2 : filter bank -> squashing -> max pooling

			-- groups	=	1
			-- model:add(nn.Reshape(	opt.batchSize * groups	, features[1] / groups	, 	resolution_after[2] , resolution_after[2]	,	false))
			ith		=	3

				gr	=	n_groups( grouped[ith] , features[ith-1] )

				model
					:add(
						cudnn.SpatialConvolution
							(
							features[ith-1]
							,
							features[ith]
							,
							kernels[ith]	,	kernels[ith]
							,
							strides[ith]	,	strides[ith]
							,
							paddings[ith]	,	paddings[ith]
							,
							gr
							)
							:init(
									'weight'
								,	nninit.orthogonal
								,	{
									dist = 'uniform',
									gain =
										{
										'lrPReLU'			,
										leakiness = 0.2
										}
									}
								)
						)
					:add(
						SBatchNorm
							(
							features[ith]
							)
						)

				model	=	pool_test( pool_after[ith] , model)

				model:add
					(
					nn.PReLU()
					)





			ith		=	4

				gr	=	n_groups( grouped[ith] , features[ith-1] )

				model
					:add(
						cudnn.SpatialConvolution(
							features[ith-1]
							,
							features[ith]
							,
							kernels[ith]			,	kernels[ith]
							,
							strides[ith]			,	strides[ith]
							,
							paddings[ith]			,	paddings[ith]
							,
							gr
							)
							:init(
									'weight'
								,	nninit.orthogonal
								,	{
									dist = 'uniform',
									gain =
										{
										'lrPReLU'			,
										leakiness = 0.2
										}
									}
								)
						)
					:add(
						SBatchNorm
							(
							features[ith]
							)
						)


				model	=	pool_test( pool_after[ith] , model)

				model:add
					(
					nn.PReLU()
					)


			-- ith		=	5
			--
			-- 	gr	=	n_groups( grouped[ith] , features[ith-1] )
			--
			-- 	model
			-- 		:add(
			-- 			cudnn.SpatialConvolution(
			-- 				features[ith-1]
			-- 				,
			-- 				features[ith]
			-- 				,
			-- 				kernels[ith]			,	kernels[ith]
			-- 				,
			-- 				strides[ith]			,	strides[ith]
			-- 				,
			-- 				paddings[ith]			,	paddings[ith]
			-- 				,
			-- 				gr
			-- 				)
			-- 				-- :init(
			-- 				-- 		'weight'
			-- 				-- 	,	nninit.kaiming
			-- 				-- 	,	{
			-- 				-- 		dist = 'uniform',
			-- 				-- 		gain =
			-- 				-- 			{
			-- 				-- 			'lrPReLU'			,
			-- 				-- 			leakiness = 0.3
			-- 				-- 			}
			-- 				-- 		}
			-- 				-- 	)
			-- 			)
			-- 		:add(
			-- 			SBatchNorm
			-- 				(
			-- 				features[ith]
			-- 				)
			-- 			)
			--
			--
			-- 	model	=	pool_test( pool_after[ith] , model)
			-- --
			-- 	model:add
			-- 		(
			-- 		nn.PReLU()
			-- 		)


			-- model:add(nn.Reshape(	opt.batchSize 			, features[2]			,	resolution_after[3] , resolution_after[3]	,	false))


			-- stage 3 : standard 2-layer neural network
			print(resolution_after[ith],	features[ith])
			model
				:add(
					nn.Reshape
						(
							resolution_after[ith]
						*	resolution_after[ith]
						*	features[ith]
						)
					)
			-- model:add(nn.View(-1))
				-- :add(
				-- 	nn.Linear
				-- 		(
				-- 			resolution_after[ith]
				-- 		*	resolution_after[ith]
				-- 		*	features[ith]
				-- 		,
				-- 		linear[1]
				-- 		)
				-- 	)
				-- :add(
				-- 	nn.PReLU()
				-- 	)

					-- :add(
					-- 	cudnn.BatchNormalization
					-- 		(
					-- 		linear[1]
					-- 		)
					-- 	)

				:add(
					-- nn.LinearDropconnect
					nn.Linear
						(
							resolution_after[ith]
						*	resolution_after[ith]
						*	features[ith]

						-- linear[1]
						,
						-- #classes
						opt and opt.num_classes or 100
						)
					)



   return model
end

return createModel


-- model=nin ./scripts/train_cifar.sh
