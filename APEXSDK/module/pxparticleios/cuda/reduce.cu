/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "include/common.h"
#include "common.cuh"
#include "blocksync.cuh"
#include "reduce.cuh"

#include "include/reduce.h"


inline __device__ void reduce1(unsigned int count, float* g_benefit, float4* g_output,
	unsigned int			*g_tmpActiveCount,
	float					*g_tmpBenefitSum,
	float					*g_tmpBenefitMin,
	float					*g_tmpBenefitMax,
	volatile unsigned int	*sdataActiveCount,
	volatile float			*sdataBenefitSum,
	volatile float			*sdataBenefitMin,
	volatile float			*sdataBenefitMax)
{
	const unsigned int BlockSize = blockDim.x;
	const unsigned int idx = threadIdx.x;

	sdataActiveCount[idx] = AddOPu::identity();
	sdataBenefitSum[idx]  = AddOPf::identity();
	sdataBenefitMin[idx]  = MinOPf::identity();
	sdataBenefitMax[idx]  = MaxOPf::identity();

	for (unsigned int pos = BlockSize*blockIdx.x + idx; pos < count; pos += BlockSize*gridDim.x)
	{
		float benefit = g_benefit[pos];
		//benefit > -FLT_MAX also handles NaN values!
		if (benefit > -FLT_MAX)
		{
			sdataActiveCount[idx] = AddOPu::apply(sdataActiveCount[idx], 1);
			sdataBenefitSum[idx]  = AddOPf::apply(sdataBenefitSum[idx], benefit);
			sdataBenefitMin[idx]  = MinOPf::apply(sdataBenefitMin[idx], benefit);
			sdataBenefitMax[idx]  = MaxOPf::apply(sdataBenefitMax[idx], benefit);
		}
	}

	//don't need to synch because we use whole WARPs here
	reduceWarp<unsigned int, AddOPu>(sdataActiveCount);
	reduceWarp<float, AddOPf>(sdataBenefitSum);
	reduceWarp<float, MinOPf>(sdataBenefitMin);
	reduceWarp<float, MaxOPf>(sdataBenefitMax);

	//merge all warps for block
	__syncthreads();

	reduceBlock<unsigned int, AddOPu>(sdataActiveCount, g_tmpActiveCount);
	reduceBlock<float, AddOPf>(sdataBenefitSum,  g_tmpBenefitSum);
	reduceBlock<float, MinOPf>(sdataBenefitMin,  g_tmpBenefitMin);
	reduceBlock<float, MaxOPf>(sdataBenefitMax,  g_tmpBenefitMax);
}

inline __device__ void reduce2(float* g_benefit, float4* g_output,
	unsigned int			*g_tmpActiveCount,
	float					*g_tmpBenefitSum,
	float					*g_tmpBenefitMin,
	float					*g_tmpBenefitMax,
	volatile unsigned int	*sdataActiveCount,
	volatile float			*sdataBenefitSum,
	volatile float			*sdataBenefitMin,
	volatile float			*sdataBenefitMax,
	unsigned int gridSize)
{
	reduceGrid<unsigned int, AddOPu>(sdataActiveCount, g_tmpActiveCount, gridSize);
	reduceGrid<float, AddOPf>(sdataBenefitSum,  g_tmpBenefitSum, gridSize);
	reduceGrid<float, MinOPf>(sdataBenefitMin,  g_tmpBenefitMin, gridSize);
	reduceGrid<float, MaxOPf>(sdataBenefitMax,  g_tmpBenefitMax, gridSize);

	if (threadIdx.x == 0)
	{
		g_output[0] = make_float4(__int_as_float( sdataActiveCount[0] ),
			sdataBenefitSum[0],
			sdataBenefitMin[0],
			sdataBenefitMax[0]
		);
	}
}

#define REDUCE_KERNEL_SETUP() \
	unsigned int* g_tmpActiveCount = g_tmp; \
	float* g_tmpBenefitSum = (float*)(g_tmp + WARP_SIZE); \
	float* g_tmpBenefitMin = (float*)(g_tmp + WARP_SIZE*2); \
	float* g_tmpBenefitMax = (float*)(g_tmp + WARP_SIZE*3); \
	extern __shared__ volatile unsigned int sdata[]; /* [BlockSize * 4] */ \
	volatile unsigned int* sdataActiveCount = sdata; \
	volatile float* sdataBenefitSum = (volatile float*)(sdata + BlockSize); \
	volatile float* sdataBenefitMin = (volatile float*)(sdata + BlockSize*2); \
	volatile float* sdataBenefitMax = (volatile float*)(sdata + BlockSize*3); \

SYNC_KERNEL_BEG(reduceSyncKernel,
	unsigned int count, float* g_benefit,
	float4* g_output, unsigned int* g_tmp
)
	REDUCE_KERNEL_SETUP()

	reduce1(count, g_benefit, g_output,
		g_tmpActiveCount, g_tmpBenefitSum, g_tmpBenefitMin, g_tmpBenefitMax,
		sdataActiveCount, sdataBenefitSum, sdataBenefitMin, sdataBenefitMax);

	if (threadIdx.x == 0) {
		__threadfence(); //only one write per block
	}

	BLOCK_SYNC_BEGIN()

	reduce2(g_benefit, g_output,
		g_tmpActiveCount, g_tmpBenefitSum, g_tmpBenefitMin, g_tmpBenefitMax,
		sdataActiveCount, sdataBenefitSum, sdataBenefitMin, sdataBenefitMax,
		gridDim.x);

	BLOCK_SYNC_END()

SYNC_KERNEL_END()

BOUND_KERNEL_BEG(reduceKernel,
	float* g_benefit, float4* g_output, unsigned int* g_tmp,
	unsigned int phase, unsigned int gridSize
)
	REDUCE_KERNEL_SETUP()

	if (phase == 1)
	{
		reduce1(_threadCount, g_benefit, g_output,
			g_tmpActiveCount, g_tmpBenefitSum, g_tmpBenefitMin, g_tmpBenefitMax,
			sdataActiveCount, sdataBenefitSum, sdataBenefitMin, sdataBenefitMax);
	}
	else
	{
		reduce2(g_benefit, g_output,
			g_tmpActiveCount, g_tmpBenefitSum, g_tmpBenefitMin, g_tmpBenefitMax,
			sdataActiveCount, sdataBenefitSum, sdataBenefitMin, sdataBenefitMax,
			gridSize);
	}

BOUND_KERNEL_END()
