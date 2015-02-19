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

#include "include/histogram.h"


const unsigned int TAG_SHIFT = (32 - LOG2_WARP_SIZE);
const unsigned int TAG_MASK = (1U << TAG_SHIFT) - 1;

typedef volatile unsigned int histogram_t;

inline __device__ void addToBin(histogram_t *s_WarpHist, unsigned int data, unsigned int threadTag)
{
	unsigned int count;
	do {
		count = s_WarpHist[data] & TAG_MASK;
		count = threadTag | (count + 1);
		s_WarpHist[data] = count;
	} while (s_WarpHist[data] != count);
}


template <int BinCount>
inline __device__ void histogram1(unsigned int count,
	const float *g_data, unsigned int bound, float dataMin, float dataMax, unsigned int* g_boundParams, unsigned int* g_tmpHistograms,
	histogram_t* s_Hist
)
{
	const unsigned int BlockSize = blockDim.x;
	const unsigned int WarpsPerBlock = (BlockSize >> LOG2_WARP_SIZE);

	const unsigned int idx = threadIdx.x;

	const unsigned int warpIdx = (idx >> LOG2_WARP_SIZE);
	histogram_t* s_WarpHist = s_Hist + warpIdx * BinCount;

	//Clear shared memory storage for current threadblock before processing
	#pragma unroll
	for(unsigned int i = 0; i < (BinCount >> LOG2_WARP_SIZE); i++) {
	   s_Hist[idx + i * BlockSize] = 0;
	}

	__syncthreads();

	const unsigned int tag = (idx & (WARP_SIZE-1)) << TAG_SHIFT;

	for(unsigned int pos = (BlockSize*blockIdx.x + idx); pos < count; pos += BlockSize*gridDim.x)
	{
		float data = g_data[pos];
		if (data >= dataMin && data < dataMax)
		{
			unsigned int bin = (data - dataMin)*BinCount/(dataMax - dataMin);
			addToBin(s_WarpHist, bin, tag);
		}
	}

	//Merge per-warp histograms into per-block and write to global memory
	__syncthreads();
	if (idx < BinCount)
	{
		unsigned int sum = 0;

		for(unsigned int i = 0; i < WarpsPerBlock; i++)
			sum += s_Hist[idx + i * BinCount] & TAG_MASK;

		g_tmpHistograms[blockIdx.x * BinCount + idx] = sum;
	}
}

template <int BinCount>
inline __device__ void histogram2(
	const float *g_data, unsigned int bound, float dataMin, float dataMax, unsigned int* g_boundParams, unsigned int* g_tmpHistograms,
	histogram_t* s_Hist, unsigned int gridSize
)
{
	const unsigned int idx = threadIdx.x;

	if (idx < BinCount)
	{
		s_Hist[idx] = 0;
		for (unsigned int i = 0; i < gridSize; ++i)
		{
			s_Hist[idx] += g_tmpHistograms[i*BinCount + idx];
		}
	}
	__syncthreads();

	//build CDF using prefix sum
	int pout = 0;
	int pin = 1;

	#pragma unroll
	for (int offset = 1; offset < BinCount; offset *= 2)
	{
		pout = 1 - pout;
		pin  = 1 - pout;

		if (idx < BinCount)
		{
			s_Hist[pout*BinCount + idx] = s_Hist[pin*BinCount + idx];
			if (idx >= offset)
				s_Hist[pout*BinCount + idx] += s_Hist[pin*BinCount + idx - offset];
#ifdef APEX_TEST
			g_tmpHistograms[pout*BinCount + idx] = s_Hist[pout*BinCount + idx];
#endif
		}

		__syncthreads();
	}

	if (idx == 0)
	{
		//unsigned int bound = g_bound[0];
		histogram_t* arr = s_Hist + pout*BinCount;
		
		//do binary search in CDF
		unsigned int beg = 0;
		unsigned int end = BinCount;
		while (beg < end)
		{
			unsigned int mid = beg + ((end - beg) >> 1);
			if (bound > arr[mid]) beg = mid + 1; else end = mid;
		}
		
		//g_dataMin[0] = dataMin + float(beg) * (dataMax - dataMin) / BinCount;
		//g_dataMax[0] = dataMin + float(beg+1) * (dataMax - dataMin) / BinCount;

		//assert( arr[beg] >= bound );
		g_boundParams[0] = bound - ((beg > 0) ? arr[beg-1] : 0);
		g_boundParams[1] = beg;
	}
}

SYNC_KERNEL_BEG(histogramSyncKernel, unsigned int _threadCount,
	const float *g_data, unsigned int bound, float dataMin, float dataMax, unsigned int* g_boundParams, unsigned int* g_tmpHistograms
)
	extern __shared__ histogram_t s_Hist[]; /* size = [BinCount * WarpsPerBlock] */

	histogram1<HISTOGRAM_BIN_COUNT>(_threadCount, g_data, bound, dataMin, dataMax, g_boundParams, g_tmpHistograms, s_Hist);
	if (threadIdx.x < HISTOGRAM_BIN_COUNT)
	{
		__threadfence();
	}

	BLOCK_SYNC_BEGIN()

	histogram2<HISTOGRAM_BIN_COUNT>(g_data, bound, dataMin, dataMax, g_boundParams, g_tmpHistograms, s_Hist, gridDim.x);

	BLOCK_SYNC_END()

SYNC_KERNEL_END()

BOUND_KERNEL_BEG(histogramKernel,
	float *g_data, unsigned int bound, float dataMin, float dataMax, unsigned int* g_boundParams, unsigned int* g_tmpHistograms,
	unsigned int phase, unsigned int gridSize
)
	extern __shared__ histogram_t s_Hist[]; /* size = [BinCount * WarpsPerBlock] */

	if (phase == 1)
	{
		histogram1<HISTOGRAM_BIN_COUNT>(_threadCount, g_data, bound, dataMin, dataMax, g_boundParams, g_tmpHistograms, s_Hist);
	}
	else
	{
		histogram2<HISTOGRAM_BIN_COUNT>(g_data, bound, dataMin, dataMax, g_boundParams, g_tmpHistograms, s_Hist, gridSize);
	}

BOUND_KERNEL_END()
