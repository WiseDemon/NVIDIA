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

#include "include/scan.h"


inline __device__ void reduceWarp(unsigned int idx, volatile unsigned int* sdata)
{
	sdata[idx] += sdata[idx + 16];
	sdata[idx] += sdata[idx +  8];
	sdata[idx] += sdata[idx +  4];
	sdata[idx] += sdata[idx +  2];
	sdata[idx] += sdata[idx +  1];
}

inline __device__ void bitScanWarp(unsigned int idx, unsigned idxInWarp, volatile unsigned int* sdata)
{
	sdata[idx] <<= idxInWarp;
	reduceWarp(idx, sdata);

	unsigned int result = sdata[ idx & ~(WARP_SIZE-1) ];

	sdata[idx] = __popc( result & (0xFFFFFFFFU >> (31-idxInWarp)) );
}

inline __device__ int evalBin(float benefit, float benefitMin, float benefitMax)
{
	//benefit > -FLT_MAX also handles NaN values!
	return (benefit > -FLT_MAX) ? ((benefit - benefitMin) * HISTOGRAM_BIN_COUNT / (benefitMax - benefitMin)) : -1;
}

inline __device__ unsigned int condition(int bin, int boundBin)
{
	return (bin < boundBin) ? 1 : 0;
}
inline __device__ unsigned int condition1(int bin, int boundBin)
{
	return (bin == boundBin) ? 1 : 0;
}


#define SCAN_LOOP_COMMON(whole) \
	sdata[idx] = marker; \
	bitScanWarp(idx, idxInWarp, sdata); \
	if (whole || pos < warpEnd) g_indices[pos] = (prefix + sdata[idx]) | (marker << HOLE_SCAN_FLAG_BIT); \
	if (whole) prefix += sdata[(warpIdx << LOG2_WARP_SIZE) + WARP_SIZE-1];

#define SCAN_LOOP_1(whole) \
	unsigned int marker = 0; \
	if (whole || pos < warpEnd) { \
		float benefit = g_benefits[pos]; \
		int bin = evalBin(benefit, benefitMin, benefitMax); \
		marker = condition(bin, markBin); \
	} \
	SCAN_LOOP_COMMON(whole)

#define SCAN_LOOP_2(whole) \
	unsigned int marker = 0; \
	if (whole || pos < warpEnd) { \
		float benefit = g_benefits[pos]; \
		int bin = evalBin(benefit, benefitMin, benefitMax); \
		marker = condition(bin, boundBin); \
		marker |= condition1(bin, boundBin) << 1; \
	} \
	if (prefix1 < boundCount) \
	{ \
		sdata1[idx] = (marker >> 1); \
		bitScanWarp(idx, idxInWarp, sdata1); \
		marker |= (marker >> 1) & ((prefix1 + sdata1[idx] <= boundCount) ? 1 : 0); \
		if (whole) prefix1 += sdata1[(warpIdx << LOG2_WARP_SIZE) + WARP_SIZE-1]; \
	} \
	marker &= 1; \
	SCAN_LOOP_COMMON(whole)


inline __device__ void scan1(unsigned int count,
	float benefitMin, float benefitMax, unsigned int* g_indices, const float* g_benefits,
	unsigned int* g_boundParams, unsigned int* g_tmpCounts, unsigned int* g_tmpCounts1,
	volatile unsigned int* sdata, volatile unsigned int* sdata1,
	const unsigned int warpBeg, const unsigned int warpEnd)
{
	const unsigned int BlockSize = blockDim.x;
	const unsigned int WarpsPerBlock = (BlockSize >> LOG2_WARP_SIZE);

	const unsigned int idx = threadIdx.x;
	const unsigned int idxInWarp = idx & (WARP_SIZE-1);

	__shared__ int boundBin;
	if (idx == 0)
	{
		boundBin = g_boundParams[1];
	}
	__syncthreads();

	sdata[idx] = 0;
	sdata1[idx] = 0;

	if (warpBeg < warpEnd)
	{
		//accum
		for (unsigned int i = warpBeg + idxInWarp; i < warpEnd; i += WARP_SIZE)
		{
			float benefit = g_benefits[i];

			int bin = evalBin(benefit, benefitMin, benefitMax);
			sdata[idx] += condition(bin, boundBin);
			sdata1[idx] += condition1(bin, boundBin);
		}
		//reduce warp
		reduceWarp(idx, sdata);
		reduceWarp(idx, sdata1);
	}

	__syncthreads();

	if (idx < WarpsPerBlock)
	{
		g_tmpCounts[blockIdx.x * WarpsPerBlock + idx] = sdata[idx << LOG2_WARP_SIZE];
		g_tmpCounts1[blockIdx.x * WarpsPerBlock + idx] = sdata1[idx << LOG2_WARP_SIZE];
	}
}


inline __device__ void scanWarp(unsigned int scanIdx, volatile unsigned int* sdata)
{
	sdata[scanIdx] += sdata[scanIdx -  1];
	sdata[scanIdx] += sdata[scanIdx -  2];
	sdata[scanIdx] += sdata[scanIdx -  4];
	sdata[scanIdx] += sdata[scanIdx -  8];
	sdata[scanIdx] += sdata[scanIdx - 16]; 
}

inline __device__ void scan2(
	float benefitMin, float benefitMax, unsigned int* g_indices, const float* g_benefits,
	unsigned int* g_boundParams, unsigned int* g_tmpCounts, unsigned int* g_tmpCounts1,
	volatile unsigned int* sdata, volatile unsigned int* sdata1,
	unsigned int gridSize)
{
	const unsigned int BlockSize = blockDim.x;
	const unsigned int WarpsPerBlock = (BlockSize >> LOG2_WARP_SIZE);
	const unsigned int WarpsPerGrid = WarpsPerBlock * gridSize;
	//gridSize can be > WARP_SIZE, so we use x2 scan below to support gridSize up to 64!!!
#if MAX_BOUND_BLOCKS > 64
#error MAX_BOUND_BLOCKS > 64 is not supported
#endif
#if MAX_WARPS_PER_BLOCK > WARP_SIZE
#error MAX_WARPS_PER_BLOCK > WARP_SIZE is not supported
#endif

	const unsigned int ScanCount = (WarpsPerGrid >> 1); //>> 1 for x2 scan
	const unsigned int ScanWarps = (ScanCount + WARP_SIZE-1) >> LOG2_WARP_SIZE;

	__shared__ volatile unsigned int sScanForWarp[MAX_WARPS_PER_BLOCK];
	__shared__ volatile unsigned int sScanForWarp1[MAX_WARPS_PER_BLOCK];

	const unsigned int idx = threadIdx.x;
	const unsigned int warpIdx = (idx >> LOG2_WARP_SIZE);
	const unsigned int idxInWarp = idx & (WARP_SIZE-1);
	unsigned int scanIdx = (warpIdx << (LOG2_WARP_SIZE + 1)) + idxInWarp;

	uint2 val, val1;
	unsigned int res;
	unsigned int res1;
	if (warpIdx < ScanWarps)
	{
		val = val1 = make_uint2(0, 0);
		if (idx < ScanCount)
		{
			val = ((uint2*)g_tmpCounts)[idx];
			val1 = ((uint2*)g_tmpCounts1)[idx];
		}

		//setup scan
		sdata[scanIdx] = 0;
		sdata1[scanIdx] = 0;
		scanIdx += WARP_SIZE;
		sdata[scanIdx] = val.x + val.y;
		sdata1[scanIdx] = val1.x + val1.y;

		scanWarp(scanIdx, sdata);
		scanWarp(scanIdx, sdata1);

		res = sdata[scanIdx];
		res1 = sdata1[scanIdx];

		if (idxInWarp == WARP_SIZE-1)
		{
			sScanForWarp[warpIdx] = res;
			sScanForWarp1[warpIdx] = res1;
		}
	}
	__syncthreads();

	//1 warp scan
	if (idx < WARP_SIZE)
	{
		sdata[scanIdx] = sScanForWarp[idx];
		sdata1[scanIdx] = sScanForWarp1[idx];

		scanWarp(scanIdx, sdata);
		scanWarp(scanIdx, sdata1);
	}
	__syncthreads();

	if (warpIdx < ScanWarps)
	{
		//-1 for exclusive scan
		res += sdata[warpIdx + WARP_SIZE - 1];
		res1 += sdata1[warpIdx + WARP_SIZE - 1];

		val.x = res - val.y; val.y = res;
		val1.x = res1 - val1.y; val1.y = res1;

		if (idx < ScanCount)
		{
			((uint2*)g_tmpCounts)[idx] = val;
			((uint2*)g_tmpCounts1)[idx] = val1;
		}
	}
}

inline __device__ void scan3(unsigned int count,
	float benefitMin, float benefitMax, unsigned int* g_indices, const float* g_benefits,
	unsigned int* g_boundParams, unsigned int* g_tmpCounts, unsigned int* g_tmpCounts1,
	volatile unsigned int* sdata, volatile unsigned int* sdata1,
	const unsigned int warpBeg, const unsigned int warpEnd)
{
	const unsigned int BlockSize = blockDim.x;
	const unsigned int WarpsPerBlock = (BlockSize >> LOG2_WARP_SIZE);

	const unsigned int idx = threadIdx.x;
	const unsigned int idxInWarp = idx & (WARP_SIZE-1);
	const unsigned int warpIdx = (idx >> LOG2_WARP_SIZE);

	__shared__ unsigned int sCounts[MAX_WARPS_PER_BLOCK+1];
	__shared__ unsigned int sCounts1[MAX_WARPS_PER_BLOCK+1];

	__shared__ unsigned int boundCount;
	__shared__ int          boundBin;
	if (idx == 0)
	{
		boundCount = g_boundParams[0];
		boundBin   = g_boundParams[1];

		sCounts[0]  = (blockIdx.x > 0) ? g_tmpCounts[blockIdx.x * WarpsPerBlock - 1] : 0;
		sCounts1[0] = (blockIdx.x > 0) ? g_tmpCounts1[blockIdx.x * WarpsPerBlock - 1] : 0;
	}
	if (idx < WarpsPerBlock)
	{
		sCounts[idx+1]  = g_tmpCounts[blockIdx.x * WarpsPerBlock + idx];
		sCounts1[idx+1] = g_tmpCounts1[blockIdx.x * WarpsPerBlock + idx];
	}
	__syncthreads();

	if (warpBeg < warpEnd)
	{
		unsigned int prefix = sCounts[warpIdx];
		unsigned int prefix1 = sCounts1[warpIdx];

		if (prefix1 >= boundCount || boundCount >= sCounts1[warpIdx+1])
		{
			prefix += min(prefix1, boundCount);
			int markBin = (prefix1 >= boundCount) ? boundBin : (boundBin + 1);

			unsigned int pos;
			for (pos = warpBeg + idxInWarp; pos < (warpEnd & ~(WARP_SIZE-1)); pos += WARP_SIZE)
			{
				SCAN_LOOP_1(true)
			}
			if ((warpEnd & (WARP_SIZE-1)) > 0)
			{
				SCAN_LOOP_1(false)
			}
		}
		else
		{
			prefix += prefix1;

			unsigned int pos;
			for (pos = warpBeg + idxInWarp; pos < (warpEnd & ~(WARP_SIZE-1)); pos += WARP_SIZE)
			{
				SCAN_LOOP_2(true)
			}
			if ((warpEnd & (WARP_SIZE-1)) > 0)
			{
				SCAN_LOOP_2(false)
			}
		}
	}
}

#define SCAN_KERNEL_SETUP(count) \
	const unsigned int DataWarpsPerGrid = ((count + WARP_SIZE-1) >> LOG2_WARP_SIZE); \
	const unsigned int DataWarpsPerBlock = (DataWarpsPerGrid + gridDim.x-1) / gridDim.x; \
	const unsigned int DataCountPerBlock = (DataWarpsPerBlock << LOG2_WARP_SIZE); \
	const unsigned int WarpBorder = DataWarpsPerBlock % WarpsPerBlock; \
	const unsigned int WarpFactor = DataWarpsPerBlock / WarpsPerBlock; \
	const unsigned int warpIdx = (threadIdx.x >> LOG2_WARP_SIZE); \
	const unsigned int blockBeg = blockIdx.x * DataCountPerBlock; \
	const unsigned int blockEnd = min(blockBeg + DataCountPerBlock, count); \
	const unsigned int WarpSelect = (warpIdx < WarpBorder) ? 1 : 0; \
	const unsigned int WarpCount = WarpFactor + WarpSelect; \
	const unsigned int WarpOffset = warpIdx * WarpCount + WarpBorder * (1 - WarpSelect); \
	const unsigned int warpBeg = blockBeg + (WarpOffset << LOG2_WARP_SIZE); \
	const unsigned int warpEnd = min(warpBeg + (WarpCount << LOG2_WARP_SIZE), blockEnd); \
	extern __shared__ volatile unsigned int sdata[]; /* size = [BlockSize * 2] */ \
	volatile unsigned int* sdata1 = sdata + BlockSize * 2; /* size = [BlockSize * 2] */ \

SYNC_KERNEL_BEG(scanSyncKernel, unsigned int count,
	float benefitMin, float benefitMax, unsigned int* g_indices, const float* g_benefits, unsigned int* g_boundParams, unsigned int* g_tmpCounts, unsigned int* g_tmpCounts1
)
	SCAN_KERNEL_SETUP(count)

	scan1(count,
		benefitMin, benefitMax, g_indices, g_benefits,
		g_boundParams, g_tmpCounts, g_tmpCounts1,
		sdata, sdata1,
		warpBeg, warpEnd);

	if (threadIdx.x < WarpsPerBlock)
	{
		__threadfence();
	}

	BLOCK_SYNC_BEGIN()

	scan2(
		benefitMin, benefitMax, g_indices, g_benefits,
		g_boundParams, g_tmpCounts, g_tmpCounts1,
		sdata, sdata1,
		gridDim.x);

	if (threadIdx.x < WarpsPerBlock * gridDim.x)
	{
		__threadfence();
	}

	BLOCK_SYNC_END()

	scan3(count,
		benefitMin, benefitMax, g_indices, g_benefits,
		g_boundParams, g_tmpCounts, g_tmpCounts1,
		sdata, sdata1,
		warpBeg, warpEnd);

SYNC_KERNEL_END()


BOUND_KERNEL_BEG(scanKernel,
	float benefitMin, float benefitMax, unsigned int* g_indices, float* g_benefits, unsigned int* g_boundParams, unsigned int* g_tmpCounts, unsigned int* g_tmpCounts1,
	unsigned int phase, unsigned int gridSize
)
	SCAN_KERNEL_SETUP(_threadCount)

	if (phase == 1)
	{
		scan1(_threadCount,
			benefitMin, benefitMax, g_indices, g_benefits,
			g_boundParams, g_tmpCounts, g_tmpCounts1,
			sdata, sdata1,
			warpBeg, warpEnd);
	}
	else if (phase == 2)
	{
		scan2(benefitMin, benefitMax, g_indices, g_benefits,
			g_boundParams, g_tmpCounts, g_tmpCounts1,
			sdata, sdata1,
			gridSize);
	}
	else
	{
		scan3(_threadCount,
			benefitMin, benefitMax, g_indices, g_benefits,
			g_boundParams, g_tmpCounts, g_tmpCounts1,
			sdata, sdata1,
			warpBeg, warpEnd);
	}

BOUND_KERNEL_END()
