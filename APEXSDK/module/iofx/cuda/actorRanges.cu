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

using namespace physx::apex;
using namespace physx::apex::iofx;
#include "include/actorRanges.h"
#include "NiIofxManager.h"


BOUND_KERNEL_BEG(actorRangeKernel,
	const physx::PxU32* sortedActorID, physx::PxU32 numActorIDs,
	physx::PxU32* actorStart, physx::PxU32* actorEnd, physx::PxU32* actorVisibleEnd,
	const physx::PxU32* sortedStateID
)
	extern __shared__ physx::PxU32 sdata[]; /* size = [BlockSize + 1] */
	physx::PxU32* sdataVisible = sdata + (BlockSize + 1);  /* size = [BlockSize + 1] */

	const physx::PxU32 idx = threadIdx.x;

	const physx::PxU32 outputCount = _threadCount;
	for (unsigned int outputBeg = BlockSize * blockIdx.x; outputBeg < outputCount; outputBeg += BlockSize * gridDim.x)
	{
		const unsigned int outputEnd = min(outputBeg + BlockSize, outputCount);
		const unsigned int output = outputBeg + idx;

		sdata[idx] = (output < outputCount) ? sortedActorID[output] : UINT_MAX;
		sdataVisible[idx] = (output < outputCount) ? (sortedStateID[output] >> 31) : UINT_MAX;
		if (idx == 0) {
			sdata[BlockSize] = (outputEnd < outputCount) ? sortedActorID[outputEnd] : UINT_MAX;
			sdataVisible[BlockSize] = (outputEnd < outputCount) ? (sortedStateID[outputEnd] >> 31) : UINT_MAX;
		}
		__syncthreads();

		if (output < outputEnd)
		{
			const physx::PxU32 currActorIndex = sdata[idx];
			const physx::PxU32 nextActorIndex = sdata[idx + 1];
			if (nextActorIndex != currActorIndex)
			{
				if (nextActorIndex != UINT_MAX)
				{
					actorStart[nextActorIndex] = output + 1;
					if (sdataVisible[idx + 1] != 0)
					{
						actorVisibleEnd[nextActorIndex] = output + 1;
					}
				}
				if (currActorIndex != UINT_MAX)
				{
					actorEnd[currActorIndex] = output + 1;
					if (sdataVisible[idx] == 0)
					{
						actorVisibleEnd[currActorIndex] = output + 1;
					}
				}
			}
			else if (sdataVisible[idx] != sdataVisible[idx + 1])
			{
				if (currActorIndex != UINT_MAX)
				{
					actorVisibleEnd[currActorIndex] = output + 1;
				}
			}
		}
		__syncthreads();
	}
BOUND_KERNEL_END()
