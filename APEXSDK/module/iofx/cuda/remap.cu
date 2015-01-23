/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#define APEX_CUDA_STORAGE_NAME remapStorage
#include "include/common.h"
#include "common.cuh"

using namespace physx::apex;
using namespace physx::apex::iofx;
#include "include/remap.h"


__device__
PX_INLINE unsigned int floatFlip(float f)
{
	unsigned int i = __float_as_int(f);
	unsigned int mask = -int(i >> 31) | 0x80000000;
	return i ^ mask;
}

INPLACE_TEMPL_ARGS_DEF
__device__ PX_INLINE physx::PxU32 getActorIndex(physx::PxU32 inputID, const ActorIDRemapArray& actorIDRemapArray, physx::PxU32 numActorsPerVolume, physx::PxU32 numActorIDs)
{
	NiIofxActorID id;
	id.value = tex1Dfetch(KERNEL_TEX_REF(RemapActorIDs), inputID);

	physx::PxU32 actorIndex = numActorIDs;
	if (id.getVolumeID() != NiIofxActorID::NO_VOLUME)
	{
		physx::PxU32 actorID;
		actorIDRemapArray.fetchElem(KERNEL_CONST_STORAGE, actorID, id.getActorClassID());
		actorIndex = numActorsPerVolume * id.getVolumeID() + actorID;
	}
	return actorIndex;
}

/* State Space */
BOUND_S2_KERNEL_BEG(makeSortKeys,
	((const physx::PxU32*, inStateToInput))((physx::PxU32, maxInputID))
	((physx::PxU32, numActorsPerVolume))((physx::PxU32, numActorIDs))
	((InplaceHandle<ActorIDRemapArray>, actorIDRemapArrayHandle))
	((const float4*, positionMass))((bool, outputDensityKeys))
	((physx::PxVec3, eyePos))((physx::PxVec3, eyeDir))((physx::PxF32, zNear))
	((physx::PxU32*, sortKey))((physx::PxU32*, sortValue))
)
	ActorIDRemapArray actorIDRemapArray;
	actorIDRemapArrayHandle.fetch(KERNEL_CONST_STORAGE, actorIDRemapArray);

	const physx::PxU32 maxStateID = _threadCount;
	for (physx::PxU32 stateID = BlockSize*blockIdx.x + threadIdx.x; stateID < maxStateID; stateID += BlockSize*gridDim.x)
	{
		physx::PxU32 key = outputDensityKeys ? 0xFFFFFFFFu : (numActorIDs + 1);
		physx::PxU32 value = stateID;

		physx::PxU32 inputID = inStateToInput[ stateID ];
		inputID &= ~NiIosBufferDesc::NEW_PARTICLE_FLAG;
		if (inputID < maxInputID) //this will check also that (inputID != NiIosBufferDesc::NOT_A_PARTICLE)
		{
			if (outputDensityKeys)
			{
				const float4 pos4 = tex1Dfetch(KERNEL_TEX_REF(RemapPositions), inputID);
				const physx::PxVec3 pos = physx::PxVec3(pos4.x, pos4.y, pos4.z);
				const float dist = zNear + (eyePos - pos).dot(eyeDir);
				key = floatFlip( dist );

				//store distance sign in the highest bit of value
				value |= (key & STATE_ID_DIST_SIGN);
			}
			else
			{
				key = getActorIndex INPLACE_TEMPL_ARGS_VAL (inputID, actorIDRemapArray, numActorsPerVolume, numActorIDs);
			}
		}
		sortKey[ stateID ] = key;
		sortValue[ stateID ] = value;
	}
BOUND_S2_KERNEL_END()


/* Sorted State Space */
BOUND_S2_KERNEL_BEG(remapKernel,
	((const physx::PxU32*, inStateToInput))((physx::PxU32, maxInputID))
	((physx::PxU32, numActorsPerVolume))((physx::PxU32, numActorIDs))
	((InplaceHandle<ActorIDRemapArray>, actorIDRemapArrayHandle))
	((const unsigned int*, inSortedValue))((unsigned int*, outSortKey))
)
	ActorIDRemapArray actorIDRemapArray;
	actorIDRemapArrayHandle.fetch(KERNEL_CONST_STORAGE, actorIDRemapArray);

	const physx::PxU32 maxStateID = _threadCount;
	for (physx::PxU32 stateID = BlockSize*blockIdx.x + threadIdx.x; stateID < maxStateID; stateID += BlockSize*gridDim.x)
	{
		physx::PxU32 actorIndex = (numActorIDs + 1);

		const physx::PxU32 sortedStateID = (inSortedValue[ stateID ] & STATE_ID_MASK);
		// sortedStateID should be < maxStateID
		physx::PxU32 inputID = tex1Dfetch(KERNEL_TEX_REF(RemapInStateToInput), sortedStateID);
		inputID &= ~NiIosBufferDesc::NEW_PARTICLE_FLAG;
		if (inputID < maxInputID) //this will check also that (inputID != NiIosBufferDesc::NOT_A_PARTICLE)
		{
			actorIndex = getActorIndex INPLACE_TEMPL_ARGS_VAL (inputID, actorIDRemapArray, numActorsPerVolume, numActorIDs);
		}

		outSortKey[ stateID ] = actorIndex;
	}
BOUND_S2_KERNEL_END()
