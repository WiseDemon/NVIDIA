/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#define APEX_CUDA_STORAGE_NAME migrationStorage
#include "include/common.h"
#include "common.cuh"

using namespace physx::apex;
using namespace physx::apex::iofx;
#include "include/migration.h"


__device__
bool contains( const physx::PxBounds3& b, const physx::PxVec3& v )
{
	return !(v.x < b.minimum.x || v.x > b.maximum.x ||
			 v.y < b.minimum.y || v.y > b.maximum.y ||
			 v.z < b.minimum.z || v.z > b.maximum.z);
}

/* Input Space */
BOUND_S2_KERNEL_BEG(volumeMigrationKernel,
	((InplaceHandle<VolumeParamsArray>, volumeParamsArrayHandle))
	((InplaceHandle<ActorClassIDBitmapArray>, actorClassIDBitmapArrayHandle))
	((physx::PxU32, numActorClasses))((physx::PxU32, numVolumes))((physx::PxU32, numActorIDValues))
	((NiIofxActorID*, actorID))((physx::PxU32, maxInputID))
	((const float4*, positionMass))
	((physx::PxU32*, actorStart))((physx::PxU32*, actorEnd))((physx::PxU32*, actorVisibleEnd))
)
	for (unsigned int input = BlockSize*blockIdx.x + threadIdx.x; input < maxInputID; input += BlockSize*gridDim.x)
	{
		NiIofxActorID id = actorID[ input ];
		const float4 pos4 = positionMass[ input ];
		const physx::PxVec3 pos = physx::PxVec3(pos4.x, pos4.y, pos4.z);

		physx::PxU32 bit = id.getActorClassID();
		if (bit == NiIofxActorID::INV_ACTOR || bit >= numActorClasses)
		{
			id.set( NiIofxActorID::NO_VOLUME, NiIofxActorID::INV_ACTOR );
		}
		else
		{
			physx::PxU32 curPri = 0;
			physx::PxU32 curVID = NiIofxActorID::NO_VOLUME;
			
			VolumeParamsArray volumeParamsArray;
			volumeParamsArrayHandle.fetch(KERNEL_CONST_STORAGE, volumeParamsArray);

			ActorClassIDBitmapArray actorClassIDBitmapArray;
			actorClassIDBitmapArrayHandle.fetch(KERNEL_CONST_STORAGE, actorClassIDBitmapArray);

			for (physx::PxU32 i = 0 ; i < numVolumes ; i++)
			{
				physx::PxU32 actorClassIDBitmapElem;
				actorClassIDBitmapArray.fetchElem(KERNEL_CONST_STORAGE, actorClassIDBitmapElem, bit >> 5);
				// check if this volume affects the particle's IOFX Asset
				if (actorClassIDBitmapElem & (1u << (bit & 31)))
				{
					VolumeParams vol;
					volumeParamsArray.fetchElem(KERNEL_CONST_STORAGE, vol, i);

					// This volume owns this particle if:
					//  1. This volume bounds contain the particle
					//  2. This volume has the highest priority or was the previous owner
					if ( contains( vol.bounds, pos ) &&
						 (curVID == NiIofxActorID::NO_VOLUME || vol.priority > curPri || (vol.priority == curPri && id.getVolumeID() == i)) )
					{
						curVID = i;
						curPri = vol.priority;
					}
				}
				bit += numActorClasses;
			}

			id.setVolumeID( curVID );
		}
		actorID[ input ] = id;
	}

	// Clear actorID start/stop table
	for (physx::PxU32 idx = BlockSize*blockIdx.x + threadIdx.x; idx < numActorIDValues; idx += BlockSize*gridDim.x)
	{
		actorStart[ idx ] = 0;
		actorEnd[ idx ] = 0;
		actorVisibleEnd[ idx ] = 0;
	}
		
BOUND_S2_KERNEL_END()
