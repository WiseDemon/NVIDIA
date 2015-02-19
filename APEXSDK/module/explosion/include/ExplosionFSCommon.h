/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __EXPLOSION_FS_COMMON_SRC_H__
#define __EXPLOSION_FS_COMMON_SRC_H__

#include "../../fieldsampler/include/FieldSamplerCommon.h"

namespace physx
{
namespace apex
{
namespace explosion
{

//struct ExplosionFSParams
#define INPLACE_TYPE_STRUCT_NAME ExplosionFSParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxMat44,	pose) \
	INPLACE_TYPE_FIELD(physx::PxF32,	radius) \
	INPLACE_TYPE_FIELD(physx::PxF32,	strength)
#include INPLACE_TYPE_BUILD()


PX_CUDA_CALLABLE PX_INLINE physx::PxVec3 executeExplosionMainFS(const ExplosionFSParams& params, const physx::PxVec3& pos, const physx::PxU32& /*totalElapsedMS*/)
{
	// bring pos to explosion's coordinate system
	physx::PxVec3 localPos = params.pose.inverseRT().rotate(pos);
	physx::PxVec3 localPosFromExplosion = localPos - params.pose.getPosition();

	if (localPosFromExplosion.magnitude() < params.radius)
	{
		physx::PxVec3 result = localPosFromExplosion;
		result.y = 2.5f * physx::PxSqrt(params.radius * params.radius - result.x * result.x - result.z * result.z);
		result = params.strength * result;
		return result;
	}

	return physx::PxVec3(0, 0, 0);
}

PX_CUDA_CALLABLE PX_INLINE physx::PxVec3 executeExplosionFS(const ExplosionFSParams& params, const physx::PxVec3& pos, const physx::PxU32& totalElapsedMS)
{
	physx::PxVec3 resultField(0, 0, 0);
	resultField += executeExplosionMainFS(params, pos, totalElapsedMS);
	return resultField;
}

}
}
} // end namespace physx::apex

#endif
