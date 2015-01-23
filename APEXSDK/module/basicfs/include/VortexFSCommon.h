/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __VORTEX_FS_COMMON_SRC_H__
#define __VORTEX_FS_COMMON_SRC_H__

#include "../../fieldsampler/include/FieldSamplerCommon.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

//struct VortexFSParams
#define INPLACE_TYPE_STRUCT_NAME VortexFSParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,			worldToDir) \
	INPLACE_TYPE_FIELD(InplaceBool,						bottomSphericalForce) \
	INPLACE_TYPE_FIELD(InplaceBool,						topSphericalForce) \
	INPLACE_TYPE_FIELD(physx::PxF32,					height) \
	INPLACE_TYPE_FIELD(physx::PxF32,					bottomRadius) \
	INPLACE_TYPE_FIELD(physx::PxF32,					topRadius) \
	INPLACE_TYPE_FIELD(physx::PxF32,					rotationalStrength) \
	INPLACE_TYPE_FIELD(physx::PxF32,					radialStrength) \
	INPLACE_TYPE_FIELD(physx::PxF32,					liftStrength)
#include INPLACE_TYPE_BUILD()


PX_CUDA_CALLABLE PX_INLINE physx::PxF32 sqr(physx::PxF32 x)
{
	return x * x;
}

/*
PX_CUDA_CALLABLE PX_INLINE physx::PxVec3 executeVortexFS_GRID(const VortexFSParams& params)
{
	return params.worldToDir.M.multiplyByTranspose(physx::PxVec3(0, params.strength, 0));
}*/

APEX_CUDA_CALLABLE PX_INLINE physx::PxVec3 executeVortexFS(const VortexFSParams& params, const physx::PxVec3& pos/*, physx::PxU32 totalElapsedMS*/)
{
	PX_ASSERT(params.bottomRadius);
	PX_ASSERT(params.topRadius);
	
	PxVec3 result(PxZero);
	PxVec3 point = params.worldToDir * pos;
	PxF32 R = PxSqrt(point.x * point.x + point.z * point.z);
	PxF32 invR = 1.f / R;
	PxF32 invRS = invR;
	PxF32 curR = 0;
	PxF32 h = params.height, r1 = params.bottomRadius, r2 = params.topRadius, y = point.y;

	if (y < h/2 && y > -h/2)
	{
		curR = r1 + (r2-r1) * (y / h + 0.5f);
	}
	else if (y <= -h/2 && y >= -h/2-r1)
	{
		curR = PxSqrt(r1*r1 - sqr(y+h/2));
		if (params.bottomSphericalForce)
		{
			PxF32 y = point.y + h/2;
			invRS = 1.f / PxSqrt(point.x * point.x + y * y + point.z * point.z);
			result.y = params.radialStrength * y;
		}
	}
	else if (y >= h/2 && y <= h/2+r2)
	{
		curR = PxSqrt(r2*r2 - sqr(y-h/2));
		if (params.topSphericalForce)
		{
			PxF32 y = point.y - h/2;
			invRS = 1.f / PxSqrt(point.x * point.x + y * y + point.z * point.z);
			result.y = params.radialStrength * y;
		}
	}

	if (curR > 0.f && R <= curR)
	{
		result.x += params.radialStrength * point.x * invRS - params.rotationalStrength * R / curR * point.z * invR;
		result.y += params.liftStrength;
		result.z += params.radialStrength * point.z * invRS + params.rotationalStrength * R / curR * point.x * invR;
	}

	return params.worldToDir.M.multiplyByTranspose(result);
}

}
}
} // namespace apex

#endif
