/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __JET_FS_COMMON_SRC_H__
#define __JET_FS_COMMON_SRC_H__

#include "../../fieldsampler/include/FieldSamplerCommon.h"
#include "SimplexNoise.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

//struct JetFSParams
#define INPLACE_TYPE_STRUCT_NAME JetFSParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxF32,					strength) \
	INPLACE_TYPE_FIELD(physx::PxF32,					instStrength) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,			worldToDir) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,			worldToInstDir) \
	INPLACE_TYPE_FIELD(fieldsampler::FieldShapeParams,	gridIncludeShape) \
	INPLACE_TYPE_FIELD(physx::PxF32,					nearRadius) \
	INPLACE_TYPE_FIELD(physx::PxF32,					pivotRadius) \
	INPLACE_TYPE_FIELD(physx::PxF32,					farRadius) \
	INPLACE_TYPE_FIELD(physx::PxF32,					directionalStretch) \
	INPLACE_TYPE_FIELD(physx::PxF32,					averageStartDistance) \
	INPLACE_TYPE_FIELD(physx::PxF32,					averageEndDistance) \
	INPLACE_TYPE_FIELD(physx::PxF32,					pivotRatio) \
	INPLACE_TYPE_FIELD(physx::PxF32,					noiseStrength) \
	INPLACE_TYPE_FIELD(physx::PxF32,					noiseSpaceScale) \
	INPLACE_TYPE_FIELD(physx::PxF32,					noiseTimeScale) \
	INPLACE_TYPE_FIELD(physx::PxU32,					noiseOctaves)
#include INPLACE_TYPE_BUILD()


PX_CUDA_CALLABLE PX_INLINE PxF32 smoothstep(PxF32 x, PxF32 edge0, PxF32 edge1)
{
	//x should be >= 0
	x = (PxClamp(x, edge0, edge1) - edge0) / (edge1 - edge0);
	// Evaluate polynomial
	return x * x * (3 - 2 * x);
}

PX_CUDA_CALLABLE PX_INLINE PxF32 smoothstep1(PxF32 x, PxF32 edge)
{
	//x should be >= 0
	x = PxMin(x, edge) / edge;
	// Evaluate polynomial
	return x * x * (3 - 2 * x);
}

PX_CUDA_CALLABLE PX_INLINE physx::PxVec3 executeJetFS_GRID(const JetFSParams& params)
{
	return params.worldToDir.M.multiplyByTranspose(physx::PxVec3(0, params.strength, 0));
}

PX_CUDA_CALLABLE PX_INLINE physx::PxVec3 evalToroidalField(const JetFSParams& params, const physx::PxVec3& pos, const physx::PxMat34Legacy& worldToDir, physx::PxF32 strength0)
{
	PxVec3 point = worldToDir * pos;

	PxF32 r = PxSqrt(point.x * point.x + point.z * point.z);
	PxF32 h = point.y / params.directionalStretch;

	PxF32 t;
	{
		const PxF32 r1 = r - params.pivotRadius;
		const PxF32 a = params.pivotRatio;
		const PxF32 b = (params.pivotRatio - 1) * r1;
		const PxF32 c = r1 * r1 + h * h;

		t = (PxSqrt(b * b + 4 * a * c) - b) / (2 * a);
	}

	const PxF32 r0 = params.pivotRadius + t * ((params.pivotRatio - 1) / 2);

	const PxF32 d = r0 - r;
	const PxF32 cosAngle = d / PxSqrt(d * d + h * h);
	const PxF32 angleLerp = (cosAngle + 1) * 0.5f;

	PxF32 rr = (r > 1e-10f) ? (1 / r) : 0;

	PxF32 xRatio = point.x * rr;
	PxF32 zRatio = point.z * rr;

	PxVec3 dir;
	dir.x = xRatio * h;
	dir.y = d * params.directionalStretch;
	dir.z = zRatio * h;

	dir.normalize();

	PxF32 strength = 0.0f;
	if (t <= params.pivotRadius)
	{
		strength = strength0 * smoothstep1(t, params.pivotRadius - params.nearRadius);

		strength *= (params.pivotRadius - t) * rr;
	}
	strength /= (angleLerp + params.pivotRatio * (1 - angleLerp));

	return strength * worldToDir.M.multiplyByTranspose(dir);
}

PX_CUDA_CALLABLE PX_INLINE physx::PxVec3 executeJetFS(const JetFSParams& params, const physx::PxVec3& pos, physx::PxU32 totalElapsedMS)
{
	physx::PxVec3 avgField = evalToroidalField(params, pos, params.worldToDir, params.strength);
	physx::PxVec3 instField = evalToroidalField(params, pos, params.worldToInstDir, params.instStrength);

	physx::PxF32 distance = (pos - params.worldToDir.t).magnitude();
	physx::PxF32 lerpFactor = smoothstep(distance, params.averageStartDistance, params.averageEndDistance);
	physx::PxVec3 result = lerpFactor * avgField + (1 - lerpFactor) * instField;

	if (params.noiseStrength > 0)
	{
		//add some noise
		PxVec3 point = params.noiseSpaceScale * (params.worldToDir * pos);
		PxF32 time = (params.noiseTimeScale * 1e-3f) * totalElapsedMS;

		PxVec4 dFx;
		dFx.setZero();
		PxVec4 dFy;
		dFy.setZero();
		PxVec4 dFz;
		dFz.setZero();
		int seed = 0;
		PxF32 amp = 1.0f;
		for (PxU32 i = 0; i < params.noiseOctaves; ++i)
		{
			dFx += amp * SimplexNoise::eval4D(point.x, point.y, point.z, time, ++seed);
			dFy += amp * SimplexNoise::eval4D(point.x, point.y, point.z, time, ++seed);
			dFz += amp * SimplexNoise::eval4D(point.x, point.y, point.z, time, ++seed);

			point *= 2;
			time *= 2;
			amp *= 0.5f;
		}
		//get rotor
		PxVec3 rot;
		rot.x = dFz.y - dFy.z;
		rot.y = dFx.z - dFz.x;
		rot.z = dFy.x - dFx.y;

		result += params.noiseStrength * params.worldToDir.M.multiplyByTranspose(rot);
	}
	return result;
}

}
}
} // namespace apex

#endif
