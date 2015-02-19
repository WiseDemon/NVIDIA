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
#include "include/fieldsampler.h"
#include "../include/FieldSamplerCommon.h"


BOUND_KERNEL_BEG(clearKernel,
	float4* g_accumField, float4* g_accumVelocity
)
	for (unsigned int idx = BlockSize*blockIdx.x + threadIdx.x; idx < _threadCount; idx += BlockSize*gridDim.x)
	{
		g_accumField[idx] = make_float4(0, 0, 0, 0);
		g_accumVelocity[idx] = make_float4(0, 0, 0, 0);
	}
BOUND_KERNEL_END()


BOUND_KERNEL_BEG(composeKernel,
	float4* g_accumField, const float4* g_accumVelocity, const float4* g_velocity, physx::PxF32 timestep
)
	for (unsigned int idx = BlockSize*blockIdx.x + threadIdx.x; idx < _threadCount; idx += BlockSize*gridDim.x)
	{
		float4 avel4 = g_accumVelocity[idx];
		physx::PxVec3 avel(avel4.x, avel4.y, avel4.z);
		physx::PxF32 avelW = avel4.w;

		if (avelW >= VELOCITY_WEIGHT_THRESHOLD)
		{
			float4 vel4 = g_velocity[idx];
			physx::PxVec3 vel(vel4.x, vel4.y, vel4.z);

			float4 field4 = g_accumField[idx];
			physx::PxVec3 field(field4.x, field4.y, field4.z);

			field += (avel - avelW * vel);

			g_accumField[idx] = make_float4(field.x, field.y, field.z, 0);
		}
	}
BOUND_KERNEL_END()


FREE_KERNEL_3D_BEG(clearGridKernel,
	physx::PxU32 numX, physx::PxU32 numY, physx::PxU32 numZ
)
	unsigned int ix = idxX;
	unsigned int iy = idxY;
	unsigned int iz = idxZ;

	if (ix < numX && iy < numY && iz < numZ)
	{
		const unsigned short zeroValue = __float2half_rn(0.0f);
		const ushort4 outValue = make_ushort4(zeroValue, zeroValue, zeroValue, zeroValue);
		surf3Dwrite(outValue, KERNEL_SURF_REF(GridAccum), ix * sizeof(ushort4), iy, iz);
	}
FREE_KERNEL_3D_END()

BOUND_KERNEL_BEG(applyParticlesKernel,
	float4* g_velocity, const float4* g_outField
)
	for (unsigned int idx = BlockSize*blockIdx.x + threadIdx.x; idx < _threadCount; idx += BlockSize*gridDim.x)
	{
		g_velocity[idx].x += g_outField[idx].x;
		g_velocity[idx].y += g_outField[idx].y;
		g_velocity[idx].z += g_outField[idx].z;
	}
BOUND_KERNEL_END()

#ifdef APEX_TEST

BOUND_KERNEL_BEG(testParticleKernel,
	float4* g_position, float4* g_velocity,
	physx::PxU32* g_flag,
	const float4* g_initPosition, const float4* g_initVelocity
)
	for (unsigned int idx = BlockSize*blockIdx.x + threadIdx.x; idx < _threadCount; idx += BlockSize*gridDim.x)
	{
		
		testParticle((physx::PxVec4&)g_position[idx], (physx::PxVec4&)g_velocity[idx], g_flag[idx], (physx::PxVec4&)g_initPosition[idx], (physx::PxVec4&)g_initVelocity[idx]);
	}
BOUND_KERNEL_END()

#endif