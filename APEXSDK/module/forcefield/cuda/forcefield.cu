/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#define APEX_CUDA_STORAGE_NAME fieldSamplerStorage
#include "include/common.h"
#include "common.cuh"

using namespace physx::apex;
#include "include/forcefield.h"

template <>
struct FieldSamplerExecutor<fieldsampler::FieldSamplerKernelType::POINTS>
{
	INPLACE_TEMPL_ARGS_DEF
	static inline __device__ physx::PxVec3 func(const fieldsampler::FieldSamplerParams* params, const fieldsampler::FieldSamplerExecuteArgs& args, physx::PxF32& fieldWeight)
	{
		physx::PxVec3 result;
		switch (params->executeType)
		{
		case 1:
			{
				forcefield::GenericForceFieldFSKernelParams executeParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, executeParams);
				result = executeForceFieldFS(executeParams, args.position, args.velocity, args.totalElapsedMS);
			}
			break;
		case 2:
			{
				forcefield::RadialForceFieldFSKernelParams executeParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, executeParams);
				result = executeForceFieldFS(executeParams, args.position, args.totalElapsedMS);
			}
			break;
		}
		return result;
	}
};

template <>
struct FieldSamplerExecutor<fieldsampler::FieldSamplerKernelType::GRID>
{
	INPLACE_TEMPL_ARGS_DEF
	static inline __device__ physx::PxVec3 func(const fieldsampler::FieldSamplerParams* params, const fieldsampler::FieldSamplerExecuteArgs& args, physx::PxF32& fieldWeight)
	{
		//we have only 1 executeType so ignore it
		return physx::PxVec3(0, 0, 0);
	}
};

#include "../../fieldsampler/cuda/include/fieldsamplerInc.cuh"
