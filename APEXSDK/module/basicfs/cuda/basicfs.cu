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
using namespace physx::apex::basicfs;
#include "include/basicfs.h"


inline __device__ physx::PxF32 evalWeightInShapeGrid(const fieldsampler::FieldShapeParams& shapeParams, const physx::PxVec3& position, const physx::PxVec3& cellSize)
{
	const physx::PxF32 cellRadius = fieldsampler::scaleToShape(shapeParams, cellSize).magnitude() * 0.5f;
	const physx::PxF32 dist = fieldsampler::evalDistInShape(shapeParams, position);
	return fieldsampler::evalFadeAntialiasing(dist, shapeParams.fade, cellRadius) * shapeParams.weight;
}


template <>
struct FieldSamplerIncludeWeightEvaluator<fieldsampler::FieldSamplerKernelType::GRID>
{
	INPLACE_TEMPL_ARGS_DEF
	static inline __device__ physx::PxF32 func(const fieldsampler::FieldSamplerParams* params, const physx::PxVec3& position, const physx::PxVec3& cellSize)
	{
		physx::PxF32 result;
		switch (params->executeType)
		{
		case 1:
			{
				JetFSParams jetParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, jetParams);			
				result = evalWeightInShapeGrid(jetParams.gridIncludeShape, position, cellSize);
			}
			break;
		default:
			{
				result = evalWeightInShapeGrid(params->includeShape, position, cellSize);
			}
			break;
		}
		return result;
	}
};

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
				JetFSParams executeParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, executeParams);
				result = executeJetFS(executeParams, args.position, args.totalElapsedMS);
			}
			break;
		case 2:
			{
				AttractorFSParams executeParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, executeParams);
				result = executeAttractorFS(executeParams, args.position);
			}
			break;
		case 3:
			{
				NoiseFSParams executeParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, executeParams);
				result = executeNoiseFS(executeParams, args.position, args.totalElapsedMS);
			}
			break;
		case 4:
			{
				VortexFSParams executeParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, executeParams);
				result = executeVortexFS(executeParams, args.position);
			}
			break;
		case 5:
			{
				WindFSParams executeParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, executeParams);
				result = executeWindFS(executeParams, args.position);
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
		physx::PxVec3 result;
		switch (params->executeType)
		{
		case 1:
			{
				JetFSParams executeParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, executeParams);
				result = executeJetFS_GRID(executeParams);
			}
			break;
		case 3:
			{
				NoiseFSParams executeParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, executeParams);
				result = executeNoiseFS_GRID(executeParams, args.position, args.totalElapsedMS);
			}
			break;
		case 4:
			{
				VortexFSParams executeParams;
				params->executeParamsHandle.fetch(KERNEL_CONST_STORAGE, executeParams);
				//result = executeVortexFS_GRID(executeParams, args.position);
				result = executeVortexFS(executeParams, args.position);
			}
			break;
		default:
			result = physx::PxVec3(0.0f);
			break;
		}
		return result;
	}
};

#include "../../fieldsampler/cuda/include/fieldsamplerInc.cuh"

