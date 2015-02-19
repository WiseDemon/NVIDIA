/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


APEX_CUDA_SURFACE_3D(surfRefFieldSamplerGridAccum)


#ifdef FIELD_SAMPLER_SEPARATE_KERNELS

APEX_CUDA_BOUND_KERNEL(FIELD_SAMPLER_POINTS_KERNEL_CONFIG, fieldSamplerPointsKernel,
                       ((fieldsampler::FieldSamplerKernelParams, baseParams))
                       ((APEX_MEM_BLOCK(float4), accumField))
                       ((APEX_MEM_BLOCK(float4), accumVelocity))
                       ((APEX_MEM_BLOCK(const float4), positionMass))
                       ((APEX_MEM_BLOCK(const float4), velocity))
                       ((fieldsampler::FieldSamplerParamsEx, paramsEx))
                       ((InplaceHandle<fieldsampler::FieldSamplerQueryParams>, queryParamsHandle))
					   ((fieldsampler::FieldSamplerKernelMode::Enum, kernelMode))
                      )
APEX_CUDA_FREE_KERNEL_3D(FIELD_SAMPLER_GRID_KERNEL_CONFIG, fieldSamplerGridKernel,
                       ((fieldsampler::FieldSamplerKernelParams, baseParams))
                       ((fieldsampler::FieldSamplerGridKernelParams, gridParams))
                       ((fieldsampler::FieldSamplerParamsEx, paramsEx))
                       ((InplaceHandle<fieldsampler::FieldSamplerQueryParams>, queryParamsHandle))
					   ((fieldsampler::FieldSamplerKernelMode::Enum, kernelMode))
                      )

#ifndef __CUDACC__
#define LAUNCH_FIELD_SAMPLER_KERNEL( launchData ) \
	const ApexCudaConstStorage& _storage_ = *getFieldSamplerCudaConstStorage(); \
	InplaceHandle<fieldsampler::FieldSamplerQueryParams> queryParamsHandle = _storage_.mappedHandle( launchData.queryParamsHandle ); \
	PxU32 fieldSamplerCount = launchData.fieldSamplerArray->size(); \
	switch( launchData.kernelType ) \
	{ \
	case fieldsampler::FieldSamplerKernelType::POINTS: \
		{ \
			const fieldsampler::NiFieldSamplerPointsKernelLaunchData& data = static_cast<const fieldsampler::NiFieldSamplerPointsKernelLaunchData&>(launchData); \
			const fieldsampler::FieldSamplerPointsKernelArgs* kernelArgs = static_cast<const fieldsampler::FieldSamplerPointsKernelArgs*>(data.kernelArgs); \
			for (PxU32 i = 0, activeIdx = 0; i < fieldSamplerCount; ++i) \
			{ \
				const fieldsampler::FieldSamplerWrapperGPU* wrapper = static_cast<const fieldsampler::FieldSamplerWrapperGPU* >( (*data.fieldSamplerArray)[i].mFieldSamplerWrapper ); \
				if (wrapper->isEnabled()) \
				{ \
					fieldsampler::FieldSamplerParamsEx paramsEx; \
					paramsEx.paramsHandle = _storage_.mappedHandle( wrapper->getParamsHandle() ); \
					paramsEx.multiplier = (*data.fieldSamplerArray)[i].mMultiplier; \
					fieldsampler::FieldSamplerKernelMode::Enum kernelMode = (++activeIdx == data.activeFieldSamplerCount) ? data.kernelMode : fieldsampler::FieldSamplerKernelMode::DEFAULT; \
					ON_LAUNCH_FIELD_SAMPLER_KERNEL( wrapper->getNiFieldSampler(), wrapper->getNiFieldSamplerDesc() ); \
					SCENE_CUDA_OBJ(this, fieldSamplerPointsKernel)( data.stream, data.threadCount, \
						*static_cast<const fieldsampler::FieldSamplerKernelParams*>(kernelArgs), \
						physx::apex::createApexCudaMemRef(kernelArgs->accumField, size_t(data.memRefSize)), \
						physx::apex::createApexCudaMemRef(kernelArgs->accumVelocity, size_t(data.memRefSize)), \
						physx::apex::createApexCudaMemRef(kernelArgs->positionMass, size_t(data.memRefSize), ApexCudaMemFlags::IN), \
						physx::apex::createApexCudaMemRef(kernelArgs->velocity, size_t(data.memRefSize), ApexCudaMemFlags::IN), \
						paramsEx, queryParamsHandle, kernelMode ); \
				} \
			} \
		} \
		return true; \
	case fieldsampler::FieldSamplerKernelType::GRID: \
		{ \
			const fieldsampler::NiFieldSamplerGridKernelLaunchData& data = static_cast<const fieldsampler::NiFieldSamplerGridKernelLaunchData&>(launchData); \
			const fieldsampler::FieldSamplerGridKernelArgs* kernelArgs = static_cast<const fieldsampler::FieldSamplerGridKernelArgs*>(data.kernelArgs); \
			SCENE_CUDA_OBJ(this, surfRefFieldSamplerGridAccum).bindTo(*data.accumArray, ApexCudaMemFlags::IN_OUT); \
			for (PxU32 i = 0, activeIdx = 0; i < fieldSamplerCount; ++i) \
			{ \
				const fieldsampler::FieldSamplerWrapperGPU* wrapper = static_cast<const fieldsampler::FieldSamplerWrapperGPU* >( (*data.fieldSamplerArray)[i].mFieldSamplerWrapper ); \
				if (wrapper->isEnabled() && wrapper->getNiFieldSamplerDesc().gridSupportType == NiFieldSamplerGridSupportType::VELOCITY_PER_CELL) \
				{ \
					fieldsampler::FieldSamplerParamsEx paramsEx; \
					paramsEx.paramsHandle = _storage_.mappedHandle( wrapper->getParamsHandle() ); \
					paramsEx.multiplier = (*data.fieldSamplerArray)[i].mMultiplier; \
					fieldsampler::FieldSamplerKernelMode::Enum kernelMode = (++activeIdx == data.activeFieldSamplerCount) ? data.kernelMode : fieldsampler::FieldSamplerKernelMode::DEFAULT; \
					ON_LAUNCH_FIELD_SAMPLER_KERNEL( wrapper->getNiFieldSampler(), wrapper->getNiFieldSamplerDesc() ); \
					SCENE_CUDA_OBJ(this, fieldSamplerGridKernel)( data.stream, data.threadCountX, data.threadCountY, data.threadCountZ, \
						*static_cast<const fieldsampler::FieldSamplerKernelParams*>(kernelArgs), \
						*static_cast<const fieldsampler::FieldSamplerGridKernelParams*>(kernelArgs), \
						paramsEx, queryParamsHandle, kernelMode ); \
				} \
			} \
			SCENE_CUDA_OBJ(this, surfRefFieldSamplerGridAccum).unbind(); \
		} \
		return true; \
	default: \
		PX_ALWAYS_ASSERT(); \
		return false; \
	};
#endif

#else

APEX_CUDA_BOUND_KERNEL(FIELD_SAMPLER_POINTS_KERNEL_CONFIG, fieldSamplerPointsKernel,
                       ((fieldsampler::FieldSamplerKernelParams, baseParams))
                       ((APEX_MEM_BLOCK(float4), accumField))
                       ((APEX_MEM_BLOCK(float4), accumVelocity))
                       ((APEX_MEM_BLOCK(const float4), positionMass))
                       ((APEX_MEM_BLOCK(const float4), velocity))
                       ((InplaceHandle<fieldsampler::FieldSamplerParamsExArray>, paramsExArrayHandle))
                       ((InplaceHandle<fieldsampler::FieldSamplerQueryParams>, queryParamsHandle))
					   ((fieldsampler::FieldSamplerKernelMode::Enum, kernelMode))
                      )
APEX_CUDA_FREE_KERNEL_3D(FIELD_SAMPLER_GRID_KERNEL_CONFIG, fieldSamplerGridKernel,
                       ((fieldsampler::FieldSamplerKernelParams, baseParams))
                       ((fieldsampler::FieldSamplerGridKernelParams, gridParams))
                       ((InplaceHandle<fieldsampler::FieldSamplerParamsExArray>, paramsExArrayHandle))
                       ((InplaceHandle<fieldsampler::FieldSamplerQueryParams>, queryParamsHandle))
					   ((fieldsampler::FieldSamplerKernelMode::Enum, kernelMode))
                      )

#ifndef __CUDACC__
#define LAUNCH_FIELD_SAMPLER_KERNEL( launchData ) \
	const ApexCudaConstStorage& _storage_ = *getFieldSamplerCudaConstStorage(); \
	InplaceHandle<fieldsampler::FieldSamplerParamsExArray> paramsExArrayHandle = _storage_.mappedHandle( launchData.paramsExArrayHandle ); \
	InplaceHandle<fieldsampler::FieldSamplerQueryParams> queryParamsHandle = _storage_.mappedHandle( launchData.queryParamsHandle ); \
	switch( launchData.kernelType ) \
	{ \
	case fieldsampler::FieldSamplerKernelType::POINTS: \
		{ \
			const fieldsampler::NiFieldSamplerPointsKernelLaunchData& data = static_cast<const fieldsampler::NiFieldSamplerPointsKernelLaunchData&>(launchData); \
			const fieldsampler::FieldSamplerPointsKernelArgs* kernelArgs = static_cast<const fieldsampler::FieldSamplerPointsKernelArgs*>(data.kernelArgs); \
			SCENE_CUDA_OBJ(this, fieldSamplerPointsKernel)( data.stream, data.threadCount, \
				*static_cast<const fieldsampler::FieldSamplerKernelParams*>(kernelArgs), \
				physx::apex::createApexCudaMemRef(kernelArgs->accumField, size_t(data.memRefSize)), \
				physx::apex::createApexCudaMemRef(kernelArgs->accumVelocity, size_t(data.memRefSize)), \
				physx::apex::createApexCudaMemRef(kernelArgs->positionMass, size_t(data.memRefSize), ApexCudaMemFlags::IN), \
				physx::apex::createApexCudaMemRef(kernelArgs->velocity, size_t(data.memRefSize), ApexCudaMemFlags::IN), \
				paramsExArrayHandle, queryParamsHandle, data.kernelMode ); \
		} \
		return true; \
	case fieldsampler::FieldSamplerKernelType::GRID: \
		{ \
			const fieldsampler::NiFieldSamplerGridKernelLaunchData& data = static_cast<const fieldsampler::NiFieldSamplerGridKernelLaunchData&>(launchData); \
			const fieldsampler::FieldSamplerGridKernelArgs* kernelArgs = static_cast<const fieldsampler::FieldSamplerGridKernelArgs*>(data.kernelArgs); \
			SCENE_CUDA_OBJ(this, surfRefFieldSamplerGridAccum).bindTo(*data.accumArray, ApexCudaMemFlags::IN_OUT); \
			SCENE_CUDA_OBJ(this, fieldSamplerGridKernel)( data.stream, data.threadCountX, data.threadCountY, data.threadCountZ, \
				*static_cast<const fieldsampler::FieldSamplerKernelParams*>(kernelArgs), \
				*static_cast<const fieldsampler::FieldSamplerGridKernelParams*>(kernelArgs), \
				paramsExArrayHandle, queryParamsHandle, data.kernelMode ); \
			SCENE_CUDA_OBJ(this, surfRefFieldSamplerGridAccum).unbind(); \
		} \
		return true; \
	default: \
		PX_ALWAYS_ASSERT(); \
		return false; \
	};
#endif

#endif //FIELD_SAMPLER_SEPARATE_KERNELS
