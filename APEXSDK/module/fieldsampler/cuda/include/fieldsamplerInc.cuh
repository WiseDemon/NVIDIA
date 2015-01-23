/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


INPLACE_TEMPL_ARGS_DEF
inline __device__ void iterateShapeGroup(InplaceHandle<fieldsampler::FieldShapeGroupParams> shapeGroupHandle, const physx::PxVec3& position, physx::PxF32& weight)
{
	fieldsampler::FieldShapeGroupParams shapeGroupParams;
	shapeGroupHandle.fetch(KERNEL_CONST_STORAGE, shapeGroupParams);

	physx::PxU32 shapeCount = shapeGroupParams.shapeArray.getSize();
	for (physx::PxU32 shapeIndex = 0; shapeIndex < shapeCount; ++shapeIndex)
	{
		fieldsampler::FieldShapeParams shapeParams;
		shapeGroupParams.shapeArray.fetchElem(KERNEL_CONST_STORAGE, shapeParams, shapeIndex);

		const physx::PxF32 shapeWeight = evalWeightInShape(shapeParams, position);
		weight = physx::PxMax(weight, shapeWeight);
	}
}

INPLACE_TEMPL_VA_ARGS_DEF(int queryType)
inline __device__ void fieldSamplerFunc(
	fieldsampler::FieldSamplerKernelParams                baseParams,
	const fieldsampler::FieldSamplerParams&               samplerParams,
	InplaceHandle<fieldsampler::FieldSamplerQueryParams>  queryParamsHandle,
	const physx::PxVec3&                                  position,
	const physx::PxVec3&                                  velocity,
	physx::PxF32                                          mass,
	physx::PxVec4&                                        accumAccel,
	physx::PxVec4&                                        accumVelocity,
	physx::PxF32                                          multiplier)
{
	physx::PxF32 excludeWeight = 0;

	physx::PxU32 shapeGroupCount = samplerParams.excludeShapeGroupHandleArray.getSize();
	for (physx::PxU32 shapeGroupIndex = 0; shapeGroupIndex < shapeGroupCount; ++shapeGroupIndex)
	{
		InplaceHandle<fieldsampler::FieldShapeGroupParams> shapeGroupHandle;
		samplerParams.excludeShapeGroupHandleArray.fetchElem(KERNEL_CONST_STORAGE, shapeGroupHandle, shapeGroupIndex);

		iterateShapeGroup INPLACE_TEMPL_ARGS_VAL ( shapeGroupHandle, position, excludeWeight );
	}

	physx::PxF32 includeWeight = FieldSamplerIncludeWeightEvaluator<queryType>::func INPLACE_TEMPL_ARGS_VAL (&samplerParams, position, baseParams.cellSize);
	physx::PxF32 weight = includeWeight * (1.0f - excludeWeight);
#if FIELD_SAMPLER_MULTIPLIER == FIELD_SAMPLER_MULTIPLIER_WEIGHT
	weight *= multiplier;
#endif

	//execute field
	fieldsampler::FieldSamplerExecuteArgs execArgs;
	execArgs.position = position;
	execArgs.velocity = velocity;
	execArgs.mass = mass;

	execArgs.elapsedTime = baseParams.elapsedTime;
	execArgs.totalElapsedMS = baseParams.totalElapsedMS;

	physx::PxF32 fieldWeight = 1;
	physx::PxVec3 fieldValue = FieldSamplerExecutor<queryType>::func INPLACE_TEMPL_ARGS_VAL (&samplerParams, execArgs, fieldWeight);
	//override field weight
	fieldWeight = weight;
#if FIELD_SAMPLER_MULTIPLIER == FIELD_SAMPLER_MULTIPLIER_VALUE
	fieldValue *= multiplier;
#endif

	//accum field
	switch (samplerParams.type)
	{
	case NiFieldSamplerType::FORCE:
		accumFORCE(execArgs, fieldValue, fieldWeight, accumAccel, accumVelocity);
		break;
	case NiFieldSamplerType::ACCELERATION:
		accumACCELERATION(execArgs, fieldValue, fieldWeight, accumAccel, accumVelocity);
		break;
	case NiFieldSamplerType::VELOCITY_DRAG:
		accumVELOCITY_DRAG(execArgs, samplerParams.dragCoeff, fieldValue, fieldWeight, accumAccel, accumVelocity);
		break;
	case NiFieldSamplerType::VELOCITY_DIRECT:
		accumVELOCITY_DIRECT(execArgs, fieldValue, fieldWeight, accumAccel, accumVelocity);
		break;
	};
}

INPLACE_TEMPL_VA_ARGS_DEF(int queryType)
inline __device__ void fieldSamplerFunc(
	fieldsampler::FieldSamplerKernelParams                baseParams,
	fieldsampler::FieldSamplerParamsEx                    paramsEx,
	InplaceHandle<fieldsampler::FieldSamplerQueryParams>  queryParamsHandle,
	const physx::PxVec3&                                  position,
	const physx::PxVec3&                                  velocity,
	physx::PxF32                                          mass,
	physx::PxVec4&                                        accumAccel,
	physx::PxVec4&                                        accumVelocity)
{
	fieldsampler::FieldSamplerParams samplerParams;
	paramsEx.paramsHandle.fetch(KERNEL_CONST_STORAGE, samplerParams);

	fieldSamplerFunc INPLACE_TEMPL_VA_ARGS_VAL(queryType) (
		baseParams, samplerParams, queryParamsHandle, position, velocity, mass,
		accumAccel, accumVelocity, paramsEx.multiplier);
}


template <int queryType>
inline __device__ bool isValidFieldSampler(const fieldsampler::FieldSamplerParams& samplerParams);

template <>
inline __device__ bool isValidFieldSampler<fieldsampler::FieldSamplerKernelType::POINTS>(const fieldsampler::FieldSamplerParams& samplerParams)
{
	return true;
}

template <>
inline __device__ bool isValidFieldSampler<fieldsampler::FieldSamplerKernelType::GRID>(const fieldsampler::FieldSamplerParams& samplerParams)
{
	return (samplerParams.gridSupportType == NiFieldSamplerGridSupportType::VELOCITY_PER_CELL);
}

INPLACE_TEMPL_VA_ARGS_DEF(int queryType)
inline __device__ void fieldSamplerFunc(
	fieldsampler::FieldSamplerKernelParams                        baseParams,
	InplaceHandle<fieldsampler::FieldSamplerParamsExArray>        paramsExArrayHandle,
	InplaceHandle<fieldsampler::FieldSamplerQueryParams>          queryParamsHandle,
	const physx::PxVec3&                                          position,
	const physx::PxVec3&                                          velocity,
	physx::PxF32                                                  mass,
	physx::PxVec4&                                                accumAccel,
	physx::PxVec4&                                                accumVelocity)
{
	fieldsampler::FieldSamplerParamsExArray paramsExArray;
	paramsExArrayHandle.fetch(KERNEL_CONST_STORAGE, paramsExArray);

	for (physx::PxU32 i = 0; i < paramsExArray.getSize(); ++i)
	{
		fieldsampler::FieldSamplerParamsEx paramsEx;
		paramsExArray.fetchElem(KERNEL_CONST_STORAGE, paramsEx, i);

		fieldsampler::FieldSamplerParams samplerParams;
		paramsEx.paramsHandle.fetch(KERNEL_CONST_STORAGE, samplerParams);

		if (isValidFieldSampler<queryType>(samplerParams))
		{
			fieldSamplerFunc INPLACE_TEMPL_VA_ARGS_VAL(queryType) (
				baseParams, samplerParams, queryParamsHandle, position, velocity, mass,
				accumAccel, accumVelocity, paramsEx.multiplier);
		}
	}
}

INPLACE_TEMPL_VA_ARGS_DEF(typename T)
inline __device__ void fieldSamplerPointsFunc(
	unsigned int                                                       count,
	fieldsampler::FieldSamplerKernelParams                             baseParams,
	float4*                                                            accumFieldArray,
	float4*                                                            accumVelocityArray,
	const float4*                                                      positionMassArray,
	const float4*                                                      velocityArray,
	T                                                                  paramsEx,
	InplaceHandle<fieldsampler::FieldSamplerQueryParams>               queryParamsHandle,
	fieldsampler::FieldSamplerKernelMode::Enum                         kernelMode)
{
	const unsigned int BlockSize = blockDim.x;

	for (unsigned int idx = BlockSize*blockIdx.x + threadIdx.x; idx < count; idx += BlockSize*gridDim.x)
	{
		float4 pos4 = positionMassArray[idx];
		float4 vel4 = velocityArray[idx];

		physx::PxVec3 position(pos4.x, pos4.y, pos4.z);
		physx::PxVec3 velocity(vel4.x, vel4.y, vel4.z);
		physx::PxF32  mass = pos4.w;

		const float4 field4 = accumFieldArray[idx];
		physx::PxVec4 accumField = physx::PxVec4(field4.x, field4.y, field4.z, field4.w);

		const float4 avel4 = accumVelocityArray[idx];
		physx::PxVec4 accumVelocity = physx::PxVec4(avel4.x, avel4.y, avel4.z, avel4.w);

		fieldSamplerFunc INPLACE_TEMPL_VA_ARGS_VAL(fieldsampler::FieldSamplerKernelType::POINTS) (
			baseParams, paramsEx, queryParamsHandle, position, velocity, mass,
			accumField, accumVelocity);

		switch (kernelMode)
		{
		case fieldsampler::FieldSamplerKernelMode::FINISH_PRIMARY:
			accumField.w = accumVelocity.w;
			accumVelocity.w = 0;
			break;
		case fieldsampler::FieldSamplerKernelMode::FINISH_SECONDARY:
			accumVelocity.w = accumField.w + accumVelocity.w * (1 - accumField.w);
			accumField.w = 0;
			break;
		default:
			break;
		};

		accumFieldArray[idx] = make_float4(accumField.x, accumField.y, accumField.z, accumField.w);
		accumVelocityArray[idx] = make_float4(accumVelocity.x, accumVelocity.y, accumVelocity.z, accumVelocity.w);
	}
}

INPLACE_TEMPL_VA_ARGS_DEF(typename T)
inline __device__ void fieldSamplerGridFunc(
	unsigned int                                                       ix,
	unsigned int                                                       iy,
	unsigned int                                                       iz,
	fieldsampler::FieldSamplerKernelParams                             baseParams,
	fieldsampler::FieldSamplerGridKernelParams                         gridParams,
	T                                                                  paramsEx,
	InplaceHandle<fieldsampler::FieldSamplerQueryParams>               queryParamsHandle,
	fieldsampler::FieldSamplerKernelMode::Enum                         kernelMode)
{
	if (ix < gridParams.numX && iy < gridParams.numY && iz < gridParams.numZ)
	{
		const physx::PxVec3 position = gridParams.gridToWorld * physx::PxVec3(ix, iy, iz);
		const physx::PxVec3 velocity(0, 0, 0);
		const physx::PxF32  mass = gridParams.mass;

		physx::PxVec4 accumField(0, 0, 0, 0);

		const ushort4 inValH4 = surf3Dread<ushort4>(KERNEL_SURF_REF(FieldSamplerGridAccum), ix * sizeof(ushort4), iy, iz);
		physx::PxVec4 accumVelocity = physx::PxVec4(__half2float(inValH4.x), __half2float(inValH4.y), __half2float(inValH4.z), __half2float(inValH4.w));

		fieldSamplerFunc INPLACE_TEMPL_VA_ARGS_VAL(fieldsampler::FieldSamplerKernelType::GRID) (
			baseParams, paramsEx, queryParamsHandle, position, velocity, mass,
			accumField, accumVelocity);

		const ushort4 outValH4 = make_ushort4(__float2half_rn(accumVelocity.x), __float2half_rn(accumVelocity.y), __float2half_rn(accumVelocity.z), __float2half_rn(accumVelocity.w));
		surf3Dwrite(outValH4, KERNEL_SURF_REF(FieldSamplerGridAccum), ix * sizeof(ushort4), iy, iz);
	}
}

#ifdef FIELD_SAMPLER_SEPARATE_KERNELS

BOUND_S2_KERNEL_BEG(fieldSamplerPointsKernel,
	((fieldsampler::FieldSamplerKernelParams, baseParams))
	((APEX_MEM_BLOCK(float4), accumField))
	((APEX_MEM_BLOCK(float4), accumVelocity))
	((APEX_MEM_BLOCK(const float4), positionMass))
	((APEX_MEM_BLOCK(const float4), velocity))
	((fieldsampler::FieldSamplerParamsEx, paramsEx))
	((InplaceHandle<fieldsampler::FieldSamplerQueryParams>, queryParamsHandle))
	((fieldsampler::FieldSamplerKernelMode::Enum, kernelMode))
)
	fieldSamplerPointsFunc INPLACE_TEMPL_ARGS_VAL (_threadCount, baseParams, accumField, accumVelocity, positionMass, velocity, paramsEx, queryParamsHandle, kernelMode);
BOUND_S2_KERNEL_END()

FREE_S2_KERNEL_3D_BEG(fieldSamplerGridKernel,
	((fieldsampler::FieldSamplerKernelParams, baseParams))
	((fieldsampler::FieldSamplerGridKernelParams, gridParams))
	((fieldsampler::FieldSamplerParamsEx, paramsEx))
	((InplaceHandle<fieldsampler::FieldSamplerQueryParams>, queryParamsHandle))
	((fieldsampler::FieldSamplerKernelMode::Enum, kernelMode))
)
	fieldSamplerGridFunc INPLACE_TEMPL_ARGS_VAL (idxX, idxY, idxZ, baseParams, gridParams, paramsEx, queryParamsHandle, kernelMode);
FREE_S2_KERNEL_3D_END()

#else

BOUND_S2_KERNEL_BEG(fieldSamplerPointsKernel,
	((fieldsampler::FieldSamplerKernelParams, baseParams))
	((APEX_MEM_BLOCK(float4), accumField))
	((APEX_MEM_BLOCK(float4), accumVelocity))
	((APEX_MEM_BLOCK(const float4), positionMass))
	((APEX_MEM_BLOCK(const float4), velocity))
	((InplaceHandle<fieldsampler::FieldSamplerParamsExArray>, paramsExArrayHandle))
	((InplaceHandle<fieldsampler::FieldSamplerQueryParams>, queryParamsHandle))
	((fieldsampler::FieldSamplerKernelMode::Enum, kernelMode))
)
	fieldSamplerPointsFunc INPLACE_TEMPL_ARGS_VAL (_threadCount, baseParams, accumField, accumVelocity, positionMass, velocity, paramsExArrayHandle, queryParamsHandle, kernelMode);
BOUND_S2_KERNEL_END()

FREE_S2_KERNEL_3D_BEG(fieldSamplerGridKernel,
	((fieldsampler::FieldSamplerKernelParams, baseParams))
	((fieldsampler::FieldSamplerGridKernelParams, gridParams))
	((InplaceHandle<fieldsampler::FieldSamplerParamsExArray>, paramsExArrayHandle))
	((InplaceHandle<fieldsampler::FieldSamplerQueryParams>, queryParamsHandle))
	((fieldsampler::FieldSamplerKernelMode::Enum, kernelMode))
)
	fieldSamplerGridFunc INPLACE_TEMPL_ARGS_VAL (idxX, idxY, idxZ, baseParams, gridParams, paramsExArrayHandle, queryParamsHandle, kernelMode);
FREE_S2_KERNEL_3D_END()


#endif //FIELD_SAMPLER_SEPARATE_KERNELS
