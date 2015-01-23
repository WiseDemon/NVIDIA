/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FIELD_SAMPLER_COMMON_H__
#define __FIELD_SAMPLER_COMMON_H__

#include "foundation/PxVec3.h"
#include "foundation/PxVec4.h"
#include <PxMat34Legacy.h>

#include <NiFieldSampler.h>
#include <NiFieldBoundary.h>

#if defined(APEX_CUDA_SUPPORT)
#pragma warning(push)
#pragma warning(disable:4201)
#pragma warning(disable:4408)

#include <vector_types.h>

#pragma warning(pop)
#endif

#define FIELD_SAMPLER_MULTIPLIER_VALUE 1
#define FIELD_SAMPLER_MULTIPLIER_WEIGHT 2
//0, FIELD_SAMPLER_MULTIPLIER_VALUE or FIELD_SAMPLER_MULTIPLIER_WEIGHT
#define FIELD_SAMPLER_MULTIPLIER FIELD_SAMPLER_MULTIPLIER_WEIGHT

namespace physx
{
namespace apex
{
namespace fieldsampler
{

#define VELOCITY_WEIGHT_THRESHOLD 0.00001f

struct FieldSamplerExecuteArgs
{
	physx::PxVec3			position;
	physx::PxF32			mass;
	physx::PxVec3			velocity;

	physx::PxF32			elapsedTime;
	physx::PxU32			totalElapsedMS;
};

//struct FieldShapeParams
#define INPLACE_TYPE_STRUCT_NAME FieldShapeParams
#define INPLACE_TYPE_STRUCT_BASE NiFieldShapeDesc
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxF32,	fade)
#include INPLACE_TYPE_BUILD()


//struct FieldShapeGroupParams
#define INPLACE_TYPE_STRUCT_NAME FieldShapeGroupParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(InplaceArray<FieldShapeParams>,	shapeArray)
#include INPLACE_TYPE_BUILD()


//struct FieldSamplerParams
#define INPLACE_TYPE_STRUCT_NAME FieldSamplerParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxU32,										executeType) \
	INPLACE_TYPE_FIELD(InplaceHandleBase,									executeParamsHandle) \
	INPLACE_TYPE_FIELD(InplaceEnum<NiFieldSamplerType::Enum>,				type) \
	INPLACE_TYPE_FIELD(InplaceEnum<NiFieldSamplerGridSupportType::Enum>,	gridSupportType) \
	INPLACE_TYPE_FIELD(physx::PxF32,										dragCoeff) \
	INPLACE_TYPE_FIELD(FieldShapeParams,									includeShape) \
	INPLACE_TYPE_FIELD(InplaceArray<InplaceHandle<FieldShapeGroupParams> >,	excludeShapeGroupHandleArray)
#include INPLACE_TYPE_BUILD()


//struct FieldSamplerQueryParams
#define INPLACE_TYPE_STRUCT_NAME FieldSamplerQueryParams
#include INPLACE_TYPE_BUILD()


//struct FieldSamplerParams
#define INPLACE_TYPE_STRUCT_NAME FieldSamplerParamsEx
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(InplaceHandle<FieldSamplerParams>,	paramsHandle) \
	INPLACE_TYPE_FIELD(physx::PxF32,						multiplier)
#include INPLACE_TYPE_BUILD()

typedef InplaceArray< FieldSamplerParamsEx, false > FieldSamplerParamsExArray;


#if defined(APEX_CUDA_SUPPORT) || defined(__CUDACC__)

struct FieldSamplerKernelType
{
	enum Enum
	{
		POINTS,
		GRID
	};
};

struct FieldSamplerKernelParams
{
	physx::PxF32	elapsedTime;
	physx::PxVec3	cellSize;
	physx::PxU32	totalElapsedMS;
};

struct FieldSamplerGridKernelParams
{
	physx::PxU32 numX, numY, numZ;

	physx::PxMat34Legacy gridToWorld;

	physx::PxF32 mass;
};


struct FieldSamplerKernelArgs : FieldSamplerKernelParams
{
};

struct FieldSamplerPointsKernelArgs : FieldSamplerKernelArgs
{
	float4* accumField;
	float4* accumVelocity;
	const float4* positionMass;
	const float4* velocity;
};

struct FieldSamplerGridKernelArgs : FieldSamplerKernelArgs, FieldSamplerGridKernelParams
{
};

struct FieldSamplerKernelMode
{
	enum Enum
	{
		DEFAULT = 0,
		FINISH_PRIMARY = 1,
		FINISH_SECONDARY = 2
	};
};

#endif

#if defined(APEX_CUDA_SUPPORT)

class FieldSamplerWrapper;

struct FieldSamplerInfo
{
	FieldSamplerWrapper*	mFieldSamplerWrapper;
	physx::PxF32			mMultiplier;
};

struct NiFieldSamplerKernelLaunchData
{
	CUstream                                        stream;
	FieldSamplerKernelType::Enum                    kernelType;
	const FieldSamplerKernelArgs*                   kernelArgs;
	InplaceHandle<FieldSamplerQueryParams>          queryParamsHandle;
	InplaceHandle<FieldSamplerParamsExArray>        paramsExArrayHandle;
	const physx::Array<FieldSamplerInfo>*           fieldSamplerArray;
	physx::PxU32                                    activeFieldSamplerCount;
	FieldSamplerKernelMode::Enum                    kernelMode;
};

struct NiFieldSamplerPointsKernelLaunchData : NiFieldSamplerKernelLaunchData
{
	physx::PxU32                                    threadCount;
	physx::PxU32                                    memRefSize;
};

struct NiFieldSamplerGridKernelLaunchData : NiFieldSamplerKernelLaunchData
{
	physx::PxU32									threadCountX;
	physx::PxU32									threadCountY;
	physx::PxU32									threadCountZ;
	ApexCudaArray*									accumArray;
};

#endif

APEX_CUDA_CALLABLE PX_INLINE physx::PxF32 evalFade(physx::PxF32 dist, physx::PxF32 fade)
{
	physx::PxF32 x = (1 - dist) / (fade + 1e-5f);
	return PxClamp<physx::PxF32>(x, 0, 1);
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxF32 evalFadeAntialiasing(physx::PxF32 dist, physx::PxF32 fade, physx::PxF32 cellRadius)
{
	const physx::PxF32 f = fade;
	const physx::PxF32 r = cellRadius;
	const physx::PxF32 x = dist - 1.0f;

	physx::PxF32 res = 0.0f;
	//linear part
	//if (x - r < -f)
	{
		const physx::PxF32 a = physx::PxMin(x - r, -f);
		const physx::PxF32 b = physx::PxMin(x + r, -f);

		res += (b - a);
	}
	//quadratic part
	if (f >= 1e-5f)
	{
		//if (x - r < 0.0f && x + r > -f)
		{
			const physx::PxF32 a = physx::PxClamp(x - r, -f, 0.0f);
			const physx::PxF32 b = physx::PxClamp(x + r, -f, 0.0f);

			res += (a*a - b*b) / (2 * f);
		}
	}
	return res / (2 * r);
}


APEX_CUDA_CALLABLE PX_INLINE physx::PxF32 evalDistInShapeNONE(const NiFieldShapeDesc& /*shapeParams*/, const physx::PxVec3& /*worldPos*/)
{
	return 0.0f; //always inside
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxF32 evalDistInShapeSPHERE(const NiFieldShapeDesc& shapeParams, const physx::PxVec3& worldPos)
{
	const physx::PxVec3 shapePos = shapeParams.worldToShape * worldPos;
	const physx::PxF32 radius = shapeParams.dimensions.x;
	return shapePos.magnitude() / radius;
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxF32 evalDistInShapeBOX(const NiFieldShapeDesc& shapeParams, const physx::PxVec3& worldPos)
{
	const physx::PxVec3 shapePos = shapeParams.worldToShape * worldPos;
	const physx::PxVec3& halfSize = shapeParams.dimensions;
	physx::PxVec3 unitPos(shapePos.x / halfSize.x, shapePos.y / halfSize.y, shapePos.z / halfSize.z);
	return physx::PxVec3(physx::PxAbs(unitPos.x), physx::PxAbs(unitPos.y), physx::PxAbs(unitPos.z)).maxElement();
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxF32 evalDistInShapeCAPSULE(const NiFieldShapeDesc& shapeParams, const physx::PxVec3& worldPos)
{
	const physx::PxVec3 shapePos = shapeParams.worldToShape * worldPos;
	const physx::PxF32 radius = shapeParams.dimensions.x;
	const physx::PxF32 halfHeight = shapeParams.dimensions.y * 0.5f;

	physx::PxVec3 clampPos = shapePos;
	clampPos.y -= physx::PxClamp(shapePos.y, -halfHeight, +halfHeight);

	return clampPos.magnitude() / radius;
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxF32 evalDistInShape(const NiFieldShapeDesc& shapeParams, const physx::PxVec3& worldPos)
{
	switch (shapeParams.type)
	{
	case NiFieldShapeType::NONE:
		return evalDistInShapeNONE(shapeParams, worldPos);
	case NiFieldShapeType::SPHERE:
		return evalDistInShapeSPHERE(shapeParams, worldPos);
	case NiFieldShapeType::BOX:
		return evalDistInShapeBOX(shapeParams, worldPos);
	case NiFieldShapeType::CAPSULE:
		return evalDistInShapeCAPSULE(shapeParams, worldPos);
	default:
		return 1.0f; //always outside
	};
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxVec3 scaleToShape(const NiFieldShapeDesc& shapeParams, const physx::PxVec3& worldVec)
{
	switch (shapeParams.type)
	{
	case NiFieldShapeType::SPHERE:
	case NiFieldShapeType::CAPSULE:
	{
		const physx::PxF32 radius = shapeParams.dimensions.x;
		return physx::PxVec3(worldVec.x / radius, worldVec.y / radius, worldVec.z / radius);
	}
	case NiFieldShapeType::BOX:
	{
		const physx::PxVec3& halfSize = shapeParams.dimensions;
		return physx::PxVec3(worldVec.x / halfSize.x, worldVec.y / halfSize.y, worldVec.z / halfSize.z);
	}
	default:
		return worldVec;
	};
}


APEX_CUDA_CALLABLE PX_INLINE physx::PxF32 evalWeightInShape(const FieldShapeParams& shapeParams, const physx::PxVec3& position)
{
	physx::PxF32 dist = physx::apex::fieldsampler::evalDistInShape(shapeParams, position);
	return physx::apex::fieldsampler::evalFade(dist, shapeParams.fade) * shapeParams.weight;
}

APEX_CUDA_CALLABLE PX_INLINE void accumFORCE(const FieldSamplerExecuteArgs& args,
	const physx::PxVec3& field, physx::PxF32 fieldW,
	physx::PxVec4& accumAccel, physx::PxVec4& accumVelocity)
{
	PX_UNUSED(accumVelocity);

	physx::PxVec3 newAccel = ((1 - accumAccel.w) * fieldW * args.elapsedTime / args.mass) * field;
	accumAccel.x += newAccel.x;
	accumAccel.y += newAccel.y;
	accumAccel.z += newAccel.z;
}

APEX_CUDA_CALLABLE PX_INLINE void accumACCELERATION(const FieldSamplerExecuteArgs& args,
	const physx::PxVec3& field, physx::PxF32 fieldW,
	physx::PxVec4& accumAccel, physx::PxVec4& accumVelocity)
{
	PX_UNUSED(accumVelocity);

	physx::PxVec3 newAccel = ((1 - accumAccel.w) * fieldW * args.elapsedTime) * field;
	accumAccel.x += newAccel.x;
	accumAccel.y += newAccel.y;
	accumAccel.z += newAccel.z;
}

APEX_CUDA_CALLABLE PX_INLINE void accumVELOCITY_DIRECT(const FieldSamplerExecuteArgs& args,
	const physx::PxVec3& field, physx::PxF32 fieldW,
	physx::PxVec4& accumAccel, physx::PxVec4& accumVelocity)
{
	PX_UNUSED(args);

	physx::PxVec3 newVelocity = ((1 - accumAccel.w) * fieldW) * field;
	accumVelocity.x += newVelocity.x;
	accumVelocity.y += newVelocity.y;
	accumVelocity.z += newVelocity.z;
	accumVelocity.w = physx::PxMax(accumVelocity.w, fieldW);
}

APEX_CUDA_CALLABLE PX_INLINE void accumVELOCITY_DRAG(const FieldSamplerExecuteArgs& args, physx::PxF32 dragCoeff,
	const physx::PxVec3& field, physx::PxF32 fieldW,
	physx::PxVec4& accumAccel, physx::PxVec4& accumVelocity)
{
#if 1
	const physx::PxF32 dragFieldW = physx::PxMin(fieldW * dragCoeff * args.elapsedTime / args.mass, 1.0f);
	accumVELOCITY_DIRECT(args, field, dragFieldW, accumAccel, accumVelocity);
#else
	const physx::PxVec3 dragForce = (field - args.velocity) * dragCoeff;
	accumFORCE(args, dragForce, fieldW, accumAccel, accumVelocity);
#endif
}


}
} // namespace apex
}
#ifdef __CUDACC__

#ifdef APEX_TEST
struct PxInternalParticleFlagGpu
{
	enum Enum
	{
		//reserved	(1<<0),
		//reserved	(1<<1),
		//reserved	(1<<2),
		//reserved	(1<<3),
		//reserved	(1<<4),
		//reserved	(1<<5),
		eCUDA_NOTIFY_CREATE					= (1 << 6),
		eCUDA_NOTIFY_SET_POSITION			= (1 << 7),
	};
};
struct PxParticleFlag
{
	enum Enum
	{
		eVALID								= (1 << 0),
		eCOLLISION_WITH_STATIC				= (1 << 1),
		eCOLLISION_WITH_DYNAMIC				= (1 << 2),
		eCOLLISION_WITH_DRAIN				= (1 << 3),
		eSPATIAL_DATA_STRUCTURE_OVERFLOW	= (1 << 4),
	};
};
struct PxParticleFlagGpu
{
	physx::PxU16 api;	// PxParticleFlag
	physx::PxU16 low;	// PxInternalParticleFlagGpu
};

APEX_CUDA_CALLABLE PX_INLINE void testParticle(physx::PxVec4& position, physx::PxVec4& velocity, physx::PxU32& flag, 
											 const physx::PxVec4& initPosition, const physx::PxVec4& initVelocity)
{
	position = initPosition;
	velocity = initVelocity;
	
	PxParticleFlagGpu& f = ((PxParticleFlagGpu&) flag);
	f.api = PxParticleFlag::eVALID;
	f.low = PxInternalParticleFlagGpu::eCUDA_NOTIFY_CREATE;
}
#endif


template <int queryType>
struct FieldSamplerExecutor;
/*
{
	INPLACE_TEMPL_ARGS_DEF
	static inline __device__ physx::PxVec3 func(const physx::apex::fieldsampler::FieldSamplerParams* params, const physx::apex::fieldsampler::FieldSamplerExecuteArgs& args, physx::PxF32& fieldWeight);
};
*/

template <int queryType>
struct FieldSamplerIncludeWeightEvaluator
{
	INPLACE_TEMPL_ARGS_DEF
	static inline __device__ physx::PxF32 func(const physx::apex::fieldsampler::FieldSamplerParams* params, const physx::PxVec3& position, const physx::PxVec3& cellSize)
	{
		return physx::apex::fieldsampler::evalWeightInShape(params->includeShape, position);
	}
};

#endif

#endif
