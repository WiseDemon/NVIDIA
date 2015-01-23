/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FORCEFIELD_FS_COMMON_SRC_H__
#define __FORCEFIELD_FS_COMMON_SRC_H__

#include "../../fieldsampler/include/FieldSamplerCommon.h"
#include "SimplexNoise.h"
#include "TableLookup.h"
#include "PxMat33.h"

namespace physx
{
namespace apex
{

template <> struct InplaceTypeTraits<TableLookup>
{
	template <int _inplace_offset_, typename R, typename RA>
	APEX_CUDA_CALLABLE PX_INLINE static void reflectType(R& r, RA ra, TableLookup& t)
	{
		InplaceTypeHelper::reflectType<APEX_OFFSETOF(TableLookup, xVals)      + _inplace_offset_>(r, ra, t.xVals,      InplaceTypeMemberDefaultTraits());
		InplaceTypeHelper::reflectType<APEX_OFFSETOF(TableLookup, yVals)      + _inplace_offset_>(r, ra, t.yVals,      InplaceTypeMemberDefaultTraits());
		InplaceTypeHelper::reflectType<APEX_OFFSETOF(TableLookup, x1)         + _inplace_offset_>(r, ra, t.x1,         InplaceTypeMemberDefaultTraits());
		InplaceTypeHelper::reflectType<APEX_OFFSETOF(TableLookup, x2)         + _inplace_offset_>(r, ra, t.x2,         InplaceTypeMemberDefaultTraits());
		InplaceTypeHelper::reflectType<APEX_OFFSETOF(TableLookup, multiplier) + _inplace_offset_>(r, ra, t.multiplier, InplaceTypeMemberDefaultTraits());
	}
};

namespace forcefield
{

struct ForceFieldShapeType
{
	enum Enum
	{
		SPHERE = 0,
		CAPSULE,
		CYLINDER,
		CONE,
		BOX,
		NONE,
	};
};

struct ForceFieldFalloffType
{
	enum Enum
	{
		LINEAR = 0,
		STEEP,
		SCURVE,
		CUSTOM,
		NONE,
	};
};

struct ForceFieldCoordinateSystemType
{
	enum Enum
	{
		CARTESIAN = 0,
		SPHERICAL,
		CYLINDRICAL,
		TOROIDAL,
	};
};

struct ForceFieldKernelType
{
	enum Enum
	{
		RADIAL = 0,
		GENERIC
	};
};

//struct ForceFieldShapeDesc
#define INPLACE_TYPE_STRUCT_NAME ForceFieldShapeDesc
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(InplaceEnum<ForceFieldShapeType::Enum>,	type) \
	INPLACE_TYPE_FIELD(physx::PxMat44,							forceFieldToShape) \
	INPLACE_TYPE_FIELD(physx::PxVec3,							dimensions)
#include INPLACE_TYPE_BUILD()


//struct NoiseParams
#define INPLACE_TYPE_STRUCT_NAME NoiseParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxF32,	strength) \
	INPLACE_TYPE_FIELD(physx::PxF32,	spaceScale) \
	INPLACE_TYPE_FIELD(physx::PxF32,	timeScale) \
	INPLACE_TYPE_FIELD(physx::PxU32,	octaves)
#include INPLACE_TYPE_BUILD()

//struct ForceFieldCoordinateSystem
#define INPLACE_TYPE_STRUCT_NAME ForceFieldCoordinateSystem
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(InplaceEnum<ForceFieldCoordinateSystemType::Enum>,	type) \
	INPLACE_TYPE_FIELD(physx::PxF32,										torusRadius)
#include INPLACE_TYPE_BUILD()

//struct ForceFieldFSKernelParams
#define INPLACE_TYPE_STRUCT_NAME ForceFieldFSKernelParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxMat44,		pose) \
	INPLACE_TYPE_FIELD(physx::PxF32,		strength) \
	INPLACE_TYPE_FIELD(ForceFieldShapeDesc,	includeShape)
#include INPLACE_TYPE_BUILD()

//struct GenericForceFieldFSKernelParams
#define INPLACE_TYPE_STRUCT_NAME GenericForceFieldFSKernelParams
#define INPLACE_TYPE_STRUCT_BASE ForceFieldFSKernelParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(ForceFieldCoordinateSystem,		cs) \
	INPLACE_TYPE_FIELD(physx::PxVec3,					constant) \
	INPLACE_TYPE_FIELD(physx::PxMat33,					positionMultiplier) \
	INPLACE_TYPE_FIELD(physx::PxVec3,					positionTarget) \
	INPLACE_TYPE_FIELD(physx::PxMat33,					velocityMultiplier) \
	INPLACE_TYPE_FIELD(physx::PxVec3,					velocityTarget) \
	INPLACE_TYPE_FIELD(physx::PxVec3,					noise) \
	INPLACE_TYPE_FIELD(physx::PxVec3,					falloffLinear) \
	INPLACE_TYPE_FIELD(physx::PxVec3,					falloffQuadratic)
#include INPLACE_TYPE_BUILD()

//struct RadialForceFieldFSKernelParams
#define INPLACE_TYPE_STRUCT_NAME RadialForceFieldFSKernelParams
#define INPLACE_TYPE_STRUCT_BASE ForceFieldFSKernelParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxF32,		radius) \
	INPLACE_TYPE_FIELD(TableLookup,			falloffTable) \
	INPLACE_TYPE_FIELD(NoiseParams,			noiseParams)
#include INPLACE_TYPE_BUILD()

APEX_CUDA_CALLABLE PX_INLINE bool isPosInShape(const ForceFieldShapeDesc& shapeParams, const physx::PxVec3& pos)
{
	// Sphere: x = radius
	// Capsule: x = radius, y = height
	// Cylinder: x = radius, y = height
	// Cone: x = top radius, y = height, z = bottom radius
	// Box: x,y,z = half-dimensions

	// transform position from force field coordinates to local shape coordinates
	physx::PxVec3 shapePos = shapeParams.forceFieldToShape.transform(pos);

	switch (shapeParams.type)
	{
	case ForceFieldShapeType::SPHERE:
		{
			return (shapePos.magnitude() <= shapeParams.dimensions.x);
		}
	case ForceFieldShapeType::CAPSULE:
		{
			physx::PxF32 halfHeight = shapeParams.dimensions.y / 2.0f;

			// check if y-position is within height of cylinder
			if (shapePos.y >= -halfHeight && shapePos.y <= halfHeight)
			{
				// check if x and z positions is inside radius height of cylinder
				if (physx::PxSqrt(shapePos.x * shapePos.x + shapePos.z * shapePos.z) <= shapeParams.dimensions.x)
				{
					return true;
				}
			}

			// check if position falls inside top sphere in capsule
			physx::PxVec3 spherePos = shapePos - physx::PxVec3(0, halfHeight, 0);
			if (spherePos.magnitude() <= shapeParams.dimensions.x)
			{
				return true;
			}

			// check if position falls inside bottom sphere in capsule
			spherePos = shapePos + physx::PxVec3(0, halfHeight, 0);
			if (spherePos.magnitude() <= shapeParams.dimensions.x)
			{
				return true;
			}

			return false;
		}
	case ForceFieldShapeType::CYLINDER:
		{
			physx::PxF32 halfHeight = shapeParams.dimensions.y / 2.0f;

			// check if y-position is within height of cylinder
			if (shapePos.y >= -halfHeight && shapePos.y <= halfHeight)
			{
				// check if x and z positions is inside radius height of cylinder
				if (physx::PxSqrt(shapePos.x * shapePos.x + shapePos.z * shapePos.z) <= shapeParams.dimensions.x)
				{
					return true;
				}
			}
			return false;
		}
	case ForceFieldShapeType::CONE:
		{
			physx::PxF32 halfHeight = shapeParams.dimensions.y / 2.0f;

			// check if y-position is within height of cone
			if (shapePos.y >= -halfHeight && shapePos.y <= halfHeight)
			{
				// cone can be normal or inverted
				physx::PxF32 smallerBase;
				physx::PxF32 heightFromSmallerBase;
				physx::PxF32 radiusDiff;
				if (shapeParams.dimensions.x > shapeParams.dimensions.z)
				{
					smallerBase = shapeParams.dimensions.z;
					heightFromSmallerBase = shapePos.y + halfHeight;
					radiusDiff = shapeParams.dimensions.x - shapeParams.dimensions.z;
				}
				else
				{
					smallerBase = shapeParams.dimensions.x;
					heightFromSmallerBase = halfHeight - shapePos.y;
					radiusDiff = shapeParams.dimensions.z - shapeParams.dimensions.x;
				}

				// compute radius at y-position along height of cone
				physx::PxF32 radiusAlongCone = smallerBase + (heightFromSmallerBase / shapeParams.dimensions.y) * radiusDiff;

				// check if x and z positions is inside radius at a specific height of cone
				if (physx::PxSqrt(shapePos.x * shapePos.x + shapePos.z * shapePos.z) <= radiusAlongCone)
				{
					return true;
				}
			}
			return false;
		}
	case ForceFieldShapeType::BOX:
		{
			return (physx::PxAbs(shapePos.x) <= shapeParams.dimensions.x && 
					physx::PxAbs(shapePos.y) <= shapeParams.dimensions.y &&
					physx::PxAbs(shapePos.z) <= shapeParams.dimensions.z);
		}
	default:
		{
			return false;
		}
	}
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxVec3 getNoise(const NoiseParams& params, const physx::PxVec3& pos, const physx::PxU32& totalElapsedMS)
{
	PxVec3 point = params.spaceScale * pos;
	PxF32 time = (params.timeScale * 1e-3f) * totalElapsedMS;

	PxVec4 dFx;
	dFx.setZero();
	PxVec4 dFy;
	dFy.setZero();
	PxVec4 dFz;
	dFz.setZero();
	int seed = 0;
	PxF32 amp = 1.0f;
	for (PxU32 i = 0; i < params.octaves; ++i)
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

	return params.strength * rot;
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxVec3 executeForceFieldMainFS(const RadialForceFieldFSKernelParams& params, const physx::PxVec3& pos, const physx::PxU32& totalElapsedMS)
{
	// bring pos to force field coordinate system
	physx::PxVec3 localPos = params.pose.inverseRT().transform(pos);

	if (isPosInShape(params.includeShape, localPos))
	{
		PxVec3 result = localPos.getNormalized();
		result = result * params.strength;

		// apply falloff
		result = result * params.falloffTable.lookupTableValue(localPos.magnitude() / params.radius);

		// apply noise
		result = result + getNoise(params.noiseParams, localPos, totalElapsedMS);

		// rotate result back to world coordinate system
		return params.pose.rotate(result);
	}

	return physx::PxVec3(0, 0, 0);
}

// function to compute the Scalar falloff for the cylinderical, toroidal, cartesian and generic
// force fields so that the falloff does not make the direction of the force vector change.
APEX_CUDA_CALLABLE PX_INLINE float falloff(physx::PxVec3 val, physx::PxVec3 linear, physx::PxVec3 quadratic)
{
	float magnitude = val.magnitude();
	float v = (linear.x * magnitude + quadratic.x * magnitude * magnitude) +
			  (linear.y * magnitude + quadratic.y * magnitude * magnitude) +
			  (linear.z * magnitude + quadratic.z * magnitude * magnitude);
	v += 1.0f;
	//PX_ASSERT(v>0);
	return 1.0f/v;
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxVec3 genericEvalLinearKernel(const physx::PxVec3& localPos, const physx::PxVec3& localVel, const GenericForceFieldFSKernelParams& params)
{
	const bool useFalloff = params.falloffLinear.magnitudeSquared() +
							params.falloffQuadratic.magnitudeSquared() != 0;

	const bool useVelMul = !(params.velocityMultiplier.column0.isZero() && 
							 params.velocityMultiplier.column1.isZero() && 
							 params.velocityMultiplier.column2.isZero()); 

	const bool usePosMul = !(params.positionMultiplier.column0.isZero() &&
							 params.positionMultiplier.column1.isZero() &&
							 params.positionMultiplier.column2.isZero());
						

	physx::PxVec3 ffPosErr;
	physx::PxMat33 tangentFrame;
	physx::PxVec3 ffForce = params.constant;
	switch(params.cs.type)
	{
	case ForceFieldCoordinateSystemType::CARTESIAN:
		{
			ffPosErr = params.positionTarget - (localPos);
		} 
		break;
	case ForceFieldCoordinateSystemType::SPHERICAL:
		{
			const float& r = localPos.x;
			ffPosErr = physx::PxVec3(params.positionTarget.x - r, 0, 0);
		} 
		break;
	case ForceFieldCoordinateSystemType::CYLINDRICAL:
		{
			const float& r = localPos.x;
			ffPosErr = physx::PxVec3(params.positionTarget.x - r, params.positionTarget.y - localPos.y, 0);
		} 
		break;
	case ForceFieldCoordinateSystemType::TOROIDAL:
		{
			float r = 0.0f;
			physx::PxVec3 t0;
			physx::PxVec3 circleDir(localPos.x, 0.0f, localPos.z);
			float l2 = circleDir.magnitudeSquared();
			if(l2<1e-4f*1e-4f)
			{
				circleDir = physx::PxVec3(0);
			}
			else
			{
				circleDir /= physx::PxSqrt(l2);	// Normalize circleDir
				physx::PxVec3 circlePoint = circleDir * params.cs.torusRadius;
				t0 = localPos - circlePoint;
				l2 = t0.magnitudeSquared();
				if(l2<1e-4f*1e-4f)
				{
					circleDir = physx::PxVec3(0);
					t0 = physx::PxVec3(0);
				}
				else
				{
					r = physx::PxSqrt(l2);
					t0 /= r;	// Normalize t0
				}
			}
			physx::PxVec3 t1(-circleDir.z, 0.0f, circleDir.x);		// physx::PxVec3 t1 = circleDir.cross(physx::PxVec3(0,1,0));
			tangentFrame.column0 = t0;
			tangentFrame.column1 = t1;
			tangentFrame.column2 = t0.cross(t1);
			ffPosErr = physx::PxVec3(params.positionTarget.x - r, 0, 0);
		}
		break;
	}
			
	physx::PxVec3 ffVel = (params.cs.type == ForceFieldCoordinateSystemType::TOROIDAL) ? tangentFrame.getInverse() * localVel : localVel; 

	if(useVelMul)
	{
		ffForce += params.velocityMultiplier * (params.velocityTarget - ffVel);
	}

	if(usePosMul)
	{
		ffForce += params.positionMultiplier * ffPosErr;
	}

	//TODO, enable noise, with existing noise functionality
	//applyNoise(ffForce, params.noise, NpPhysicsSDK::instance->getNpPhysicsTls()->mNpLinearKernelRnd);
	//ffForce *= PxVec3(1) + params.noise * [-1, 1]^3

	if(useFalloff)
	{
		ffForce *= falloff(ffPosErr, params.falloffLinear, params.falloffQuadratic);
	}

	physx::PxVec3 force = (params.cs.type == ForceFieldCoordinateSystemType::TOROIDAL) ? tangentFrame * ffForce : ffForce;
	return force;
}

APEX_CUDA_CALLABLE PX_INLINE void genericEvalCartesian(physx::PxVec3& force, const physx::PxVec3& localPos, const physx::PxVec3& localVel, const GenericForceFieldFSKernelParams& params)
{
	// compute tangent frame
	// -- none here

	// evaluate kernel
	force = genericEvalLinearKernel(localPos, localVel, params);
}

APEX_CUDA_CALLABLE PX_INLINE void genericEvalSpherical(physx::PxVec3& force, const physx::PxVec3& localPos, const physx::PxVec3& localVel, const GenericForceFieldFSKernelParams& params)
{
	// compute tangent frame
	physx::PxMat33 tangentFrame(physx::PxZero);
	float l2 = localPos.magnitudeSquared();
	physx::PxVec3 tangentPos;
	if(l2>1e-4f*1e-4f)
	{
		float r = physx::PxSqrt(l2);
		physx::PxVec3 t0 = localPos/r;
		tangentFrame.column0 = t0;
		tangentPos = physx::PxVec3(r, 0, 0);
	}
	else
	{
		tangentPos = physx::PxVec3(0);
	}

	// compute tangent frame velocity
	const physx::PxVec3 tangentVel = tangentFrame.getInverse() * localVel;

	// evaluate kernel
	force = genericEvalLinearKernel(tangentPos, tangentVel, params);

	// transform back to local space
	force = tangentFrame * force;
}

APEX_CUDA_CALLABLE PX_INLINE void genericEvalCylindrical(physx::PxVec3& force, const physx::PxVec3& localPos, const physx::PxVec3& localVel, const GenericForceFieldFSKernelParams& params)
{
	// compute tangent frame
	physx::PxMat33 tangentFrame;
	physx::PxVec3 t0(localPos.x, 0, localPos.z); // Project to Cylindrical
	const float len = t0.magnitude();
	physx::PxVec3 tangentPos;
	if(len > 1e-4f)
	{
		t0 /= len;
		tangentPos = physx::PxVec3(len, localPos.y, 0);
	}
	else
	{
		t0 = physx::PxVec3(0);
		tangentPos = physx::PxVec3(0, localPos.y, 0);
	}
	tangentFrame.column0 = t0;
	tangentFrame.column1 = PxVec3(0.0f, 1.0f, 0.0f);	// t1
	tangentFrame.column2 = PxVec3(-t0.z, 0.0f, t0.x);	// t0.cross(t1)

	// compute tangent frame velocity
	const physx::PxVec3 tangentVel = tangentFrame.getInverse() * localVel;

	// evaluate kernel
	force = genericEvalLinearKernel(tangentPos, tangentVel, params);

	// transform back to local space
	force = tangentFrame * force;
}

APEX_CUDA_CALLABLE PX_INLINE void genericEvalToroidal(physx::PxVec3& force, const physx::PxVec3& localPos, const physx::PxVec3& localVel, const GenericForceFieldFSKernelParams& params)
{
	// evaluate kernel
	force = genericEvalLinearKernel(localPos, localVel, params);
}

/**
This is more or less a 1:1 copy of PhysX 2.8.4/UE3 generic force fields. 
Since 2.8.4 supported more flexible callbacks, we should probably optimize 
this quite a bit. 
*/
APEX_CUDA_CALLABLE PX_INLINE physx::PxVec3 executeForceFieldMainFS(const GenericForceFieldFSKernelParams& params, const physx::PxVec3& pos, const physx::PxVec3& vel, const physx::PxU32&)
{
	// bring pos to force field coordinate system
	physx::PxVec3 localPos = params.pose.inverseRT().transform(pos);

	if (isPosInShape(params.includeShape, localPos))
	{
		physx::PxVec3 localVel = params.pose.inverseRT().rotate(vel); 

		physx::PxVec3 result(0);
		switch (params.cs.type)
		{
		case ForceFieldCoordinateSystemType::CARTESIAN:
			genericEvalCartesian(result, localPos, localVel, params);
			break;
		case ForceFieldCoordinateSystemType::SPHERICAL:
			genericEvalSpherical(result, localPos, localVel, params);
			break;
		case ForceFieldCoordinateSystemType::CYLINDRICAL:
			genericEvalCylindrical(result, localPos, localVel, params);
			break;
		case ForceFieldCoordinateSystemType::TOROIDAL:
			genericEvalToroidal(result, localPos, localVel, params);
			break;
		default:
			break;
		}

		// apply strength
		result = result * params.strength;

		// rotate result back to world coordinate system
		return params.pose.rotate(result);
	}

	return physx::PxVec3(0, 0, 0);
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxVec3 executeForceFieldFS(const RadialForceFieldFSKernelParams& params, const physx::PxVec3& pos, const physx::PxU32& totalElapsedMS)
{
	physx::PxVec3 resultField(0, 0, 0);
	resultField += executeForceFieldMainFS(params, pos, totalElapsedMS);
	return resultField;
}

APEX_CUDA_CALLABLE PX_INLINE physx::PxVec3 executeForceFieldFS(const GenericForceFieldFSKernelParams& params, const physx::PxVec3& pos, const physx::PxVec3& vel, const physx::PxU32& totalElapsedMS)
{
	physx::PxVec3 resultField(0, 0, 0);
	resultField += executeForceFieldMainFS(params, pos, vel, totalElapsedMS);
	return resultField;
}

}
}
} // end namespace physx::apex

#endif
