/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __BASIC_IOS_COMMON_H__
#define __BASIC_IOS_COMMON_H__

#include "PxMat34Legacy.h"
#include "foundation/PxBounds3.h"
#include "foundation/PxVec3.h"
#include "InplaceTypes.h"

namespace physx
{
	namespace apex
	{
		class ApexCpuInplaceStorage;

		namespace basicios
		{

//struct InjectorParams
#define INPLACE_TYPE_STRUCT_NAME InjectorParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxF32,	mLODMaxDistance) \
	INPLACE_TYPE_FIELD(physx::PxF32,	mLODDistanceWeight) \
	INPLACE_TYPE_FIELD(physx::PxF32,	mLODSpeedWeight) \
	INPLACE_TYPE_FIELD(physx::PxF32,	mLODLifeWeight) \
	INPLACE_TYPE_FIELD(physx::PxF32,	mLODBias) \
	INPLACE_TYPE_FIELD(physx::PxU32,	mLocalIndex)
#include INPLACE_TYPE_BUILD()

			typedef InplaceArray<InjectorParams> InjectorParamsArray;

//struct CollisionData
#define INPLACE_TYPE_STRUCT_NAME CollisionData
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxVec3,	bodyCMassPosition) \
	INPLACE_TYPE_FIELD(physx::PxVec3,	bodyLinearVelocity) \
	INPLACE_TYPE_FIELD(physx::PxVec3,	bodyAngluarVelocity) \
	INPLACE_TYPE_FIELD(physx::PxF32,	materialRestitution)
#include INPLACE_TYPE_BUILD()

//struct CollisionSphereData
#define INPLACE_TYPE_STRUCT_NAME CollisionSphereData
#define INPLACE_TYPE_STRUCT_BASE CollisionData
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxBounds3,		aabb) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,	pose) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,	inversePose) \
	INPLACE_TYPE_FIELD(physx::PxF32,			radius)
#include INPLACE_TYPE_BUILD()

//struct CollisionCapsuleData
#define INPLACE_TYPE_STRUCT_NAME CollisionCapsuleData
#define INPLACE_TYPE_STRUCT_BASE CollisionData
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxBounds3,		aabb) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,	pose) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,	inversePose) \
	INPLACE_TYPE_FIELD(physx::PxF32,			halfHeight) \
	INPLACE_TYPE_FIELD(physx::PxF32,			radius)
#include INPLACE_TYPE_BUILD()

//struct CollisionBoxData
#define INPLACE_TYPE_STRUCT_NAME CollisionBoxData
#define INPLACE_TYPE_STRUCT_BASE CollisionData
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxBounds3,		aabb) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,	pose) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,	inversePose) \
	INPLACE_TYPE_FIELD(physx::PxVec3,			halfSize)
#include INPLACE_TYPE_BUILD()

//struct CollisionHalfSpaceData
#define INPLACE_TYPE_STRUCT_NAME CollisionHalfSpaceData
#define INPLACE_TYPE_STRUCT_BASE CollisionData
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxVec3,	normal) \
	INPLACE_TYPE_FIELD(physx::PxVec3,	origin)
#include INPLACE_TYPE_BUILD()

//struct CollisionConvexMeshData
#define INPLACE_TYPE_STRUCT_NAME CollisionConvexMeshData
#define INPLACE_TYPE_STRUCT_BASE CollisionData
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxBounds3,		aabb) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,	pose) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,	inversePose) \
	INPLACE_TYPE_FIELD(physx::PxU32,			numPolygons) \
	INPLACE_TYPE_FIELD(physx::PxU32,			firstPlane) \
	INPLACE_TYPE_FIELD(physx::PxU32,			firstVertex) \
	INPLACE_TYPE_FIELD(physx::PxU32,			polygonsDataOffset)
#include INPLACE_TYPE_BUILD()

//struct CollisionTriMeshData
#define INPLACE_TYPE_STRUCT_NAME CollisionTriMeshData
#define INPLACE_TYPE_STRUCT_BASE CollisionData
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxBounds3,		aabb) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,	pose) \
	INPLACE_TYPE_FIELD(physx::PxMat34Legacy,	inversePose) \
	INPLACE_TYPE_FIELD(physx::PxU32,			numTriangles) \
	INPLACE_TYPE_FIELD(physx::PxU32,			firstIndex) \
	INPLACE_TYPE_FIELD(physx::PxU32,			firstVertex)
#include INPLACE_TYPE_BUILD()

//struct SimulationParams
#define INPLACE_TYPE_STRUCT_NAME SimulationParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxF32,							collisionThreshold) \
	INPLACE_TYPE_FIELD(physx::PxF32,							collisionDistance) \
	INPLACE_TYPE_FIELD(InplaceArray<CollisionBoxData>,			boxes) \
	INPLACE_TYPE_FIELD(InplaceArray<CollisionSphereData>,		spheres) \
	INPLACE_TYPE_FIELD(InplaceArray<CollisionCapsuleData>,		capsules) \
	INPLACE_TYPE_FIELD(InplaceArray<CollisionHalfSpaceData>,	halfSpaces) \
	INPLACE_TYPE_FIELD(InplaceArray<CollisionConvexMeshData>,	convexMeshes) \
	INPLACE_TYPE_FIELD(InplaceArray<CollisionTriMeshData>,		trimeshes) \
	INPLACE_TYPE_FIELD(physx::PxPlane*,							convexPlanes) \
	INPLACE_TYPE_FIELD(physx::PxVec4*,							convexVerts) \
	INPLACE_TYPE_FIELD(physx::PxU32*,							convexPolygonsData) \
	INPLACE_TYPE_FIELD(physx::PxVec4*,							trimeshVerts) \
	INPLACE_TYPE_FIELD(physx::PxU32*,							trimeshIndices)
#define INPLACE_TYPE_STRUCT_LEAVE_OPEN 1
#include INPLACE_TYPE_BUILD()

	APEX_CUDA_CALLABLE PX_INLINE SimulationParams() : convexPlanes(NULL), convexVerts(NULL), convexPolygonsData(NULL) {}
};




			struct GridDensityParams
			{
				bool Enabled;
				physx::PxF32 GridSize;
				physx::PxU32 GridMaxCellCount;
				PxU32 GridResolution;
				physx::PxVec3 DensityOrigin;
				GridDensityParams(): Enabled(false) {}
			};
		
			struct GridDensityFrustumParams
			{
				PxReal nearDimX;
				PxReal farDimX;
				PxReal nearDimY;
				PxReal farDimY;
				PxReal dimZ; 
			};

#ifdef __CUDACC__

#define SIM_FETCH_PLANE(plane, name, idx) { float4 f4 = tex1Dfetch(KERNEL_TEX_REF(name), idx); plane = physx::PxPlane(f4.x, f4.y, f4.z, f4.w); }
#define SIM_FETCH(name, idx) tex1Dfetch(KERNEL_TEX_REF(name), idx)
#define SIM_FLOAT4 float4
#define SIM_INT_AS_FLOAT(x) __int_as_float(x)
#define SIM_INJECTOR_ARRAY const InjectorParamsArray&
#define SIM_FETCH_INJECTOR(injectorArray, injParams, injector) injectorArray.fetchElem(INPLACE_STORAGE_ARGS_VAL, injParams, injector)

			__device__ PX_INLINE physx::PxReal splitFloat4(physx::PxVec3& v3, const SIM_FLOAT4& f4)
			{
				v3.x = f4.x;
				v3.y = f4.y;
				v3.z = f4.z;
				return f4.w;
			}
			__device__ PX_INLINE SIM_FLOAT4 combineFloat4(const physx::PxVec3& v3, physx::PxReal w)
			{
				return make_float4(v3.x, v3.y, v3.z, w);
			}
#else

#define SIM_FETCH_PLANE(plane, name, idx) plane = mem##name[idx];
#define SIM_FETCH(name, idx) mem##name[idx]
#define SIM_FLOAT4 physx::PxVec4
#define SIM_INT_AS_FLOAT(x) *(const PxF32*)(&x)
#define SIM_INJECTOR_ARRAY const InjectorParams*
#define SIM_FETCH_INJECTOR(injectorArray, injParams, injector) injParams = injectorArray[injector];

			PX_INLINE physx::PxReal splitFloat4(physx::PxVec3& v3, const SIM_FLOAT4& f4)
			{
				v3 = f4.getXYZ();
				return f4.w;
			}
			PX_INLINE SIM_FLOAT4 combineFloat4(const physx::PxVec3& v3, physx::PxReal w)
			{
				return physx::PxVec4(v3.x, v3.y, v3.z, w);
			}

#endif
		}
	}
} // namespace physx::apex

#endif
