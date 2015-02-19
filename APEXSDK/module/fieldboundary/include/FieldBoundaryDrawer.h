/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FIELD_BOUNDARY_DRAWER_H__
#define __FIELD_BOUNDARY_DRAWER_H__

#include "PsShare.h"
namespace physx
{
namespace apex
{

class NxApexRenderDebug;

namespace fieldboundary
{

class FieldBoundaryDrawer
{
public:
	static void drawFieldBoundarySphere(NxApexRenderDebug&, const physx::PxVec3& position, physx::PxF32 radius, physx::PxU32 stepCount);
	static void drawFieldBoundaryBox(NxApexRenderDebug&, const physx::PxVec3& bmin, const physx::PxVec3& bmax);
	static void drawFieldBoundaryCapsule(NxApexRenderDebug&, physx::PxF32 radius, physx::PxF32 height, physx::PxU32 subdivision, const physx::PxMat44& transform);
	static void drawFieldBoundaryConvex(NxApexRenderDebug&, const physx::PxVec3& position, physx::PxU32 numVertices, physx::PxU32 numTriangles, physx::PxU32 pointStrideBytes, physx::PxU32 triangleStrideBytes, const void* pointsBase, const void* trianglesBase);
};

} //namespace fieldboundary
} //namespace apex
} //namespace physx

#endif //__FIELD_BOUNDARY_DRAWER_H__