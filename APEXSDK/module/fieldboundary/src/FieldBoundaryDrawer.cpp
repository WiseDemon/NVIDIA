/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "NxApexDefs.h"
#include "MinPhysxSdkVersion.h"
#if NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED && NX_SDK_VERSION_MAJOR == 2

#include "FieldBoundaryDrawer.h"
#include "NiApexRenderDebug.h"

namespace physx
{
namespace apex
{
namespace fieldboundary
{

void FieldBoundaryDrawer::drawFieldBoundarySphere(NxApexRenderDebug& rd, const physx::PxVec3& position, physx::PxF32 radius, physx::PxU32 stepCount)
{
	PX_ASSERT(&rd != NULL);
	rd.debugDetailedSphere(position, radius, stepCount);
}

void FieldBoundaryDrawer::drawFieldBoundaryBox(NxApexRenderDebug& rd, const physx::PxVec3& bmin, const physx::PxVec3& bmax)
{
	PX_ASSERT(&rd != NULL);
	rd.debugBound(bmin, bmax);
}

void FieldBoundaryDrawer::drawFieldBoundaryCapsule(NxApexRenderDebug& rd, physx::PxF32 radius, physx::PxF32 height, physx::PxU32 subdivision, const physx::PxMat44& transform)
{
	PX_ASSERT(&rd != NULL);
	rd.debugOrientedCapsule(radius, height, subdivision, transform);
}

void FieldBoundaryDrawer::drawFieldBoundaryConvex(NxApexRenderDebug& rd, const physx::PxVec3& position, physx::PxU32 numVertices, physx::PxU32 numTriangles, physx::PxU32 pointStrideBytes, physx::PxU32 triangleStrideBytes, const void* pointsBase, const void* trianglesBase)
{
	PX_UNUSED(pointStrideBytes);
	PX_UNUSED(numVertices);
	PX_ASSERT(&rd != NULL);
	PX_ASSERT(numTriangles != 0u);
	PX_ASSERT((triangleStrideBytes == (3 * sizeof(physx::PxU32))) || (triangleStrideBytes == (3 * sizeof(physx::PxU16))));
	PX_ASSERT(numVertices != 0);
	PX_ASSERT(pointStrideBytes == sizeof(physx::PxVec3));
	for (physx::PxU32 k = 0; k < numTriangles; k++)
	{
		physx::PxU32 triIndices[3];
		physx::PxVec3 triPoints[3];
		if (triangleStrideBytes == (3 * sizeof(physx::PxU32)))
		{
			triIndices[0] = *((physx::PxU32*)trianglesBase + 3 * k + 0);
			triIndices[1] = *((physx::PxU32*)trianglesBase + 3 * k + 1);
			triIndices[2] = *((physx::PxU32*)trianglesBase + 3 * k + 2);
		}
		else
		{
			triIndices[0] = *((physx::PxU16*)trianglesBase + 3 * k + 0);
			triIndices[1] = *((physx::PxU16*)trianglesBase + 3 * k + 1);
			triIndices[2] = *((physx::PxU16*)trianglesBase + 3 * k + 2);
		}
		PX_ASSERT(triIndices[0] < numVertices);
		PX_ASSERT(triIndices[1] < numVertices);
		PX_ASSERT(triIndices[2] < numVertices);
		triPoints[0] = *((physx::PxVec3*)pointsBase + triIndices[0]);
		triPoints[1] = *((physx::PxVec3*)pointsBase + triIndices[1]);
		triPoints[2] = *((physx::PxVec3*)pointsBase + triIndices[2]);
		triPoints[0] += position;
		triPoints[1] += position;
		triPoints[2] += position;
		rd.debugPolygon(3, triPoints);
	}
}

} //namespace fieldboundary
} //namespace apex
} //namespace physx

#endif