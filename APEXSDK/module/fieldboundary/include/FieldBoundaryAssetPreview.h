/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FIELD_BOUNDARY_ASSET_PREVIEW_H__
#define __FIELD_BOUNDARY_ASSET_PREVIEW_H__

#if 0
#include "NxApexSDK.h"
#include "ApexInterface.h"
#endif
#include "PsArray.h"
#include "NxFieldBoundaryPreview.h"
#include "ApexPreview.h"
#include "ApexRWLockable.h"

namespace physx
{
namespace apex
{
namespace fieldboundary
{

class FieldBoundaryAsset;
class FieldBoundaryPreviewParameters;
class FieldBoundaryAssetPreview : public NxFieldBoundaryPreview, public ApexResource, public ApexPreview, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	FieldBoundaryAssetPreview(NxApexSDK&, FieldBoundaryAsset&, const FieldBoundaryPreviewParameters&);

	//NxApexInterface.h
	void release();

	//NxApexRenderDataProvider.h
	void lockRenderResources();
	void unlockRenderResources();
	void updateRenderResources(bool, void*);
	void dispatchRenderResources(NxUserRenderer&);

	//NxApexRenderable.h
	physx::PxBounds3 getBounds() const;

	//NxApexAssetPreview.h
	void setPose(const physx::PxMat44&);
	const physx::PxMat44 getPose() const;

	//NxFieldBoundaryPreview.h
	void setDetailLevel(physx::PxU32) const;

	void destroy();
private:
	FieldBoundaryAssetPreview();
	~FieldBoundaryAssetPreview();
	void onInit();

	NxApexRenderDebug* mApexRenderDebug;
	FieldBoundaryAsset& mAsset;
	const FieldBoundaryPreviewParameters& mPreviewParams;

	physx::PxI32 mDrawIconID;
	physx::PxI32 mDrawIconBoldID;
	physx::Array<physx::PxI32> mDrawBoundariesID;
};

} //namespace fieldboundary
} //namespace apex
} //namespace physx

#endif //__FIELD_BOUNDARY_ASSET_PREVIEW_H__