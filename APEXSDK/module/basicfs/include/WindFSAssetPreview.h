/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __WIND_ASSET_PREVIEW_H__
#define __WIND_ASSET_PREVIEW_H__

#include "ApexPreview.h"
#include "ApexRWLockable.h"
#include "NiAPexSDK.h"
#include "NxWindFSPreview.h"
#include "NxApexRenderDebug.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

class WindFSAsset;

/**
\brief Descriptor for a WindFS Preview Asset
*/
class NxWindFSPreviewDesc
{
public:
	NxWindFSPreviewDesc() :
		mPose(physx::PxMat44()),
		mPreviewDetail(APEX_WIND::WIND_DRAW_FULL_DETAIL)
	{
		mPose = PxMat44::createIdentity();
	};

	/**
	\brief The pose that translates from preview coordinates to world coordinates.
	*/
	physx::PxMat44			mPose;
	/**
	\brief The detail options of the preview drawing
	*/
	physx::PxU32			mPreviewDetail;
};

/*
	APEX asset preview asset.
	Preview.
*/
class WindFSAssetPreview : public NxWindFSPreview, public ApexResource, public ApexPreview, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	WindFSAssetPreview(const NxWindFSPreviewDesc& PreviewDesc, NxApexSDK* myApexSDK, WindFSAsset* myAsset, NxApexAssetPreviewScene* previewScene);
	void					destroy();

	void					setPose(const physx::PxMat44& pose);	// Sets the preview instance's pose.  This may include scaling.
	const physx::PxMat44	getPose() const;

	// from NxApexRenderDataProvider
	void					lockRenderResources(void);
	void					unlockRenderResources(void);
	void					updateRenderResources(bool rewriteBuffers = false, void* userRenderData = 0);

	// from NxApexRenderable.h
	void					dispatchRenderResources(NxUserRenderer& renderer);
	PxBounds3				getBounds(void) const;

	// from NxApexInterface.h
	void					release(void);

private:

	~WindFSAssetPreview();
	physx::PxMat44					mPose;						// the pose for the preview rendering
	NxApexSDK*						mApexSDK;					// pointer to the APEX SDK
	WindFSAsset*					mAsset;						// our parent WindFS Asset
	NxApexRenderDebug*				mApexRenderDebug;			// Pointer to the RenderLines class to draw the
	NxApexAssetPreviewScene*		mPreviewScene;

	// preview stuff
	physx::PxU32					mPreviewDetail;				// the detail options of the preview drawing


//	void							setHalfLengthDimensions(physx::PxVec3 halfLenDim);
	void							setDetailLevel(physx::PxU32 detail);

	void							drawPreviewAssetInfo();
	void							drawInfoLine(physx::PxU32 lineNum, const char* str);
};

}
}
} // namespace physx::apex

#endif // __NOISE_ASSET_PREVIEW_H__
