/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __ATTRACTOR_FSPREVIEW_H__
#define __ATTRACTOR_FSPREVIEW_H__

#include "ApexPreview.h"
#include "ApexRWLockable.h"
#include "NiAPexSDK.h"
#include "NxAttractorFSPreview.h"
#include "NxApexRenderDebug.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

class AttractorFSAsset;

/**
\brief Descriptor for a AttractorFS Preview Asset
*/
class NxAttractorFSPreviewDesc
{
public:
	NxAttractorFSPreviewDesc() :
		mPose(physx::PxMat44()),
		mPreviewDetail()
	{
		mPose = PxMat44::createIdentity();
	};

	/**
	\brief The pose that translates from preview coordinates to world coordinates.
	*/
	physx::PxMat44			mPose;
	/**
	\brief Radius of the attractor.
	*/
	physx::PxF32			mRadius;
	/**
	\brief The detail options of the preview drawing
	*/
	physx::PxU32			mPreviewDetail;
};

/*
	APEX asset preview asset.
	Preview.
*/
class AttractorFSAssetPreview : public NxAttractorFSPreview, public ApexResource, public ApexPreview, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	AttractorFSAssetPreview(const NxAttractorFSPreviewDesc& PreviewDesc, NxApexSDK* myApexSDK, AttractorFSAsset* myAsset, NxApexAssetPreviewScene* previewScene);
	void					drawAttractorFSPreview(void);
	void					destroy();

	PxF32					getAttractorRadius(NxParameterized::Interface* assetParams);

	void					setPose(const physx::PxMat44& pose);	// Sets the preview instance's pose.  This may include scaling.
	const physx::PxMat44	getPose() const;

	void					setRadius(PxF32 radius);
	const PxF32				getRadius() const;

	void					setDetailLevel(physx::PxU32 detail);

	// from NxApexRenderDataProvider
	void lockRenderResources(void);
	void unlockRenderResources(void);
	void updateRenderResources(bool rewriteBuffers = false, void* userRenderData = 0);

	// from NxApexRenderable.h
	void dispatchRenderResources(NxUserRenderer& renderer);
	PxBounds3 getBounds(void) const;

	// from NxApexInterface.h
	void	release(void);

private:
	~AttractorFSAssetPreview();

	physx::PxMat44					mPose;						// the pose for the preview rendering
	NxApexSDK*						mApexSDK;					// pointer to the APEX SDK
	AttractorFSAsset*				mAsset;						// our parent AttractorFS Asset
	NxApexRenderDebug*				mApexRenderDebug;			// Pointer to the RenderLines class to draw the
	NxApexAssetPreviewScene*		mPreviewScene;
																// preview stuff
	physx::PxF32					mRadius;					// the radius of the attractor
	physx::PxU32					mPreviewDetail;				// the detail options of the preview drawing

	physx::PxI32					mDrawGroupBox;

	void drawPreviewShape();
	void drawPreviewAssetInfo();
	void toggleDrawPreview();
	void setDrawGroupsPose();

	void drawInfoLine(physx::PxU32 lineNum, const char* str);
};

}
}
} // namespace physx::apex

#endif // __TURBULENCE_ASSET_PREVIEW_H__
