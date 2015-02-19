/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __EXPLOSION_ASSET_PREVIEW_H__
#define __EXPLOSION_ASSET_PREVIEW_H__

#include "ApexPreview.h"

#include "NiAPexSDK.h"
#include "NxExplosionPreview.h"
#include "NxApexRenderDebug.h"
#include "ExplosionAsset.h"
#include "ApexRWLockable.h"

namespace physx
{
namespace apex
{
namespace explosion
{

/*
	APEX asset preview explosion asset.
	Preview.
*/
class ExplosionAssetPreview : public NxExplosionPreview, public ApexResource, public ApexPreview, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ExplosionAssetPreview(const NxExplosionPreviewDesc& PreviewDesc, NxApexSDK* myApexSDK, ExplosionAsset* myExplosionAsset);
	void				drawExplosionPreview(void);
	void				drawExplosionPreviewUnscaled(void);
	void				drawExplosionPreviewScaled(void);
	void				drawExplosionPreviewIcon(void);
	void				drawExplosionBoundaries(void);
	void				drawExplosionWithCylinder();
	void				destroy();

	void				setPose(const physx::PxMat44& pose);	// Sets the preview instance's pose.  This may include scaling.
	const physx::PxMat44	getPose() const;

	// from NxApexRenderDataProvider
	void lockRenderResources(void);
	void unlockRenderResources(void);
	void updateRenderResources(bool rewriteBuffers = false, void* userRenderData = 0);

	// from NxApexRenderable.h
	void dispatchRenderResources(NxUserRenderer& renderer);
	physx::PxBounds3 getBounds(void) const;

	// from NxApexInterface.h
	void	release(void);
	typedef struct
	{
		physx::PxF32 x, y;
	} point2;

private:

	~ExplosionAssetPreview();
	NxUserRenderResourceManager*	mRrm;						// pointer to the users render resource manager
	physx::PxMat34Legacy			mPose;						// the pose for the preview rendering
	physx::PxMat34Legacy			mInversePose;				// the inverse Pose for the preview rendering
	NxApexSDK*						mApexSDK;					// pointer to the APEX SDK
	ExplosionAsset*					mAsset;						// our parent Explosion Asset
	NxApexRenderDebug*				mApexRenderDebug;			// Pointer to the RenderLines class to draw the
	// preview stuff
	physx::PxU32					mDrawGroupFixed;			// the ApexDebugRenderer draw group for Explosion Preview
	// fixed assets
	physx::PxU32					mDrawGroupScaled;			// the ApexDebugRenderer draw group for Explosion Preview
	// assets that are scaled
	physx::PxU32					mDrawGroupIconScaled;		// the ApexDebugRenderer draw group for the Icon
	physx::PxU32					mDrawGroupCylinder;
	physx::PxU32					mPreviewDetail;				// the detail options of the preview drawing
	physx::PxF32					mIconScale;					// the scale for the icon
	bool							mDrawWithCylinder;

	void							drawIcon(void);
	void							drawMultilinePoint2(const point2* pts, physx::PxU32 numPoints, physx::PxU32 color);
	void							setIconScale(physx::PxF32 scale)
	{
		mIconScale = scale;
		drawExplosionPreview();
	};
	void							setPose(physx::PxMat34Legacy pose)
	{
		mPose = pose;
		drawExplosionPreview();
	};
	void							setDetailLevel(physx::PxU32 detail)
	{
		mPreviewDetail = detail;
		drawExplosionPreview();
	};
};

}
}
} // end namespace physx::apex

#endif // __EXPLOSION_ASSET_PREVIEW_H__
