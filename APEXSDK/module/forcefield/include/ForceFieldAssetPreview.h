/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FORCEFIELD_ASSET_PREVIEW_H__
#define __FORCEFIELD_ASSET_PREVIEW_H__

#include "ApexPreview.h"

#include "NiAPexSDK.h"
#include "NxForceFieldPreview.h"
#include "NxApexRenderDebug.h"
#include "ForceFieldAsset.h"

#include "ApexRWLockable.h"
#include "ReadCheck.h"
#include "WriteCheck.h"

namespace physx
{
namespace apex
{

namespace forcefield
{

/*
	APEX asset preview explosion asset.
	Preview.
*/
class ForceFieldAssetPreview : public NxForceFieldPreview, public ApexResource, public ApexPreview, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ForceFieldAssetPreview(const NxForceFieldPreviewDesc& PreviewDesc, NxApexSDK* myApexSDK, ForceFieldAsset* myForceFieldAsset, NxApexAssetPreviewScene* previewScene);
	void					drawForceFieldPreview(void);
	void					drawForceFieldPreviewUnscaled(void);
	void					drawForceFieldPreviewScaled(void);
	void					drawForceFieldPreviewIcon(void);
	void					drawForceFieldBoundaries(void);
	void					drawForceFieldWithCylinder();
	void					destroy();

	void					setPose(const physx::PxMat44& pose);	// Sets the preview instance's pose.  This may include scaling.
	const physx::PxMat44	getPose() const;

	// from NxApexRenderDataProvider
	void					lockRenderResources(void);
	void					unlockRenderResources(void);
	void					updateRenderResources(bool rewriteBuffers = false, void* userRenderData = 0);

	// from NxApexRenderable.h
	void					dispatchRenderResources(NxUserRenderer& renderer);
	physx::PxBounds3		getBounds(void) const;

	// from NxApexInterface.h
	void					release(void);

	void					setDetailLevel(physx::PxU32 detail);

	typedef struct
	{
		physx::PxF32 x, y;
	} point2;

private:

	~ForceFieldAssetPreview();
	physx::PxMat44					mPose;						// the pose for the preview rendering
	NxApexSDK*						mApexSDK;					// pointer to the APEX SDK
	ForceFieldAsset*				mAsset;						// our parent ForceField Asset
	NxApexRenderDebug*				mApexRenderDebug;			// Pointer to the RenderLines class to draw the
	// preview stuff
	physx::PxU32					mDrawGroupIconScaled;		// the ApexDebugRenderer draw group for the Icon
	physx::PxU32					mDrawGroupCylinder;
	physx::PxU32					mPreviewDetail;				// the detail options of the preview drawing
	physx::PxF32					mIconScale;					// the scale for the icon
	NxApexAssetPreviewScene*		mPreviewScene;

	bool							mDrawWithCylinder;

	void							drawIcon(void);
	void							drawMultilinePoint2(const point2* pts, physx::PxU32 numPoints, physx::PxU32 color);
	void							setIconScale(physx::PxF32 scale);

	void							setPose(physx::PxMat44 pose);

	void							toggleDrawPreview();
	void							setDrawGroupsPoseScaled();
};

}
}
} // end namespace physx::apex

#endif // __FORCEFIELD_ASSET_PREVIEW_H__
