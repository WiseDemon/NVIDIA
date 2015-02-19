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
#if NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED

#include "NxParamUtils.h"
#include "WindFSAsset.h"
#include "WindFSAssetParams.h"
#include "NxWindFSPreview.h"
#include "WindFSAssetPreview.h"
#include "ModulePerfScope.h"
#include "PsShare.h"

#include "WriteCheck.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

using namespace APEX_WIND;


#define ASSET_INFO_XPOS					(-0.9f)	// left position of the asset info
#define ASSET_INFO_YPOS					( 0.9f)	// top position of the asset info
#define DEBUG_TEXT_HEIGHT				(0.35f)	//in screen space -- would be nice to know this!


void WindFSAssetPreview::drawInfoLine(physx::PxU32 lineNum, const char* str)
{
#ifdef WITHOUT_DEBUG_VISUALIZE
	PX_UNUSED(lineNum);
	PX_UNUSED(str);
#else
	PxMat44 cameraMatrix = mPreviewScene->getCameraMatrix();
	mApexRenderDebug->setCurrentColor(mApexRenderDebug->getDebugColor(physx::DebugColors::Blue));
	PxVec3 textLocation = mPose.getPosition(); 
	textLocation += cameraMatrix.column1.getXYZ() * (ASSET_INFO_YPOS - (lineNum * DEBUG_TEXT_HEIGHT));
	cameraMatrix.setPosition(textLocation);
	mApexRenderDebug->debugOrientedText(cameraMatrix, str);
#endif
}

void WindFSAssetPreview::drawPreviewAssetInfo()
{
#ifndef WITHOUT_DEBUG_VISUALIZE
	if (!mApexRenderDebug)
	{
		return;
	}

		char buf[128];
		buf[sizeof(buf) - 1] = 0;

		ApexSimpleString myString;
		ApexSimpleString floatStr;
		physx::PxU32 lineNum = 0;

		mApexRenderDebug->pushRenderState();
		mApexRenderDebug->addToCurrentState(physx::DebugRenderState::NoZbuffer);
		mApexRenderDebug->setCurrentTextScale(1.0f);

		// asset name
		APEX_SPRINTF_S(buf, sizeof(buf) - 1, "%s %s", mAsset->getObjTypeName(), mAsset->getName());
		drawInfoLine(lineNum++, buf);
		lineNum++;

		if(mPreviewScene->getShowFullInfo())
		{
			// TODO: cache strings
			WindFSAssetParams& assetParams = *static_cast<WindFSAssetParams*>(mAsset->getAssetNxParameterized());

			APEX_SPRINTF_S(buf, sizeof(buf) - 1, "fieldStrength = %f",
				assetParams.fieldStrength
				);
			drawInfoLine(lineNum++, buf);
		}
		mApexRenderDebug->popRenderState();
#endif
}

WindFSAssetPreview::~WindFSAssetPreview(void)
{
	if (mApexRenderDebug)
	{
		mApexRenderDebug->reset();
		mApexRenderDebug->release();
		mApexRenderDebug = NULL;
	}
}

void WindFSAssetPreview::setPose(const physx::PxMat44& pose)
{
	mPose = pose;
}

const physx::PxMat44 WindFSAssetPreview::getPose() const
{
	return	mPose;
}


// from NxApexRenderDataProvider
void WindFSAssetPreview::lockRenderResources(void)
{
	ApexRenderable::renderDataLock();
}

void WindFSAssetPreview::unlockRenderResources(void)
{
	ApexRenderable::renderDataUnLock();
}

void WindFSAssetPreview::updateRenderResources(bool /*rewriteBuffers*/, void* /*userRenderData*/)
{
	if (mApexRenderDebug)
	{
		mApexRenderDebug->updateRenderResources();
	}
}

// from NxApexRenderable.h
void WindFSAssetPreview::dispatchRenderResources(NxUserRenderer& renderer)
{
	if (mApexRenderDebug)
	{
		if (mPreviewDetail & WIND_DRAW_ASSET_INFO)
		{
			drawPreviewAssetInfo();
		}
		mApexRenderDebug->dispatchRenderResources(renderer);
	}
}

physx::PxBounds3 WindFSAssetPreview::getBounds(void) const
{
	if (mApexRenderDebug)
	{
		return(mApexRenderDebug->getBounds());
	}
	else
	{
		physx::PxBounds3 b;
		b.setEmpty();
		return b;
	}
}

void WindFSAssetPreview::destroy(void)
{
	delete this;
}

void WindFSAssetPreview::release(void)
{
	if (mInRelease)
	{
		return;
	}
	mInRelease = true;
	mAsset->releaseWindFSPreview(*this);
}

WindFSAssetPreview::WindFSAssetPreview(const NxWindFSPreviewDesc& PreviewDesc, NxApexSDK* myApexSDK, WindFSAsset* myWindFSAsset, NxApexAssetPreviewScene* previewScene) :
	mPose(PreviewDesc.mPose),
	mApexSDK(myApexSDK),
	mAsset(myWindFSAsset),
	mPreviewScene(previewScene),
	mPreviewDetail(PreviewDesc.mPreviewDetail)
{
	mApexRenderDebug = mApexSDK->createApexRenderDebug();
};


void WindFSAssetPreview::setDetailLevel(physx::PxU32 detail)
{
	NX_WRITE_ZONE();
	mPreviewDetail = detail;
}

}
}
} // namespace physx::apex

#endif
