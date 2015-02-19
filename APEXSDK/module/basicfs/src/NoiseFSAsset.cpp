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

#include "NxApex.h"

#include "NoiseFSAsset.h"
#include "NoiseFSActor.h"
#include "ModuleBasicFS.h"

#include "BasicFSScene.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

NxAuthObjTypeID	NoiseFSAsset::mAssetTypeID;

NoiseFSAsset::NoiseFSAsset(ModuleBasicFS* module, NxResourceList& list, const char* name) 
			: BasicFSAsset(module, name)
			, mDefaultActorParams(NULL)
			, mDefaultPreviewParams(NULL)
{
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	mParams = static_cast<NoiseFSAssetParams*>(traits->createNxParameterized(NoiseFSAssetParams::staticClassName()));
	PX_ASSERT(mParams);

	list.add(*this);
}

NoiseFSAsset::NoiseFSAsset(ModuleBasicFS* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) 
			: BasicFSAsset(module, name)
			, mParams(static_cast<NoiseFSAssetParams*>(params))
			, mDefaultActorParams(NULL)
			, mDefaultPreviewParams(NULL)
{
	list.add(*this);
}

NoiseFSAsset::~NoiseFSAsset()
{
}


void NoiseFSAsset::destroy()
{
	if (mParams)
	{
		mParams->destroy();
		mParams = 0;
	}

	if (mDefaultActorParams)
	{
		mDefaultActorParams->destroy();
		mDefaultActorParams = 0;
	}

	/* Actors are automatically cleaned up on deletion by NxResourceList dtor */
	delete this;
}

NxParameterized::Interface* NoiseFSAsset::getDefaultActorDesc()
{
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	PX_ASSERT(traits);
	if (!traits)
	{
		return NULL;
	}

	// create if not yet created
	if (!mDefaultActorParams)
	{
		NxParameterized::Interface* param = traits->createNxParameterized(NoiseFSActorParams::staticClassName());
		mDefaultActorParams = static_cast<NoiseFSActorParams*>(param);

		PX_ASSERT(param);
		if (!param)
		{
			return NULL;
		}
	}
	else
	{
		mDefaultActorParams->initDefaults();
	}

	return mDefaultActorParams;
}

NxApexActor* NoiseFSAsset::createApexActor(const NxParameterized::Interface& params, NxApexScene& apexScene)
{
	NxApexActor* ret = 0;

	if (strcmp(params.className(), NoiseFSActorParams::staticClassName()) == 0)
	{
		const NoiseFSActorParams& actorParams = static_cast<const NoiseFSActorParams&>(params);

		BasicFSScene* es = mModule->getBasicFSScene(apexScene);
		ret = es->createNoiseFSActor(actorParams, *this, mFSActors);
	}
	return ret;
}

NxParameterized::Interface* NoiseFSAsset::getDefaultAssetPreviewDesc()
{
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	PX_ASSERT(traits);
	if (!traits)
	{
		return NULL;
	}

	// create if not yet created
	if (!mDefaultPreviewParams)
	{
		const char* className = NoiseFSPreviewParams::staticClassName();
		NxParameterized::Interface* param = traits->createNxParameterized(className);
		mDefaultPreviewParams = static_cast<NoiseFSPreviewParams*>(param);

		PX_ASSERT(param);
		if (!param)
		{
			return NULL;
		}
	}

	return mDefaultPreviewParams;
}

NxApexAssetPreview* NoiseFSAsset::createApexAssetPreview(const NxParameterized::Interface& params, NxApexAssetPreviewScene* previewScene)
{
	NxApexAssetPreview* ret = 0;

	const char* className = params.className();
	if (strcmp(className, NoiseFSPreviewParams::staticClassName()) == 0)
	{
		NxNoiseFSPreviewDesc desc;
		const NoiseFSPreviewParams* pDesc = static_cast<const NoiseFSPreviewParams*>(&params);

		desc.mPose = pDesc->globalPose;

		desc.mPreviewDetail = 0;
		if (pDesc->drawShape)
		{
			desc.mPreviewDetail |= APEX_NOISE::NOISE_DRAW_SHAPE;
		}
		if (pDesc->drawAssetInfo)
		{
			desc.mPreviewDetail |= APEX_NOISE::NOISE_DRAW_ASSET_INFO;
		}

		ret = createNoiseFSPreview(desc, previewScene);
	}

	return ret;
}

NxNoiseFSPreview* NoiseFSAsset::createNoiseFSPreview(const NxNoiseFSPreviewDesc& desc, NxApexAssetPreviewScene* previewScene)
{
	return createNoiseFSPreviewImpl(desc, this, previewScene);
}

NxNoiseFSPreview* NoiseFSAsset::createNoiseFSPreviewImpl(const NxNoiseFSPreviewDesc& desc, NoiseFSAsset* jetAsset, NxApexAssetPreviewScene* previewScene)
{
	return PX_NEW(NoiseFSAssetPreview)(desc, mModule->mSdk, jetAsset, previewScene);
}

void NoiseFSAsset::releaseNoiseFSPreview(NxNoiseFSPreview& nxpreview)
{
	NoiseFSAssetPreview* preview = DYNAMIC_CAST(NoiseFSAssetPreview*)(&nxpreview);
	preview->destroy();
}

}
}
} // end namespace physx::apex


#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
