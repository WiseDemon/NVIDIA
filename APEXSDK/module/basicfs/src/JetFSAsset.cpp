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

#include "JetFSAsset.h"
#include "JetFSActor.h"
#include "ModuleBasicFS.h"

#include "BasicFSScene.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

NxAuthObjTypeID JetFSAsset::mAssetTypeID; 

JetFSAsset::JetFSAsset(ModuleBasicFS* module, NxResourceList& list, const char* name) 
			: BasicFSAsset(module, name)
			, mDefaultActorParams(NULL)
			, mDefaultPreviewParams(NULL)
{
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	mParams = static_cast<JetFSAssetParams*>(traits->createNxParameterized(JetFSAssetParams::staticClassName()));
	PX_ASSERT(mParams);

	list.add(*this);
}

JetFSAsset::JetFSAsset(ModuleBasicFS* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) 
			: BasicFSAsset(module, name)
			, mParams(static_cast<JetFSAssetParams*>(params))
			, mDefaultActorParams(NULL)
			, mDefaultPreviewParams(NULL)
{
	list.add(*this);
}

JetFSAsset::~JetFSAsset()
{
}


void JetFSAsset::destroy()
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

NxParameterized::Interface* JetFSAsset::getDefaultActorDesc()
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
		NxParameterized::Interface* param = traits->createNxParameterized(JetFSActorParams::staticClassName());
		mDefaultActorParams = static_cast<JetFSActorParams*>(param);

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

NxApexActor* JetFSAsset::createApexActor(const NxParameterized::Interface& params, NxApexScene& apexScene)
{
	NxApexActor* ret = 0;

	if (strcmp(params.className(), JetFSActorParams::staticClassName()) == 0)
	{
		const JetFSActorParams& actorParams = static_cast<const JetFSActorParams&>(params);

		BasicFSScene* es = mModule->getBasicFSScene(apexScene);
		ret = es->createJetFSActor(actorParams, *this, mFSActors);
	}
	return ret;
}

NxParameterized::Interface* JetFSAsset::getDefaultAssetPreviewDesc()
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
		const char* className = JetFSPreviewParams::staticClassName();
		NxParameterized::Interface* param = traits->createNxParameterized(className);
		mDefaultPreviewParams = static_cast<JetFSPreviewParams*>(param);

		PX_ASSERT(param);
		if (!param)
		{
			return NULL;
		}
	}

	return mDefaultPreviewParams;
}

NxApexAssetPreview* JetFSAsset::createApexAssetPreview(const NxParameterized::Interface& params, NxApexAssetPreviewScene* previewScene)
{
	NxApexAssetPreview* ret = 0;

	const char* className = params.className();
	if (strcmp(className, JetFSPreviewParams::staticClassName()) == 0)
	{
		NxJetFSPreviewDesc desc;
		const JetFSPreviewParams* pDesc = static_cast<const JetFSPreviewParams*>(&params);

		desc.mPose = pDesc->globalPose;

		desc.mPreviewDetail = 0;
		if (pDesc->drawShape)
		{
			desc.mPreviewDetail |= APEX_JET::JET_DRAW_SHAPE;
		}
		if (pDesc->drawAssetInfo)
		{
			desc.mPreviewDetail |= APEX_JET::JET_DRAW_ASSET_INFO;
		}

		ret = createJetFSPreview(desc, previewScene);
	}

	return ret;
}

NxJetFSPreview* JetFSAsset::createJetFSPreview(const NxJetFSPreviewDesc& desc, NxApexAssetPreviewScene* previewScene)
{
	return createJetFSPreviewImpl(desc, this, previewScene);
}

NxJetFSPreview* JetFSAsset::createJetFSPreviewImpl(const NxJetFSPreviewDesc& desc, JetFSAsset* jetAsset, NxApexAssetPreviewScene* previewScene)
{
	return PX_NEW(JetFSAssetPreview)(desc, mModule->mSdk, jetAsset, previewScene);
}

void JetFSAsset::releaseJetFSPreview(NxJetFSPreview& nxpreview)
{
	JetFSAssetPreview* preview = DYNAMIC_CAST(JetFSAssetPreview*)(&nxpreview);
	preview->destroy();
}

}
}
} // end namespace physx::apex


#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
