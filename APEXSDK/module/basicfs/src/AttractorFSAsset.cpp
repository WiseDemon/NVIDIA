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

#include "AttractorFSAsset.h"
#include "AttractorFSActor.h"
#include "ModuleBasicFS.h"
#include "BasicFSScene.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

NxAuthObjTypeID	AttractorFSAsset::mAssetTypeID;

AttractorFSAsset::AttractorFSAsset(ModuleBasicFS* module, NxResourceList& list, const char* name) 
			: BasicFSAsset(module, name)
			, mDefaultActorParams(NULL)
			, mDefaultPreviewParams(NULL)
{
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	mParams = static_cast<AttractorFSAssetParams*>(traits->createNxParameterized(AttractorFSAssetParams::staticClassName()));
	PX_ASSERT(mParams);

	list.add(*this);
}

AttractorFSAsset::AttractorFSAsset(ModuleBasicFS* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) 
			: BasicFSAsset(module, name)
			, mParams(static_cast<AttractorFSAssetParams*>(params))
			, mDefaultActorParams(NULL)
			, mDefaultPreviewParams(NULL)
{
	list.add(*this);
}

AttractorFSAsset::~AttractorFSAsset()
{
}


void AttractorFSAsset::destroy()
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

	if (mDefaultPreviewParams)
	{
		mDefaultPreviewParams->destroy();
		mDefaultPreviewParams = 0;
	}


	/* Actors are automatically cleaned up on deletion by NxResourceList dtor */
	delete this;
}

NxParameterized::Interface* AttractorFSAsset::getDefaultActorDesc()
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
		NxParameterized::Interface* param = traits->createNxParameterized(AttractorFSActorParams::staticClassName());
		mDefaultActorParams = static_cast<AttractorFSActorParams*>(param);

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

NxApexActor* AttractorFSAsset::createApexActor(const NxParameterized::Interface& params, NxApexScene& apexScene)
{
	NxApexActor* ret = 0;

	if (strcmp(params.className(), AttractorFSActorParams::staticClassName()) == 0)
	{
		const AttractorFSActorParams& actorParams = static_cast<const AttractorFSActorParams&>(params);

		BasicFSScene* es = mModule->getBasicFSScene(apexScene);
		ret = es->createAttractorFSActor(actorParams, *this, mFSActors);
	}
	return ret;
}


NxAttractorFSPreview* AttractorFSAsset::createAttractorFSPreview(const NxAttractorFSPreviewDesc& desc, NxApexAssetPreviewScene* previewScene)
{
	return createAttractorFSPreviewImpl(desc, this, previewScene);
}

NxAttractorFSPreview* AttractorFSAsset::createAttractorFSPreviewImpl(const NxAttractorFSPreviewDesc& desc, AttractorFSAsset* TurboAsset, NxApexAssetPreviewScene* previewScene)
{
	return PX_NEW(AttractorFSAssetPreview)(desc, mModule->mSdk, TurboAsset, previewScene);
}

void AttractorFSAsset::releaseAttractorFSPreview(NxAttractorFSPreview& nxpreview)
{
	AttractorFSAssetPreview* preview = DYNAMIC_CAST(AttractorFSAssetPreview*)(&nxpreview);
	preview->destroy();
}

NxParameterized::Interface* AttractorFSAsset::getDefaultAssetPreviewDesc()
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
		const char* className = AttractorFSPreviewParams::staticClassName();
		NxParameterized::Interface* param = traits->createNxParameterized(className);
		mDefaultPreviewParams = static_cast<AttractorFSPreviewParams*>(param);

		PX_ASSERT(param);
		if (!param)
		{
			return NULL;
		}
	}

	return mDefaultPreviewParams;
}

NxApexAssetPreview* AttractorFSAsset::createApexAssetPreview(const NxParameterized::Interface& params, NxApexAssetPreviewScene* previewScene)
{
	NxApexAssetPreview* ret = 0;

	const char* className = params.className();
	if (strcmp(className, AttractorFSPreviewParams::staticClassName()) == 0)
	{
		NxAttractorFSPreviewDesc desc;
		const AttractorFSPreviewParams* pDesc = static_cast<const AttractorFSPreviewParams*>(&params);

		desc.mPose = pDesc->globalPose;

		desc.mPreviewDetail = 0;
		if (pDesc->drawShape)
		{
			desc.mPreviewDetail |= APEX_ATTRACT::ATTRACT_DRAW_SHAPE;
		}
		if (pDesc->drawAssetInfo)
		{
			desc.mPreviewDetail |= APEX_ATTRACT::ATTRACT_DRAW_ASSET_INFO;
		}

		ret = createAttractorFSPreview(desc, previewScene);
	}

	return ret;
}

}
}
} // end namespace physx::apex


#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
