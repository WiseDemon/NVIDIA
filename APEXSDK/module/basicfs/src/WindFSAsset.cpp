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

#include "WindFSAsset.h"
#include "WindFSActor.h"
#include "ModuleBasicFS.h"

#include "BasicFSScene.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

NxAuthObjTypeID	WindFSAsset::mAssetTypeID;

WindFSAsset::WindFSAsset(ModuleBasicFS* module, NxResourceList& list, const char* name) 
			: BasicFSAsset(module, name)
			, mDefaultActorParams(NULL)
			, mDefaultPreviewParams(NULL)
{
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	mParams = static_cast<WindFSAssetParams*>(traits->createNxParameterized(WindFSAssetParams::staticClassName()));
	PX_ASSERT(mParams);

	list.add(*this);
}

WindFSAsset::WindFSAsset(ModuleBasicFS* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) 
			: BasicFSAsset(module, name)
			, mParams(static_cast<WindFSAssetParams*>(params))
			, mDefaultActorParams(NULL)
			, mDefaultPreviewParams(NULL)
{
	list.add(*this);
}

WindFSAsset::~WindFSAsset()
{
}


void WindFSAsset::destroy()
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

NxParameterized::Interface* WindFSAsset::getDefaultActorDesc()
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
		NxParameterized::Interface* param = traits->createNxParameterized(WindFSActorParams::staticClassName());
		mDefaultActorParams = static_cast<WindFSActorParams*>(param);

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

NxApexActor* WindFSAsset::createApexActor(const NxParameterized::Interface& params, NxApexScene& apexScene)
{
	NxApexActor* ret = 0;

	if (strcmp(params.className(), WindFSActorParams::staticClassName()) == 0)
	{
		const WindFSActorParams& actorParams = static_cast<const WindFSActorParams&>(params);

		BasicFSScene* es = mModule->getBasicFSScene(apexScene);
		ret = es->createWindFSActor(actorParams, *this, mFSActors);
	}
	return ret;
}

NxParameterized::Interface* WindFSAsset::getDefaultAssetPreviewDesc()
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
		const char* className = WindFSPreviewParams::staticClassName();
		NxParameterized::Interface* param = traits->createNxParameterized(className);
		mDefaultPreviewParams = static_cast<WindFSPreviewParams*>(param);

		PX_ASSERT(param);
		if (!param)
		{
			return NULL;
		}
	}

	return mDefaultPreviewParams;
}

NxApexAssetPreview* WindFSAsset::createApexAssetPreview(const NxParameterized::Interface& params, NxApexAssetPreviewScene* previewScene)
{
	NxApexAssetPreview* ret = 0;

	const char* className = params.className();
	if (strcmp(className, WindFSPreviewParams::staticClassName()) == 0)
	{
		NxWindFSPreviewDesc desc;
		const WindFSPreviewParams* pDesc = static_cast<const WindFSPreviewParams*>(&params);

		desc.mPose = pDesc->globalPose;

		desc.mPreviewDetail = 0;
		if (pDesc->drawAssetInfo)
		{
			desc.mPreviewDetail |= APEX_WIND::WIND_DRAW_ASSET_INFO;
		}

		ret = createWindFSPreview(desc, previewScene);
	}

	return ret;
}

NxWindFSPreview* WindFSAsset::createWindFSPreview(const NxWindFSPreviewDesc& desc, NxApexAssetPreviewScene* previewScene)
{
	return createWindFSPreviewImpl(desc, this, previewScene);
}

NxWindFSPreview* WindFSAsset::createWindFSPreviewImpl(const NxWindFSPreviewDesc& desc, WindFSAsset* jetAsset, NxApexAssetPreviewScene* previewScene)
{
	return PX_NEW(WindFSAssetPreview)(desc, mModule->mSdk, jetAsset, previewScene);
}

void WindFSAsset::releaseWindFSPreview(NxWindFSPreview& nxpreview)
{
	WindFSAssetPreview* preview = DYNAMIC_CAST(WindFSAssetPreview*)(&nxpreview);
	preview->destroy();
}

}
}
} // end namespace physx::apex


#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
