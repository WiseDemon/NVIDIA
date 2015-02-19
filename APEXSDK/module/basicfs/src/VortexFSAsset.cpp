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

#include "VortexFSAsset.h"
#include "VortexFSActor.h"
#include "ModuleBasicFS.h"
#include "BasicFSScene.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

NxAuthObjTypeID	VortexFSAsset::mAssetTypeID;

VortexFSAsset::VortexFSAsset(ModuleBasicFS* module, NxResourceList& list, const char* name) 
			: BasicFSAsset(module, name)
			, mDefaultActorParams(NULL)
			, mDefaultPreviewParams(NULL)
{
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	mParams = static_cast<VortexFSAssetParams*>(traits->createNxParameterized(VortexFSAssetParams::staticClassName()));
	PX_ASSERT(mParams);

	list.add(*this);
}

VortexFSAsset::VortexFSAsset(ModuleBasicFS* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) 
			: BasicFSAsset(module, name)
			, mParams(static_cast<VortexFSAssetParams*>(params))
			, mDefaultActorParams(NULL)
			, mDefaultPreviewParams(NULL)
{
	list.add(*this);
}

VortexFSAsset::~VortexFSAsset()
{
}


void VortexFSAsset::destroy()
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

NxParameterized::Interface* VortexFSAsset::getDefaultActorDesc()
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
		NxParameterized::Interface* param = traits->createNxParameterized(VortexFSActorParams::staticClassName());
		mDefaultActorParams = static_cast<VortexFSActorParams*>(param);

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

NxApexActor* VortexFSAsset::createApexActor(const NxParameterized::Interface& params, NxApexScene& apexScene)
{
	NxApexActor* ret = 0;

	if (strcmp(params.className(), VortexFSActorParams::staticClassName()) == 0)
	{
		const VortexFSActorParams& actorParams = static_cast<const VortexFSActorParams&>(params);

		BasicFSScene* es = mModule->getBasicFSScene(apexScene);
		ret = es->createVortexFSActor(actorParams, *this, mFSActors);
	}
	return ret;
}


NxVortexFSPreview* VortexFSAsset::createVortexFSPreview(const NxVortexFSPreviewDesc& desc, NxApexAssetPreviewScene* previewScene)
{
	return createVortexFSPreviewImpl(desc, this, previewScene);
}

NxVortexFSPreview* VortexFSAsset::createVortexFSPreviewImpl(const NxVortexFSPreviewDesc& desc, VortexFSAsset* TurboAsset, NxApexAssetPreviewScene* previewScene)
{
	return PX_NEW(VortexFSAssetPreview)(desc, mModule->mSdk, TurboAsset, previewScene);
}

void VortexFSAsset::releaseVortexFSPreview(NxVortexFSPreview& nxpreview)
{
	VortexFSAssetPreview* preview = DYNAMIC_CAST(VortexFSAssetPreview*)(&nxpreview);
	preview->destroy();
}

NxParameterized::Interface* VortexFSAsset::getDefaultAssetPreviewDesc()
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
		const char* className = VortexFSPreviewParams::staticClassName();
		NxParameterized::Interface* param = traits->createNxParameterized(className);
		mDefaultPreviewParams = static_cast<VortexFSPreviewParams*>(param);

		PX_ASSERT(param);
		if (!param)
		{
			return NULL;
		}
	}

	return mDefaultPreviewParams;
}

NxApexAssetPreview* VortexFSAsset::createApexAssetPreview(const NxParameterized::Interface& params, NxApexAssetPreviewScene* previewScene)
{
	NxApexAssetPreview* ret = 0;

	const char* className = params.className();
	if (strcmp(className, VortexFSPreviewParams::staticClassName()) == 0)
	{
		NxVortexFSPreviewDesc desc;
		const VortexFSPreviewParams* pDesc = static_cast<const VortexFSPreviewParams*>(&params);

		desc.mPose = pDesc->globalPose;

		desc.mPreviewDetail = 0;
		if (pDesc->drawShape)
		{
			desc.mPreviewDetail |= APEX_VORTEX::VORTEX_DRAW_SHAPE;
		}
		if (pDesc->drawAssetInfo)
		{
			desc.mPreviewDetail |= APEX_VORTEX::VORTEX_DRAW_ASSET_INFO;
		}

		ret = createVortexFSPreview(desc, previewScene);
	}

	return ret;
}


}
}
} // end namespace physx::apex


#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
