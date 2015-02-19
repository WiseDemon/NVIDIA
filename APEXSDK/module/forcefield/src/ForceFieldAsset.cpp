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
#include "ForceFieldAssetPreview.h"
#include "ForceFieldAsset.h"
#include "ForceFieldActor.h"
#include "ModuleForceField.h"
#include "ForceFieldScene.h"
#include "ApexResourceHelper.h"
#include "NxApexAssetPreviewScene.h"


namespace physx
{
namespace apex
{
namespace forcefield
{

ForceFieldAsset::ForceFieldAsset(ModuleForceField* module, NxResourceList& list, const char* name) :
	mModule(module),
	mName(name),
	mDefaultActorParams(NULL),
	mDefaultPreviewParams(NULL)
{
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	mParams = (ForceFieldAssetParams*)traits->createNxParameterized(ForceFieldAssetParams::staticClassName());

	initializeAssetNameTable();

	list.add(*this);
}

ForceFieldAsset::ForceFieldAsset(ModuleForceField* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) :
	mModule(module),
	mName(name),
	mParams((ForceFieldAssetParams*)params),
	mDefaultActorParams(NULL),
	mDefaultPreviewParams(NULL)
{
	initializeAssetNameTable();

	list.add(*this);
}

ForceFieldAsset::~ForceFieldAsset()
{
}

void ForceFieldAsset::destroy()
{
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
	if (mParams)
	{
		mParams->destroy();
		mParams = NULL;
	}
	/* Assets that were forceloaded or loaded by actors will be automatically
	 * released by the ApexAssetTracker member destructors.
	 */

	/* Actors are automatically cleaned up on deletion by NxResourceList dtor */
	delete this;
}

physx::PxU32 ForceFieldAsset::forceLoadAssets()
{
	NX_WRITE_ZONE();
	// Is there anything to be done here?
	return NULL;
}

void ForceFieldAsset::initializeAssetNameTable()
{
	ApexSimpleString tmpStr;
	NxParameterized::Handle h(*mParams);
	NxParameterized::Interface* refPtr;

	if (mParams->forceFieldKernelType == NULL)
	{
		NxParameterized::Handle h(mParams);
		h.getParameter("forceFieldKernelType");
		h.initParamRef(h.parameterDefinition()->refVariantVal(0), true);
	}

	mParams->getParameterHandle("forceFieldKernelType", h);
	mParams->getParamRef(h, refPtr);
	PX_ASSERT(refPtr);
	if (!refPtr)
	{
		APEX_INTERNAL_ERROR("No force field kernel specified");
		return;
	}

	tmpStr = refPtr->className();

	mGenericParams = NULL;
	mRadialParams = NULL;
	mFalloffParams = NULL;
	mNoiseParams = NULL;

	if (tmpStr == GenericForceFieldKernelParams::staticClassName())
	{
		mGenericParams = static_cast<GenericForceFieldKernelParams*>(refPtr);
	}
	else if (tmpStr == RadialForceFieldKernelParams::staticClassName())
	{
		mRadialParams = static_cast<RadialForceFieldKernelParams*>(refPtr);

		NxParameterized::Handle h(*mRadialParams);

		mRadialParams->getParameterHandle("falloffParameters", h);
		mRadialParams->getParamRef(h, refPtr);
		PX_ASSERT(refPtr);
		
		mFalloffParams = static_cast<ForceFieldFalloffParams*>(refPtr);
		
		mRadialParams->getParameterHandle("noiseParameters", h);
		mRadialParams->getParamRef(h, refPtr);
		PX_ASSERT(refPtr);

		mNoiseParams = static_cast<ForceFieldNoiseParams*>(refPtr);		
	}
	else
	{
		PX_ASSERT(0 && "Invalid force field kernel type for APEX_ForceField.");
		return;
	}
}

NxParameterized::Interface* ForceFieldAsset::getDefaultActorDesc()
{
	NX_WRITE_ZONE();
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	PX_ASSERT(traits);
	if (!traits)
	{
		return NULL;
	}

	// create if not yet created
	if (!mDefaultActorParams)
	{
		NxParameterized::ErrorType error = NxParameterized::ERROR_NONE;
		PX_UNUSED(error);

		const char* className = ForceFieldActorParams::staticClassName();
		NxParameterized::Interface* param = traits->createNxParameterized(className);
		NxParameterized::Handle h(param);
		mDefaultActorParams = static_cast<ForceFieldActorParams*>(param);

		PX_ASSERT(param);
		if (!param)
		{
			return NULL;
		}
	}

	return mDefaultActorParams;
}


NxParameterized::Interface* ForceFieldAsset::getDefaultAssetPreviewDesc()
{
	NX_WRITE_ZONE();
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	PX_ASSERT(traits);
	if (!traits)
	{
		return NULL;
	}

	// create if not yet created
	if (!mDefaultPreviewParams)
	{
		const char* className = ForceFieldAssetPreviewParams::staticClassName();
		NxParameterized::Interface* param = traits->createNxParameterized(className);
		mDefaultPreviewParams = static_cast<ForceFieldAssetPreviewParams*>(param);

		PX_ASSERT(param);
		if (!param)
		{
			return NULL;
		}
	}
	else
	{
		mDefaultPreviewParams->initDefaults();
	}

	return mDefaultPreviewParams;
}

NxApexActor* ForceFieldAsset::createApexActor(const NxParameterized::Interface& parms, NxApexScene& apexScene)
{
	NX_WRITE_ZONE();
	if (!isValidForActorCreation(parms, apexScene))
	{
		return NULL;
	}

	NxApexActor* ret = 0;

	if (strcmp(parms.className(), ForceFieldActorParams::staticClassName()) == 0)
	{
		NxForceFieldActorDesc desc;

		const ForceFieldActorParams* pDesc = static_cast<const ForceFieldActorParams*>(&parms);
		desc.initialPose		= pDesc->initialPose;
		desc.scale				= pDesc->scale;
#if NX_SDK_VERSION_MAJOR == 2
		desc.samplerFilterData  = ApexResourceHelper::resolveCollisionGroup64(pDesc->fieldSamplerFilterDataName ? pDesc->fieldSamplerFilterDataName : mParams->fieldSamplerFilterDataName);
		desc.boundaryFilterData = ApexResourceHelper::resolveCollisionGroup64(pDesc->fieldBoundaryFilterDataName ? pDesc->fieldBoundaryFilterDataName : mParams->fieldBoundaryFilterDataName);
#else
		desc.samplerFilterData  = ApexResourceHelper::resolveCollisionGroup128(pDesc->fieldSamplerFilterDataName ? pDesc->fieldSamplerFilterDataName : mParams->fieldSamplerFilterDataName);
		desc.boundaryFilterData = ApexResourceHelper::resolveCollisionGroup128(pDesc->fieldBoundaryFilterDataName ? pDesc->fieldBoundaryFilterDataName : mParams->fieldBoundaryFilterDataName);
#endif
		ForceFieldScene* es = mModule->getForceFieldScene(apexScene);
		ret = es->createForceFieldActor(desc, *this, mForceFieldActors);
	}

	return ret;
}

NxApexAssetPreview* ForceFieldAsset::createApexAssetPreview(const NxParameterized::Interface& parms, NxApexAssetPreviewScene* previewScene)
{
	NX_WRITE_ZONE();
	NxApexAssetPreview* ret = 0;

	const char* className = parms.className();
	if (strcmp(className, ForceFieldAssetPreviewParams::staticClassName()) == 0)
	{
		NxForceFieldPreviewDesc desc;
		const ForceFieldAssetPreviewParams* pDesc = static_cast<const ForceFieldAssetPreviewParams*>(&parms);

		desc.mPose	= pDesc->pose;
		desc.mIconScale = pDesc->iconScale;
		desc.mPreviewDetail = 0;
		if (pDesc->drawIcon)
		{
			desc.mPreviewDetail |= APEX_FORCEFIELD::FORCEFIELD_DRAW_ICON;
		}
		if (pDesc->drawBoundaries)
		{
			desc.mPreviewDetail |= APEX_FORCEFIELD::FORCEFIELD_DRAW_BOUNDARIES;
		}
		if (pDesc->drawBold)
		{
			desc.mPreviewDetail |= APEX_FORCEFIELD::FORCEFIELD_DRAW_WITH_CYLINDERS;
		}

		ret = createForceFieldPreview(desc, previewScene);
	}

	return ret;
}

void ForceFieldAsset::releaseForceFieldActor(NxForceFieldActor& nxactor)
{
	NX_WRITE_ZONE();
	ForceFieldActor* actor = DYNAMIC_CAST(ForceFieldActor*)(&nxactor);
	actor->destroy();
}

NxForceFieldPreview* ForceFieldAsset::createForceFieldPreview(const NxForceFieldPreviewDesc& desc, NxApexAssetPreviewScene* previewScene)
{
	return(createForceFieldPreviewImpl(desc, this, previewScene));
}

NxForceFieldPreview* ForceFieldAsset::createForceFieldPreviewImpl(const NxForceFieldPreviewDesc& desc, ForceFieldAsset* ForceFieldAsset, NxApexAssetPreviewScene* previewScene)
{
	return(PX_NEW(ForceFieldAssetPreview)(desc, mModule->mSdk, ForceFieldAsset, previewScene));
}

void ForceFieldAsset::releaseForceFieldPreview(NxForceFieldPreview& nxpreview)
{
	ForceFieldAssetPreview* preview = DYNAMIC_CAST(ForceFieldAssetPreview*)(&nxpreview);
	preview->destroy();
}

}
}
} // end namespace physx::apex

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
