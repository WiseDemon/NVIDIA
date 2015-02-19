/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "EffectPackageAsset.h"
#include "EffectPackageActorParams.h"
#include "EffectPackageActor.h"
#include "ModuleParticles.h"
#include "EmitterEffect.h"
#include "ReadCheck.h"
#include "WriteCheck.h"

#pragma warning(disable:4100)

namespace physx
{

namespace apex
{

namespace particles
{

EffectPackageAsset::EffectPackageAsset(ModuleParticles*, NxResourceList&, const char* name)
{
	PX_ALWAYS_ASSERT();
}

EffectPackageAsset::EffectPackageAsset(ModuleParticles* moduleParticles, NxResourceList& resourceList, NxParameterized::Interface* params, const char* name)
{
	mDefaultActorParams = NULL;
	mName = name;
	mModule = moduleParticles;
	mParams = static_cast< EffectPackageAssetParams*>(params);
	initializeAssetNameTable();
	resourceList.add(*this);
}

EffectPackageAsset::~EffectPackageAsset()
{
}

physx::PxU32	EffectPackageAsset::forceLoadAssets()
{
	NX_WRITE_ZONE();
	return 0;
}

NxParameterized::Interface* EffectPackageAsset::getDefaultActorDesc()
{
	NX_READ_ZONE();
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	PX_ASSERT(traits);
	if (!traits)
	{
		return NULL;
	}
	// create if not yet created
	if (!mDefaultActorParams)
	{
		const char* className = EffectPackageActorParams::staticClassName();
		NxParameterized::Interface* param = traits->createNxParameterized(className);
		NxParameterized::Handle h(param);
		mDefaultActorParams = static_cast<EffectPackageActorParams*>(param);
		PX_ASSERT(param);
		if (!param)
		{
			return NULL;
		}
	}
	return mDefaultActorParams;

}

void	EffectPackageAsset::release()
{
	mModule->mSdk->releaseAsset(*this);
}


NxParameterized::Interface* EffectPackageAsset::getDefaultAssetPreviewDesc()
{
	NX_READ_ZONE();
	PX_ALWAYS_ASSERT();
	return NULL;
}

NxApexActor* EffectPackageAsset::createApexActor(const NxParameterized::Interface& parms, NxApexScene& apexScene)
{
	NX_WRITE_ZONE();
	NxApexActor* ret = NULL;

	ParticlesScene* ds = mModule->getParticlesScene(apexScene);
	if (ds)
	{
		const EffectPackageAssetParams* assetParams = mParams;
		const EffectPackageActorParams* actorParams = static_cast<const EffectPackageActorParams*>(&parms);
		EffectPackageActor* ea = PX_NEW(EffectPackageActor)(this, assetParams, actorParams,
		                         *NiGetApexSDK(),
		                         apexScene,
		                         *ds,
		                         mModule->getModuleTurbulenceFS());

		ret = static_cast< NxApexActor*>(ea);
	}
	return ret;
}

void	EffectPackageAsset::destroy()
{
	if (mDefaultActorParams)
	{
		mDefaultActorParams->destroy();
		mDefaultActorParams = 0;
	}

	if (mParams)
	{
		mParams->destroy();
		mParams = NULL;
	}

	/* Actors are automatically cleaned up on deletion by NxResourceList dtor */
	delete this;

}

void EffectPackageAsset::initializeAssetNameTable()
{
}

PxF32 EffectPackageAsset::getDuration() const
{
	NX_READ_ZONE();
	PxF32 ret = 0;

	for (PxI32 i = 0; i < mParams->Effects.arraySizes[0]; i++)
	{
		NxParameterized::Interface* iface = mParams->Effects.buf[i];
		if (iface && strcmp(iface->className(), EmitterEffect::staticClassName()) == 0)
		{
			EmitterEffect* ee = static_cast< EmitterEffect*>(iface);
			PxF32 v = 0;
			if (ee->EffectProperties.Duration != 0 && ee->EffectProperties.RepeatCount != 9999)
			{
				v = ee->EffectProperties.InitialDelayTime + ee->EffectProperties.RepeatCount * ee->EffectProperties.Duration + ee->EffectProperties.RepeatCount * ee->EffectProperties.RepeatDelay;
			}
			if (v == 0)	// any infinite lifespan sub-effect means the entire effect package has an infinite life
			{
				ret = 0;
				break;
			}
			else if (v > ret)
			{
				ret = v;
			}
		}
	}

	return ret;
}

bool EffectPackageAsset::useUniqueRenderVolume() const
{
	NX_READ_ZONE();
	return mParams ? mParams->LODSettings.UniqueRenderVolume : false;
}

void EffectPackageAssetAuthoring::setToolString(const char* toolString)
{
	if (mParams != NULL)
	{
		NxParameterized::Handle handle(*mParams, "toolString");
		PX_ASSERT(handle.isValid());
		if (handle.isValid())
		{
			PX_ASSERT(handle.parameterDefinition()->type() == NxParameterized::TYPE_STRING);
			handle.setParamString(toolString);
		}
	}
}

} // end of particles namespace
} // end of apex namespace
} // end of physx namespace
