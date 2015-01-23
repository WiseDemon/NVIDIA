/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "NxApex.h"

#include "BasicIosAsset.h"
#include "BasicIosActor.h"
//#include "ApexSharedSerialization.h"
#include "ModuleBasicIos.h"
#include "PsShare.h"

#if defined(APEX_CUDA_SUPPORT)
#include "BasicIosActorGPU.h"
#endif

namespace physx
{
namespace apex
{
namespace basicios
{

void BasicIosAsset::processParams()
{
	NxParameterized::Handle handle(mParams);
	if (NxParameterized::ERROR_NONE != mParams->getParameterHandle("particleMass.type", handle))
	{
		PX_ALWAYS_ASSERT();
		return;
	}

	const char* type = 0;
	if (NxParameterized::ERROR_NONE != handle.getParamEnum(type))
	{
		PX_ALWAYS_ASSERT();
		return;
	}

	mMassDistribType = 0 == strcmp("uniform", type) ? UNIFORM : NORMAL;
}

BasicIosAsset::BasicIosAsset(ModuleBasicIos* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) :
	mModule(module),
	mName(name),
	mParams((BasicIOSAssetParam*)params)
{
	list.add(*this);
	processParams();
}

BasicIosAsset::BasicIosAsset(ModuleBasicIos* module, NxResourceList& list, const char* name):
	mModule(module),
	mName(name),
	mParams(0)
{
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	mParams = (BasicIOSAssetParam*)traits->createNxParameterized(BasicIOSAssetParam::staticClassName());

	list.add(*this);

	processParams();
}

physx::PxF32 BasicIosAsset::getParticleMass() const
{
	NX_READ_ZONE();
	physx::PxF32 m = 0.0f;
	switch (mMassDistribType)
	{
	case UNIFORM:
		m = mParams->particleMass.center + mParams->particleMass.spread * mSRand.getNext();
		break;
	case NORMAL:
		m = mNormRand.getScaled(mParams->particleMass.center, mParams->particleMass.spread);
		break;
	default:
		PX_ALWAYS_ASSERT();
	}

	return m <= 0 ? mParams->particleMass.center : m; // Clamp
}

void BasicIosAsset::release()
{
	mModule->mSdk->releaseAsset(*this);
}

void BasicIosAsset::destroy()
{
	if (mParams)
	{
		mParams->destroy();
		mParams = NULL;
	}

	delete this;
}

BasicIosAsset::~BasicIosAsset()
{
}

BasicIosActor* BasicIosAsset::getIosActorInScene(NxApexScene& scene, bool mesh) const
{
	BasicIosScene* iosScene = mModule->getBasicIosScene(scene);
	if (iosScene != 0)
	{
		for (physx::PxU32 i = 0 ; i < mIosActorList.getSize() ; i++)
		{
			BasicIosActor* iosActor = DYNAMIC_CAST(BasicIosActor*)(mIosActorList.getResource(i));
			if (iosActor->mBasicIosScene == iosScene && iosActor->mIsMesh == mesh)
			{
				return iosActor;
			}
		}
	}
	return NULL;
}

NxApexActor* BasicIosAsset::createIosActor(NxApexScene& scene, physx::apex::NxIofxAsset* iofxAsset)
{
	BasicIosActor* iosActor = getIosActorInScene(scene, iofxAsset->getMeshAssetCount() > 0);
	if (iosActor == 0)
	{
		BasicIosScene* iosScene = mModule->getBasicIosScene(scene);
		if (iosScene != 0)
		{
			iosActor = iosScene->createIosActor(mIosActorList, *this, *iofxAsset);
			iosActor->mIsMesh = iofxAsset->getMeshAssetCount() > 0;
		}
	}
	PX_ASSERT(iosActor);
	return iosActor;
}

void BasicIosAsset::releaseIosActor(NxApexActor& actor)
{
	BasicIosActor* iosActor = DYNAMIC_CAST(BasicIosActor*)(&actor);
	iosActor->destroy();
}

PxU32 BasicIosAsset::forceLoadAssets()
{
	return 0;
}

bool BasicIosAsset::getSupportsDensity() const
{
	BasicIOSAssetParam* gridParams = (BasicIOSAssetParam*)(getAssetNxParameterized());
	return (gridParams->GridDensity.Enabled);
}

#ifndef WITHOUT_APEX_AUTHORING
/*******************   BasicIosAssetAuthoring *******************/
BasicIosAssetAuthoring::BasicIosAssetAuthoring(ModuleBasicIos* module, NxResourceList& list):
	BasicIosAsset(module, list, "Authoring")
{
}
BasicIosAssetAuthoring::BasicIosAssetAuthoring(ModuleBasicIos* module, NxResourceList& list, const char* name):
	BasicIosAsset(module, list, name)
{
}

BasicIosAssetAuthoring::BasicIosAssetAuthoring(ModuleBasicIos* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) :
	BasicIosAsset(module, list, params, name)
{
}

void BasicIosAssetAuthoring::release()
{
	delete this;
}

void BasicIosAssetAuthoring::setCollisionGroupName(const char* collisionGroupName)
{
	NxParameterized::Handle h(*mParams, "collisionGroupName");
	h.setParamString(collisionGroupName);
}

void BasicIosAssetAuthoring::setCollisionGroupMaskName(const char* collisionGroupMaskName)
{
	NxParameterized::Handle h(*mParams, "collisionGroupMaskName");
	h.setParamString(collisionGroupMaskName);
}


#endif

}
}
} // namespace physx::apex
