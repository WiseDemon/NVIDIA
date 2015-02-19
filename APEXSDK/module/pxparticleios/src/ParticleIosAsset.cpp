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
#if NX_SDK_VERSION_MAJOR == 3

#include "ParticleIosAsset.h"
#include "ParticleIosActor.h"
#include "ModuleParticleIos.h"
#include "PsShare.h"
#include "FluidParticleSystemParams.h"

#if defined(APEX_CUDA_SUPPORT)
#include "ParticleIosActorGPU.h"
#endif

namespace physx
{
namespace apex
{
namespace pxparticleios
{

ParticleIosAsset::ParticleIosAsset(ModuleParticleIos* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) :
	mModule(module),
	mName(name),
	mParams((ParticleIosAssetParam*)params)
{
	list.add(*this);
}

ParticleIosAsset::ParticleIosAsset(ModuleParticleIos* module, NxResourceList& list, const char* name):
	mModule(module),
	mName(name),
	mParams(0)
{
	NxParameterized::Traits* traits = NiGetApexSDK()->getParameterizedTraits();
	mParams = (ParticleIosAssetParam*)traits->createNxParameterized(ParticleIosAssetParam::staticClassName());

	list.add(*this);
}

void ParticleIosAsset::release()
{
	mModule->mSdk->releaseAsset(*this);
}

void ParticleIosAsset::destroy()
{
	if (mParams)
	{
		mParams->destroy();
		mParams = NULL;
	}

	delete this;
}

ParticleIosAsset::~ParticleIosAsset()
{
}

ParticleIosActor* ParticleIosAsset::getIosActorInScene(NxApexScene& scene, bool mesh) const
{
	ParticleIosScene* iosScene = mModule->getParticleIosScene(scene);
	if (iosScene != 0)
	{
		for (physx::PxU32 i = 0 ; i < mIosActorList.getSize() ; i++)
		{
			ParticleIosActor* iosActor = DYNAMIC_CAST(ParticleIosActor*)(mIosActorList.getResource(i));
			if (iosActor->mParticleIosScene == iosScene && iosActor->mIsMesh == mesh)
			{
				return iosActor;
			}
		}
	}
	return NULL;
}

NxApexActor* ParticleIosAsset::createIosActor(NxApexScene& scene, NxIofxAsset* iofxAsset)
{
	NX_WRITE_ZONE();
	ParticleIosActor* iosActor = getIosActorInScene(scene, iofxAsset->getMeshAssetCount() > 0);
	if (iosActor == 0)
	{
		ParticleIosScene* iosScene = mModule->getParticleIosScene(scene);
		if (iosScene != 0)
		{
			iosActor = iosScene->createIosActor(mIosActorList, *this, *iofxAsset);
			iosActor->mIsMesh = iofxAsset->getMeshAssetCount() > 0;
		}
	}
	PX_ASSERT(iosActor);
	return iosActor;
}

void ParticleIosAsset::releaseIosActor(NxApexActor& actor)
{
	NX_WRITE_ZONE();
	ParticleIosActor* iosActor = DYNAMIC_CAST(ParticleIosActor*)(&actor);
	iosActor->destroy();
}

PxU32 ParticleIosAsset::forceLoadAssets()
{
	NX_WRITE_ZONE();
	return 0;
}


#ifndef WITHOUT_APEX_AUTHORING
/*******************   ParticleIosAssetAuthoring *******************/
ParticleIosAssetAuthoring::ParticleIosAssetAuthoring(ModuleParticleIos* module, NxResourceList& list):
	ParticleIosAsset(module, list, "Authoring")
{
}
ParticleIosAssetAuthoring::ParticleIosAssetAuthoring(ModuleParticleIos* module, NxResourceList& list, const char* name):
	ParticleIosAsset(module, list, name)
{
}

ParticleIosAssetAuthoring::ParticleIosAssetAuthoring(ModuleParticleIos* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) :
	ParticleIosAsset(module, list, params, name)
{
}

void ParticleIosAssetAuthoring::release()
{
	delete this;
}

void ParticleIosAssetAuthoring::setCollisionGroupName(const char* collisionGroupName)
{
	NxParameterized::Handle h(*mParams, "collisionGroupName");
	h.setParamString(collisionGroupName);
}

void ParticleIosAssetAuthoring::setCollisionGroupMaskName(const char* collisionGroupMaskName)
{
	NxParameterized::Handle h(*mParams, "collisionGroupMaskName");
	h.setParamString(collisionGroupMaskName);
}
#endif // !WITHOUT_APEX_AUTHORING

}
}
} // namespace physx::apex

#endif // NX_SDK_VERSION_MAJOR == 3
