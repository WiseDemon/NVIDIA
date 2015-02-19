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
#if NX_SDK_VERSION_MAJOR == 2

#include "FluidIosScene.h"
#include "ModuleFluidIos.h"
#include "FluidIosActor.h"
#include "PsShare.h"
#include "NiApexScene.h"
#include "ModulePerfScope.h"

namespace physx
{
namespace apex
{
namespace nxfluidios
{

FluidIosScene::FluidIosScene(ModuleFluidIos& _module, NiApexScene& scene, NiApexRenderDebug* renderDebug, NxResourceList& list) :
	mModule(&_module),
	mPhysXScene(NULL),
	mApexScene(&scene),
	mCompartment(NULL),
	mSPHCompartment(NULL),
	mSumBenefit(0.0f),
	mRenderDebug(renderDebug)
	
{
	list.add(*this);

	/* Initialize reference to FluidIosDebugRenderParams */
	mDebugRenderParams = DYNAMIC_CAST(DebugRenderParams*)(mApexScene->getDebugRenderParams());
	PX_ASSERT(mDebugRenderParams);
	NxParameterized::Handle handle(*mDebugRenderParams), memberHandle(*mDebugRenderParams);
	int size;

	if (mDebugRenderParams->getParameterHandle("moduleName", handle) == NxParameterized::ERROR_NONE)
	{
		handle.getArraySize(size, 0);
		handle.resizeArray(size + 1);
		if (handle.getChildHandle(size, memberHandle) == NxParameterized::ERROR_NONE)
		{
			memberHandle.initParamRef(FluidIosDebugRenderParams::staticClassName(), true);
		}
	}

	/* Load reference to FluidIosDebugRenderParams */
	NxParameterized::Interface* refPtr = NULL;
	memberHandle.getParamRef(refPtr);
	mFluidIosDebugRenderParams = DYNAMIC_CAST(FluidIosDebugRenderParams*)(refPtr);
	PX_ASSERT(mFluidIosDebugRenderParams);
}

FluidIosScene::~FluidIosScene()
{
}

void FluidIosScene::destroy()
{
	removeAllActors();
	mApexScene->moduleReleased(*this);
	delete this;
}

void FluidIosScene::setModulePhysXScene(NxScene* s)
{
	if (mPhysXScene == s)
	{
		return;
	}

	mPhysXScene = s;
	for (physx::PxU32 i = 0; i < mActorArray.size(); ++i)
	{
		FluidIosActor* actor = DYNAMIC_CAST(FluidIosActor*)(mActorArray[i]);
		actor->setPhysXScene(mPhysXScene);
	}
}

void FluidIosScene::setCompartment(NxCompartment& comp)
{
	mCompartment = &comp;
}

void FluidIosScene::setSPHCompartment(NxCompartment& comp)
{
	mSPHCompartment = &comp;
}

const NxCompartment* FluidIosScene::getCompartment() const
{
	return mCompartment;
}

const NxCompartment* FluidIosScene::getSPHCompartment() const
{
	return mSPHCompartment;
}

void FluidIosScene::submitTasks(PxF32 /*elapsedTime*/, PxF32 /*substepSize*/, PxU32 /*numSubSteps*/)
{
	for (PxU32 i = 0; i < mActorArray.size(); ++i)
	{
		FluidIosActor* actor = DYNAMIC_CAST(FluidIosActor*)(mActorArray[i]);
		actor->submitTasks();
	}
}

void FluidIosScene::setTaskDependencies()
{
	for (physx::PxU32 i = 0; i < mActorArray.size(); ++i)
	{
		FluidIosActor* actor = DYNAMIC_CAST(FluidIosActor*)(mActorArray[i]);
		actor->setTaskDependencies();
	}
}

void FluidIosScene::fetchResults()
{
	PX_PROFILER_PERF_SCOPE("ParticleSceneFetchResults");

	for (physx::PxU32 i = 0; i < mActorArray.size(); ++i)
	{
		FluidIosActor* actor = DYNAMIC_CAST(FluidIosActor*)(mActorArray[i]);
		actor->fetchResults();
	}
}

physx::PxF32	FluidIosScene::getBenefit()
{
	ApexActor** ss = mActorArray.begin();
	ApexActor** ee = mActorArray.end();

	// the address of a FluidIosActor* and ApexActor* must be identical, otherwise the reinterpret cast will break
	PX_ASSERT(ss == NULL || ((void*)DYNAMIC_CAST(FluidIosActor*)(*ss) == (void*)(*ss)));

	mSumBenefit = LODCollection<FluidIosActor>::computeSumBenefit(reinterpret_cast<FluidIosActor**>(ss), reinterpret_cast<FluidIosActor**>(ee));
	return mSumBenefit;
}

physx::PxF32	FluidIosScene::setResource(physx::PxF32 suggested, physx::PxF32 maxRemaining, physx::PxF32 relativeBenefit)
{
	PX_UNUSED(maxRemaining);

	physx::PxF32 resourceUsed = LODCollection<FluidIosActor>::distributeResource(reinterpret_cast<FluidIosActor**>(mActorArray.begin()), reinterpret_cast<FluidIosActor**>(mActorArray.end()), mSumBenefit, relativeBenefit, suggested);
	return resourceUsed;
}

void FluidIosScene::visualize()
{
#ifndef WITHOUT_DEBUG_VISUALIZE
	if (!mFluidIosDebugRenderParams->VISUALIZE_FLUID_IOS_ACTOR)
	{
		return;
	}

	for (physx::PxU32 i = 0; i < mActorArray.size(); ++i)
	{
		FluidIosActor* actor = DYNAMIC_CAST(FluidIosActor*)(mActorArray[i]);
		actor->visualize();
	}
#endif
}

}
}
} // namespace physx::apex

#endif // NX_SDK_VERSION_MAJOR == 2