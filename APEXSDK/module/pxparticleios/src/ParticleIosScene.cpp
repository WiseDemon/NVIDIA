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

#include "NxApex.h"
#include "ParticleIosScene.h"
#include "ModuleParticleIos.h"
#include "ParticleIosActor.h"
#include "ParticleIosActorCPU.h"
#include "NiApexScene.h"
#include "NiModuleFieldSampler.h"
#include "ModulePerfScope.h"
#include "PsShare.h"
#include "NiApexRenderDebug.h"


#if defined(APEX_CUDA_SUPPORT)
#include <cuda.h>
#include "ApexCutil.h"
#include "ParticleIosActorGPU.h"
#include "ApexCudaSource.h"
#endif

#include "NxLock.h"

#define CUDA_OBJ(name) SCENE_CUDA_OBJ(*this, name)

namespace physx
{
namespace apex
{
namespace pxparticleios
{


#pragma warning(push)
#pragma warning(disable:4355)

ParticleIosScene::ParticleIosScene(ModuleParticleIos& _module, NiApexScene& scene, NiApexRenderDebug* renderDebug, NxResourceList& list)
	: mPhysXScene(NULL)
	, mModule(&_module)
	, mApexScene(&scene)
	, mRenderDebug(renderDebug)
	, mSumBenefit(0.0f)
	, mFieldSamplerManager(NULL)
	, mInjectorAllocator(this)
{
	list.add(*this);

	/* Initialize reference to ParticleIosDebugRenderParams */
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
			memberHandle.initParamRef(ParticleIosDebugRenderParams::staticClassName(), true);
		}
	}

	/* Load reference to ParticleIosDebugRenderParams */
	NxParameterized::Interface* refPtr = NULL;
	memberHandle.getParamRef(refPtr);
	mParticleIosDebugRenderParams = DYNAMIC_CAST(ParticleIosDebugRenderParams*)(refPtr);
	PX_ASSERT(mParticleIosDebugRenderParams);
}
#pragma warning(pop)

ParticleIosScene::~ParticleIosScene()
{
}

void ParticleIosScene::destroy()
{
	removeAllActors();
	mApexScene->moduleReleased(*this);
	delete this;
}

#if NX_SDK_VERSION_MAJOR == 2
void ParticleIosScene::setModulePhysXScene(NxScene* s)
#elif NX_SDK_VERSION_MAJOR == 3
void ParticleIosScene::setModulePhysXScene(PxScene* s)
#endif
{
	if (mPhysXScene == s)
	{
		return;
	}

	mPhysXScene = s;
	for (physx::PxU32 i = 0; i < mActorArray.size(); ++i)
	{
		ParticleIosActor* actor = DYNAMIC_CAST(ParticleIosActor*)(mActorArray[i]);
		actor->setPhysXScene(mPhysXScene);
	}
}

void ParticleIosScene::visualize()
{
#ifndef WITHOUT_DEBUG_VISUALIZE
	if (!mParticleIosDebugRenderParams->VISUALIZE_PARTICLE_IOS_ACTOR)
	{
		return;
	}

	mRenderDebug->pushRenderState();
	for (PxU32 i = 0 ; i < mActorArray.size() ; i++)
	{
		ParticleIosActor* testActor = DYNAMIC_CAST(ParticleIosActor*)(mActorArray[ i ]);
		testActor->visualize();
	}
	mRenderDebug->popRenderState();
#endif
}

physx::PxF32	ParticleIosScene::getBenefit()
{
	ApexActor** ss = mActorArray.begin();
	ApexActor** ee = mActorArray.end();

	// the address of a ParticleIosActor* and ApexActor* must be identical, otherwise the reinterpret cast will break
	PX_ASSERT(ss == NULL || ((void*)DYNAMIC_CAST(ParticleIosActor*)(*ss) == (void*)(*ss)));

	mSumBenefit = LODCollection<ParticleIosActor>::computeSumBenefit(reinterpret_cast<ParticleIosActor**>(ss), reinterpret_cast<ParticleIosActor**>(ee));
	return mSumBenefit;
}

physx::PxF32	ParticleIosScene::setResource(physx::PxF32 suggested, physx::PxF32 maxRemaining, physx::PxF32 relativeBenefit)
{
	PX_UNUSED(maxRemaining);

	physx::PxF32 resourceUsed = LODCollection<ParticleIosActor>::distributeResource(reinterpret_cast<ParticleIosActor**>(mActorArray.begin()), reinterpret_cast<ParticleIosActor**>(mActorArray.end()), mSumBenefit, relativeBenefit, suggested);
	return resourceUsed;
}

void ParticleIosScene::submitTasks(PxF32 /*elapsedTime*/, PxF32 /*substepSize*/, PxU32 /*numSubSteps*/)
{
	physx::PxTaskManager* tm;
	{
		NX_READ_LOCK(*mApexScene);
		tm = mApexScene->getTaskManager();
	}

	for (PxU32 i = 0; i < mActorArray.size(); ++i)
	{
		ParticleIosActor* actor = DYNAMIC_CAST(ParticleIosActor*)(mActorArray[i]);
		actor->submitTasks(tm);
	}
}

void ParticleIosScene::setTaskDependencies()
{
	physx::PxTaskManager* tm;
	{
		NX_READ_LOCK(*mApexScene);
		tm	= mApexScene->getTaskManager();
	}
#if 0
	//run IOS after PhysX
	physx::PxTaskID		taskStartAfterID	= tm->getNamedTask(AST_PHYSX_CHECK_RESULTS);
	physx::PxTaskID		taskFinishBeforeID	= 0;
#else
	//run IOS before PhysX
	physx::PxTaskID		taskStartAfterID	= 0;
	physx::PxTaskID		taskFinishBeforeID	= tm->getNamedTask(AST_PHYSX_SIMULATE);
#endif

	for (physx::PxU32 i = 0; i < mActorArray.size(); ++i)
	{
		ParticleIosActor* actor = DYNAMIC_CAST(ParticleIosActor*)(mActorArray[i]);
		actor->setTaskDependencies(taskStartAfterID, taskFinishBeforeID);
	}

	onSimulationStart();
}

void ParticleIosScene::fetchResults()
{
	onSimulationFinish();

	for (physx::PxU32 i = 0; i < mActorArray.size(); ++i)
	{
		ParticleIosActor* actor = DYNAMIC_CAST(ParticleIosActor*)(mActorArray[i]);
		actor->fetchResults();
	}
}

NiFieldSamplerManager* ParticleIosScene::getNiFieldSamplerManager()
{
	if (mFieldSamplerManager == NULL)
	{
		NiModuleFieldSampler* moduleFieldSampler = mModule->getNiModuleFieldSampler();
		if (moduleFieldSampler != NULL)
		{
			mFieldSamplerManager = moduleFieldSampler->getNiFieldSamplerManager(*mApexScene);
			PX_ASSERT(mFieldSamplerManager != NULL);
		}
	}
	return mFieldSamplerManager;
}

/******************************** CPU Version ********************************/

void ParticleIosSceneCPU::TimerCallback::operator()(void* stream)
{
	PX_UNUSED(stream);

	PxReal elapsed = (PxReal)mTimer.peekElapsedSeconds();
	mMinTime = PxMin(elapsed, mMinTime);
	mMaxTime = PxMax(elapsed, mMaxTime);
}

void ParticleIosSceneCPU::TimerCallback::reset()
{
	mTimer.getElapsedSeconds();
	mMinTime = 1e20;
	mMaxTime = 0.f;
}

PxF32 ParticleIosSceneCPU::TimerCallback::getElapsedTime() const
{
	return (mMaxTime - mMinTime) * 1000.f;
}

ParticleIosSceneCPU::ParticleIosSceneCPU(ModuleParticleIos& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list) :
	ParticleIosScene(module, scene, debugRender, list)
{
}

ParticleIosSceneCPU::~ParticleIosSceneCPU()
{
}

ParticleIosActor* ParticleIosSceneCPU::createIosActor(NxResourceList& list, ParticleIosAsset& asset, NxIofxAsset& iofxAsset)
{
	ParticleIosActorCPU* actor = PX_NEW(ParticleIosActorCPU)(list, asset, *this, iofxAsset);

	actor->setOnStartFSCallback(&mTimerCallback);
	actor->setOnFinishIOFXCallback(&mTimerCallback);
	return actor;
}

void ParticleIosSceneCPU::fetchResults()
{
	ParticleIosScene::fetchResults();

	physx::apex::ApexStatValue val;
	val.Float = mTimerCallback.getElapsedTime();
	mTimerCallback.reset();
	if (val.Float > 0.f)
	{
		mApexScene->setApexStatValue(NiApexScene::ParticleSimulationTime, val);
	}
}

/******************************** GPU Version ********************************/

#if defined(APEX_CUDA_SUPPORT)

ParticleIosSceneGPU::EventCallback::EventCallback() : mIsCalled(false), mEvent(NULL)
{
}
void ParticleIosSceneGPU::EventCallback::init()
{
	if (mEvent == NULL)
	{
		CUT_SAFE_CALL(cuEventCreate((CUevent*)(&mEvent), CU_EVENT_DEFAULT));
	}
}

ParticleIosSceneGPU::EventCallback::~EventCallback()
{
	if (mEvent != NULL)
	{
		CUT_SAFE_CALL(cuEventDestroy((CUevent)mEvent));
	}
}

void ParticleIosSceneGPU::EventCallback::operator()(void* stream)
{
	if (mEvent != NULL)
	{
		CUT_SAFE_CALL(cuEventRecord((CUevent)mEvent, (CUstream)stream));
		mIsCalled = true;
	}
}

ParticleIosSceneGPU::ParticleIosSceneGPU(ModuleParticleIos& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list)
	: ParticleIosScene(module, scene, debugRender, list)
	, CudaModuleScene(scene, *mModule, APEX_CUDA_TO_STR(APEX_CUDA_MODULE_PREFIX))
	, mInjectorConstMemGroup(APEX_CUDA_OBJ_NAME(simulateStorage))
{
	{
		physx::PxGpuDispatcher* gd = mApexScene->getTaskManager()->getGpuDispatcher();
		PX_ASSERT(gd != NULL);
		physx::PxScopedCudaLock _lock_(*gd->getCudaContextManager());

		mOnSimulationStart.init();
//CUDA module objects
#include "../cuda/include/moduleList.h"
	}

	{
		mInjectorConstMemGroup.begin();
		mInjectorParamsArrayHandle.alloc(mInjectorConstMemGroup.getStorage());
		//injectorParamsArray.resize( mInjectorConstMemGroup.getStorage(), MAX_INJECTOR_COUNT );
		mInjectorConstMemGroup.end();
	}

}

ParticleIosSceneGPU::~ParticleIosSceneGPU()
{
	for (physx::PxU32 i = 0; i < mOnStartCallbacks.size(); i++)
	{
		PX_DELETE(mOnStartCallbacks[i]);
	}
	for (physx::PxU32 i = 0; i < mOnFinishCallbacks.size(); i++)
	{
		PX_DELETE(mOnFinishCallbacks[i]);
	}
	CudaModuleScene::destroy(*mApexScene);
}

ParticleIosActor* ParticleIosSceneGPU::createIosActor(NxResourceList& list, ParticleIosAsset& asset, NxIofxAsset& iofxAsset)
{
	ParticleIosActorGPU* actor = PX_NEW(ParticleIosActorGPU)(list, asset, *this, iofxAsset);
	mOnStartCallbacks.pushBack(PX_NEW(EventCallback)());
	mOnFinishCallbacks.pushBack(PX_NEW(EventCallback)());
	{
		physx::PxGpuDispatcher* gd = mApexScene->getTaskManager()->getGpuDispatcher();
		PX_ASSERT(gd != NULL);
		physx::PxScopedCudaLock _lock_(*gd->getCudaContextManager());

		mOnStartCallbacks.back()->init();
		mOnFinishCallbacks.back()->init();
	}
	actor->setOnStartFSCallback(mOnStartCallbacks.back());
	actor->setOnFinishIOFXCallback(mOnFinishCallbacks.back());
	return actor;
}

void ParticleIosSceneGPU::fetchInjectorParams(PxU32 injectorID, Px3InjectorParams& injParams)
{
	APEX_CUDA_CONST_MEM_GROUP_SCOPE(mInjectorConstMemGroup);

	InjectorParamsArray injectorParamsArray;
	mInjectorParamsArrayHandle.fetch(_storage_, injectorParamsArray);
	PX_ASSERT(injectorID < injectorParamsArray.getSize());
	injectorParamsArray.fetchElem(_storage_, injParams, injectorID);
}
void ParticleIosSceneGPU::updateInjectorParams(PxU32 injectorID, const Px3InjectorParams& injParams)
{
	APEX_CUDA_CONST_MEM_GROUP_SCOPE(mInjectorConstMemGroup);

	InjectorParamsArray injectorParamsArray;
	mInjectorParamsArrayHandle.fetch(_storage_, injectorParamsArray);
	PX_ASSERT(injectorID < injectorParamsArray.getSize());
	injectorParamsArray.updateElem(_storage_, injParams, injectorID);
}

void ParticleIosSceneGPU::fetchResults()
{
	ParticleIosScene::fetchResults();

	physx::apex::ApexStatValue val;	
	val.Float = 0.f;
	PxF32 minTime = 1e30;
	
	for (physx::PxU32 i = 0 ; i < this->mOnStartCallbacks.size(); i++)
	{
		if (mOnStartCallbacks[i]->mIsCalled && mOnFinishCallbacks[i]->mIsCalled)
		{
			mOnStartCallbacks[i]->mIsCalled = false;
			mOnFinishCallbacks[i]->mIsCalled = false;
			CUT_SAFE_CALL(cuEventSynchronize((CUevent)mOnStartCallbacks[i]->getEvent()));
			CUT_SAFE_CALL(cuEventSynchronize((CUevent)mOnFinishCallbacks[i]->getEvent()));
			PxF32 tmp;
			CUT_SAFE_CALL(cuEventElapsedTime(&tmp, (CUevent)mOnSimulationStart.getEvent(), (CUevent)mOnStartCallbacks[i]->getEvent()));
			minTime = physx::PxMin(tmp, minTime);
			CUT_SAFE_CALL(cuEventElapsedTime(&tmp, (CUevent)mOnSimulationStart.getEvent(), (CUevent)mOnFinishCallbacks[i]->getEvent()));
			val.Float = physx::PxMax(tmp, val.Float);
		}
	}
	val.Float -= physx::PxMin(minTime, val.Float);	
	
	if (val.Float > 0.f)
	{
		mApexScene->setApexStatValue(NiApexScene::ParticleSimulationTime, val);
	}
}

bool ParticleIosSceneGPU::growInjectorStorage(physx::PxU32 newSize)
{
	APEX_CUDA_CONST_MEM_GROUP_SCOPE(mInjectorConstMemGroup);

	InjectorParamsArray injectorParamsArray;
	mInjectorParamsArrayHandle.fetch(_storage_, injectorParamsArray);
	if (injectorParamsArray.resize(_storage_, newSize))
	{
		mInjectorParamsArrayHandle.update(_storage_, injectorParamsArray);
		return true;
	}
	return false;
}


void ParticleIosSceneGPU::onSimulationStart()
{
	ParticleIosScene::onSimulationStart();

	physx::PxGpuDispatcher* gd = mApexScene->getTaskManager()->getGpuDispatcher();
	PX_ASSERT(gd != NULL);
	physx::PxScopedCudaLock _lock_(*gd->getCudaContextManager());

	//we pass default 0 stream so that this copy happens before any kernel launches
	APEX_CUDA_OBJ_NAME(simulateStorage).copyToDevice(gd->getCudaContextManager(), 0);

	mOnSimulationStart(NULL);
}

#endif

// ParticleIosInjectorAllocator
physx::PxU32 ParticleIosInjectorAllocator::allocateInjectorID()
{
	if (mFreeInjectorListStart == NULL_INJECTOR_INDEX)
	{
		//try to get new injectors
		physx::PxU32 size = mInjectorList.size();
		if (mStorage->growInjectorStorage(size + 1) == false)
		{
			return NULL_INJECTOR_INDEX;
		}

		mFreeInjectorListStart = size;
		mInjectorList.resize(size + 1);
		mInjectorList.back() = NULL_INJECTOR_INDEX;
	}
	physx::PxU32 injectorID = mFreeInjectorListStart;
	mFreeInjectorListStart = mInjectorList[injectorID];
	mInjectorList[injectorID] = USED_INJECTOR_INDEX;
	return injectorID;
}

void ParticleIosInjectorAllocator::releaseInjectorID(physx::PxU32 injectorID)
{
	//add to released injector list
	PX_ASSERT(mInjectorList[injectorID] == USED_INJECTOR_INDEX);
	mInjectorList[injectorID] = mReleasedInjectorListStart;
	mReleasedInjectorListStart = injectorID;
}

void ParticleIosInjectorAllocator::flushReleased()
{
	//add all released injectors to free injector list
	while (mReleasedInjectorListStart != NULL_INJECTOR_INDEX)
	{
		physx::PxU32 injectorID = mInjectorList[mReleasedInjectorListStart];

		//add to free injector list
		mInjectorList[mReleasedInjectorListStart] = mFreeInjectorListStart;
		mFreeInjectorListStart = mReleasedInjectorListStart;

		mReleasedInjectorListStart = injectorID;
	}
}

}
}
} // namespace physx::apex

#endif // NX_SDK_VERSION_MAJOR == 3