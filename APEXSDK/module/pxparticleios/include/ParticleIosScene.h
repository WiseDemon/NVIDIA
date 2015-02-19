/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __PARTICLE_IOS_SCENE_H__
#define __PARTICLE_IOS_SCENE_H__

#include "NxApex.h"
#include "NxModuleParticleIos.h"
#include "NiApexSDK.h"
#include "NiModule.h"
#include "ModuleParticleIos.h"
#include "ApexSharedUtils.h"
#include "ApexSDKHelpers.h"
#include "ApexContext.h"
#include "ApexActor.h"
#include "ModulePerfScope.h"

#include "PsTime.h"

#include "DebugRenderParams.h"
#include "ParticleIosDebugRenderParams.h"

#include "ParticleIosCommon.h"

#include "NiFieldSamplerQuery.h"

#if defined(APEX_CUDA_SUPPORT)
#include "ApexCudaWrapper.h"
#include "ApexCuda.h"
#include "CudaModuleScene.h"

#include "../cuda/include/common.h"

#define SCENE_CUDA_OBJ(scene, name) static_cast<ParticleIosSceneGPU&>(scene).APEX_CUDA_OBJ_NAME(name)

#endif

namespace physx
{
namespace apex
{

class NiApexRenderDebug;
class NiFieldSamplerManager;

namespace pxparticleios
{

class ParticleIosInjectorStorage
{
public:
	virtual bool growInjectorStorage(physx::PxU32 newSize) = 0;
};
class ParticleIosInjectorAllocator
{
public:
	ParticleIosInjectorAllocator(ParticleIosInjectorStorage* storage) : mStorage(storage)
	{
		mFreeInjectorListStart = NULL_INJECTOR_INDEX;
		mReleasedInjectorListStart = NULL_INJECTOR_INDEX;
	}

	physx::PxU32				allocateInjectorID();
	void						releaseInjectorID(physx::PxU32);
	void						flushReleased();

	static const PxU32			NULL_INJECTOR_INDEX = 0xFFFFFFFFu;
	static const PxU32			USED_INJECTOR_INDEX = 0xFFFFFFFEu;

private:
	ParticleIosInjectorStorage*	mStorage;

	physx::Array<PxU32>			mInjectorList;
	PxU32						mFreeInjectorListStart;
	PxU32						mReleasedInjectorListStart;
};


class ParticleIosScene : public NiModuleScene, public ApexContext, public NxApexResource, public ApexResource, protected ParticleIosInjectorStorage
{
public:
	ParticleIosScene(ModuleParticleIos& module, NiApexScene& scene, NiApexRenderDebug* renderDebug, NxResourceList& list);
	~ParticleIosScene();

	/* NiModuleScene */
	void									release()
	{
		mModule->releaseNiModuleScene(*this);
	}

#if NX_SDK_VERSION_MAJOR == 2
	NxScene*                                getModulePhysXScene() const
	{
		return mPhysXScene;
	}
	void                                    setModulePhysXScene(NxScene*);
	NxScene* 								mPhysXScene;
#elif NX_SDK_VERSION_MAJOR == 3
	PxScene*                                getModulePhysXScene() const
	{
		return mPhysXScene;
	}
	void                                    setModulePhysXScene(PxScene*);
	PxScene* 								mPhysXScene;
#endif

	void									visualize();

	virtual NxModule*						getNxModule()
	{
		return mModule;
	}

	virtual NxApexSceneStats* getStats()
	{
		return 0;
	}

	physx::PxF32							getBenefit();
	physx::PxF32							setResource(physx::PxF32 , physx::PxF32, physx::PxF32);

	bool							lockRenderResources()
	{
		renderLockAllActors();	// Lock options not implemented yet
		return true;
	}

	bool							unlockRenderResources()
	{
		renderUnLockAllActors();	// Lock options not implemented yet
		return true;
	}

	/* NxApexResource */
	physx::PxU32							getListIndex() const
	{
		return m_listIndex;
	}
	void                                    setListIndex(NxResourceList& list, physx::PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}

	virtual ParticleIosActor*				createIosActor(NxResourceList& list, ParticleIosAsset& asset, NxIofxAsset& iofxAsset) = 0;

	void									submitTasks(PxF32 elapsedTime, PxF32 substepSize, PxU32 numSubSteps);
	void									setTaskDependencies();
	void									fetchResults();

	NiFieldSamplerManager* 					getNiFieldSamplerManager();
	NiApexScene&							getApexScene() const
	{
		return *mApexScene;
	}
	PX_INLINE ParticleIosInjectorAllocator&	getInjectorAllocator()
	{
		return mInjectorAllocator;
	}
	virtual void							fetchInjectorParams(PxU32 injectorID, Px3InjectorParams& injParams) = 0;
	virtual void							updateInjectorParams(PxU32 injectorID, const Px3InjectorParams& injParams) = 0;

protected:
	virtual void onSimulationStart() {}
	virtual void onSimulationFinish()
	{
		mInjectorAllocator.flushReleased();
	}

	void									destroy();
	physx::PxF32							computeAABBDistanceSquared(const physx::PxBounds3& aabb);

	ModuleParticleIos* 						mModule;
	NiApexScene* 							mApexScene;

	NiApexRenderDebug* 						mRenderDebug;
	physx::PxF32							mSumBenefit;

	DebugRenderParams* 						mDebugRenderParams;
	ParticleIosDebugRenderParams* 			mParticleIosDebugRenderParams;

	NiFieldSamplerManager* 					mFieldSamplerManager;

	ParticleIosInjectorAllocator			mInjectorAllocator;

	friend class ParticleIosActor;
	friend class ParticleIosAsset;
	friend class ModuleParticleIos;
};

class ParticleIosSceneCPU : public ParticleIosScene
{
	class TimerCallback : public NiFieldSamplerCallback, public NiIofxManagerCallback, public physx::UserAllocated
	{
		physx::shdfnd::Time mTimer;
		PxReal mMinTime, mMaxTime;
	public:
		TimerCallback() {}		
		void operator()(void* stream = NULL);
		void reset();
		PxReal getElapsedTime() const;
	};
public:
	ParticleIosSceneCPU(ModuleParticleIos& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list);
	~ParticleIosSceneCPU();

	virtual ParticleIosActor*	createIosActor(NxResourceList& list, ParticleIosAsset& asset, NxIofxAsset& iofxAsset);

	virtual void				fetchInjectorParams(PxU32 injectorID, Px3InjectorParams& injParams)
	{
		PX_ASSERT(injectorID < mInjectorParamsArray.size());
		injParams = mInjectorParamsArray[ injectorID ];
	}
	virtual void				updateInjectorParams(PxU32 injectorID, const Px3InjectorParams& injParams)
	{
		PX_ASSERT(injectorID < mInjectorParamsArray.size());
		mInjectorParamsArray[ injectorID ] = injParams;
	}

	void							fetchResults();

protected:
	virtual bool growInjectorStorage(physx::PxU32 newSize)
	{
		mInjectorParamsArray.resize(newSize);
		return true;
	}

private:
	physx::Array<Px3InjectorParams> mInjectorParamsArray;
	TimerCallback					mTimerCallback;

	friend class ParticleIosActorCPU;
};

#if defined(APEX_CUDA_SUPPORT)
class ParticleIosSceneGPU : public ParticleIosScene, public CudaModuleScene
{
	class EventCallback : public NiFieldSamplerCallback, public NiIofxManagerCallback, public physx::UserAllocated
	{
		void* mEvent;
	public:
		EventCallback();
		void init();
		virtual ~EventCallback();
		void operator()(void* stream);
		PX_INLINE void* getEvent()
		{
			return mEvent;
		}
		bool mIsCalled;
	};
public:
	ParticleIosSceneGPU(ModuleParticleIos& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list);
	~ParticleIosSceneGPU();

	virtual ParticleIosActor*		createIosActor(NxResourceList& list, ParticleIosAsset& asset, NxIofxAsset& iofxAsset);

	virtual void					fetchInjectorParams(PxU32 injectorID, Px3InjectorParams& injParams);
	virtual void					updateInjectorParams(PxU32 injectorID, const Px3InjectorParams& injParams);

	void							fetchResults();

	void*							getHeadCudaObj()
	{
		return CudaModuleScene::getHeadCudaObj();
	}
//CUDA module objects
#include "../cuda/include/moduleList.h"

protected:
	virtual bool growInjectorStorage(physx::PxU32 newSize);

	void onSimulationStart();

private:
	ApexCudaConstMemGroup				mInjectorConstMemGroup;
	InplaceHandle<InjectorParamsArray>	mInjectorParamsArrayHandle;

	EventCallback						mOnSimulationStart;
	physx::Array<EventCallback*>		mOnStartCallbacks;
	physx::Array<EventCallback*>		mOnFinishCallbacks;

	friend class ParticleIosActorGPU;
};
#endif

}
}
} // namespace physx::apex

#endif // __PARTICLE_IOS_SCENE_H__
