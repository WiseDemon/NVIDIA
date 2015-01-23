/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __PARTICLES_SCENE_H__
#define __PARTICLES_SCENE_H__

#include "NxApex.h"
#include "PairFilter.h"
#include "ModuleParticles.h"

#include "ApexInterface.h"
#include "ApexContext.h"
#include "ApexSDKHelpers.h"

#include "NiApexRenderDebug.h"
#include "NiApexSDK.h"
#include "NiModule.h"

#include "DebugRenderParams.h"
#include "ParticlesDebugRenderParams.h"
#include "PxTask.h"

namespace physx
{

namespace apex
{

class NiApexScene;

namespace emitter
{
class NxApexEmitterActor;
class NxApexEmitterAsset;
}

namespace particles
{

class ModuleParticles;

#define EMITTER_FORGET_TIME 30

class EmitterPool
{
public:
	EmitterPool()
	{
		mEmitter = NULL;
		mEmitterTime = 0;
	}

	EmitterPool(NxApexEmitterActor* emitterActor, PxF32 simTime);

	~EmitterPool()
	{
	}

	bool process(PxF32 simTime);
	void releaseEmitter();

	NxApexEmitterActor* mEmitter;
	PxF32				mEmitterTime;
};

typedef Array< EmitterPool > EmitterPoolVector;

class ParticlesScene : public NiModuleScene, public ApexContext, public NxApexResource, public ApexResource
{
public:


	ParticlesScene(ModuleParticles& module, NiApexScene& scene, NiApexRenderDebug* renderDebug, NxResourceList& list);
	~ParticlesScene();



	/* NiModuleScene */
	void						updateActors(physx::PxF32 deltaTime);
	void						submitTasks(PxF32 elapsedTime, PxF32 substepSize, PxU32 numSubSteps);
	virtual	void				updateFromSimulate(physx::PxF32 dt);

	virtual void				visualize();
	virtual void				fetchResults();
	virtual void				fetchResultsPostRenderUnlock();
#if NX_SDK_VERSION_MAJOR == 3
	virtual void				setModulePhysXScene(PxScene* s);
	virtual PxScene*			getModulePhysXScene() const
	{
		return mPhysXScene;
	}
#else
	virtual void				setModulePhysXScene(NxScene* s);
	virtual NxScene*			getModulePhysXScene() const
	{
		return mPhysXScene;
	}
#endif
	virtual physx::PxF32		setResource(physx::PxF32, physx::PxF32, physx::PxF32)
	{
		return 0.0f;
	}
	virtual physx::PxF32		getBenefit()
	{
		return 0.0f;
	}
	virtual NxModule*			getNxModule()
	{
		return mModule;
	}

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
	PxU32						getListIndex() const
	{
		return m_listIndex;
	}
	void						setListIndex(NxResourceList& list, PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}
	virtual void				release()
	{
		mModule->releaseNiModuleScene(*this);
	}

	virtual NxApexSceneStats* 	getStats()
	{
		return NULL;
	}


	NiApexScene*                mApexScene;

	ModuleParticles* getModuleParticles() const
	{
		return mModule;
	}

	void addToEmitterPool(NxApexEmitterActor* emitterActor);
	NxApexEmitterActor* getEmitterFromPool(NxApexEmitterAsset* assert);

	void resetEmitterPool();

private:
	void						destroy();

	ModuleParticles* 				mModule;
#if NX_SDK_VERSION_MAJOR == 3
	PxScene*                    mPhysXScene;
#else
	NxScene*					mPhysXScene;
#endif
	NiApexRenderDebug* 			mRenderDebug;

	DebugRenderParams*			mDebugRenderParams;
	ParticlesDebugRenderParams*	mParticlesDebugRenderParams;

	class TaskUpdate : public physx::PxTask
	{
	public:
		TaskUpdate(ParticlesScene& owner) : mOwner(owner) {}
		const char* getName() const
		{
			return "ParticlesScene::Update";
		}
		void run();

	protected:
		ParticlesScene& mOwner;

	private:
		TaskUpdate& operator=(const TaskUpdate&);
	};

	TaskUpdate						mUpdateTask;
	friend class ModuleParticles;
	friend class ParticlesActor;
	friend class TaskUpdate;


	PxF32								mCheckTime;
	PxF32								mSimTime;
	EmitterPoolVector					mEmitterPool;
};

}
}
} // end namespace physx::apex

#endif
