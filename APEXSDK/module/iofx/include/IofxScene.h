/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __IOFX_SCENE_H__
#define __IOFX_SCENE_H__

#include "NxApex.h"

#include "NiApexSDK.h"
#include "NiModule.h"
#include "ApexInterface.h"
#include "ApexContext.h"
#include "ApexSDKHelpers.h"
#include "ApexRenderVolume.h"

#include "PxGpuCopyDescQueue.h"

namespace physx
{
namespace apex
{

class NiApexScene;
class NiIofxManagerDesc;
class DebugRenderParams;

namespace iofx
{

class ModuleIofx;
class IofxDebugRenderParams;
class IofxManager;

class IofxScene : public NiModuleScene, public ApexContext, public NxApexResource, public ApexResource
{
public:
	enum StatsDataEnum
	{
		SimulatedSpriteParticlesCount,
		SimulatedMeshParticlesCount,
		// insert new items before this line
		NumberOfStats			// The number of stats
	};
public:
	IofxScene(ModuleIofx& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list);
	~IofxScene();

	/* NiModuleScene */
	void				visualize();
#if NX_SDK_VERSION_MAJOR == 2
	void				setModulePhysXScene(NxScene* s);
	NxScene*			getModulePhysXScene() const
	{
		return mPhysXScene;
	}
	NxScene*			mPhysXScene;
#elif NX_SDK_VERSION_MAJOR == 3
	void				setModulePhysXScene(PxScene* s);
	PxScene*			getModulePhysXScene() const
	{
		return mPhysXScene;
	}
	PxScene*			mPhysXScene;
#endif
	physx::PxF32		setResource(physx::PxF32, physx::PxF32, physx::PxF32)
	{
		return 0.0f;
	}
	physx::PxF32		getBenefit()
	{
		return 0.0f;
	}
	virtual NxModule*	getNxModule();

	virtual NxApexSceneStats* getStats()
	{
		return &mModuleSceneStats;
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
	physx::PxU32		getListIndex() const
	{
		return m_listIndex;
	}
	void				setListIndex(NxResourceList& list, physx::PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}
	void				release();

	virtual IofxManager* createIofxManager(const NxIofxAsset& asset, const NiIofxManagerDesc& desc);
	void				releaseIofxManager(IofxManager* manager);

	virtual bool		copyDirtySceneData(PxGpuCopyDescQueue& queue) = 0;
	void				submitTasks(PxF32 elapsedTime, PxF32 substepSize, PxU32 numSubSteps);
	void				fetchResults();

	void				fetchResultsPreRenderLock()
	{
		lockLiveRenderVolumes();
	}
	void				fetchResultsPostRenderUnlock()
	{
		unlockLiveRenderVolumes();
	}

	void				prepareRenderResources();

	PX_INLINE void		lockLiveRenderVolumes();
	PX_INLINE void		unlockLiveRenderVolumes();

	void				createModuleStats(void);
	void				destroyModuleStats(void);
	void				setStatValue(StatsDataEnum index, ApexStatValue dataVal);

	ModuleIofx*		    mModule;
	NiApexScene*		mApexScene;
	NiApexRenderDebug*	mDebugRender;

	physx::Mutex		mFetchResultsLock;
	physx::ReadWriteLock mManagersLock;

	physx::ReadWriteLock mLiveRenderVolumesLock;
	physx::Mutex		mAddedRenderVolumesLock;
	physx::Mutex		mDeletedRenderVolumesLock;
	
	NxResourceList		mActorManagers;

	physx::Array<ApexRenderVolume*> mLiveRenderVolumes;
	physx::Array<ApexRenderVolume*> mAddedRenderVolumes;
	physx::Array<ApexRenderVolume*> mDeletedRenderVolumes;

	DebugRenderParams*				mDebugRenderParams;
	IofxDebugRenderParams*			mIofxDebugRenderParams;

	NxApexSceneStats	mModuleSceneStats;	

	physx::PxU32		mPrevTotalSimulatedSpriteParticles;
	physx::PxU32		mPrevTotalSimulatedMeshParticles;

	void                destroy();

	void				processDeferredRenderVolumes();
};

}
}
} // namespace physx::apex

#endif
