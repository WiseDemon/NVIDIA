/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FIELD_SAMPLER_SCENE_H__
#define __FIELD_SAMPLER_SCENE_H__

#include "NxApex.h"

#include "ModuleFieldSampler.h"
#include "NiFieldSamplerScene.h"
#include "NiFieldSamplerQuery.h"
#include "NiApexSDK.h"
#include "NiModule.h"
#include "ApexInterface.h"
#include "ApexContext.h"
#include "ApexSDKHelpers.h"
#include "PsArray.h"

#include "PxTask.h"

#if defined(APEX_CUDA_SUPPORT)
#include "ApexCudaWrapper.h"
#include "ApexCuda.h"
#include "CudaModuleScene.h"

#include "../cuda/include/common.h"

#define SCENE_CUDA_OBJ(scene, name) static_cast<FieldSamplerSceneGPU*>(scene)->APEX_CUDA_OBJ_NAME(name)
#endif


namespace physx
{
namespace apex
{
class NiApexScene;

namespace fieldsampler
{

class ModuleFieldSampler;
class FieldSamplerPhysXMonitor;
class FieldSamplerManager;
class FieldSamplerQuery;

class FieldSamplerScene : public NiModuleScene, public ApexContext, public NxApexResource, public ApexResource
{
public:
	FieldSamplerScene(ModuleFieldSampler& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list);
	~FieldSamplerScene();

	/* NiModuleScene */
	void				visualize();

#if NX_SDK_VERSION_MAJOR == 2
	NxScene*			getModulePhysXScene() const
	{
		return mPhysXScene;
	}
	void				setModulePhysXScene(NxScene*);
	NxScene* 			mPhysXScene;
#elif NX_SDK_VERSION_MAJOR == 3
	PxScene*			getModulePhysXScene() const
	{
		return mPhysXScene;
	}
	void				setModulePhysXScene(PxScene*);
	PxScene* 			mPhysXScene;
#endif

	PxReal				setResource(PxReal, PxReal, PxReal)
	{
		return 0.0f;
	}
	PxReal				getBenefit()
	{
		return 0.0f;
	}

	void				submitTasks(PxF32 elapsedTime, PxF32 substepSize, PxU32 numSubSteps);
	void				setTaskDependencies();
	void				fetchResults();

	virtual NxModule*	getNxModule()
	{
		return mModule;
	}

	virtual NxApexSceneStats* getStats()
	{
		return 0;
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
	PxU32				getListIndex() const
	{
		return m_listIndex;
	}
	void				setListIndex(NxResourceList& list, PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}
	void				release()
	{
		mModule->releaseNiModuleScene(*this);
	}
	NiApexScene& getApexScene() const
	{
		return *mApexScene;
	}

	NiFieldSamplerManager* getManager();

#if NX_SDK_VERSION_MAJOR == 3
	PxU32 createForceSampleBatch(PxU32 maxCount, const physx::PxFilterData filterData);
	void releaseForceSampleBatch(PxU32 batchId);
	void submitForceSampleBatch(	PxU32 batchId, PxVec4* forces, const PxU32 forcesStride,
									const PxVec3* positions, const PxU32 positionsStride,
									const PxVec3* velocities, const PxU32 velocitiesStride,
									const PxF32* mass, const PxU32 massStride,
									const PxU32* indices, const PxU32 numIndices);

	/* Toggle PhysX Monitor on/off */
	void enablePhysXMonitor(bool enable);

	void setPhysXFilterData(physx::PxFilterData filterData);
#endif

protected:
	void                destroy();

	virtual FieldSamplerManager* createManager() = 0;

	class TaskPhysXMonitorLoad : public physx::PxTask
	{
	public:
		TaskPhysXMonitorLoad() {}
		const char* getName() const
		{
			return FSST_PHYSX_MONITOR_LOAD;
		}		
		void run() {/* void task */};
	};
	TaskPhysXMonitorLoad	mPhysXMonitorLoadTask;
	class TaskPhysXMonitorFetch : public physx::PxTask
	{
	public:
		TaskPhysXMonitorFetch() {}
		const char* getName() const
		{
			return FSST_PHYSX_MONITOR_FETCH;
		}		
		void run() {/* void task */};
	};
	TaskPhysXMonitorFetch	mPhysXMonitorFetchTask;

	ModuleFieldSampler*		mModule;
	NiApexScene*			mApexScene;
	NiApexRenderDebug*		mDebugRender;
	FieldSamplerPhysXMonitor* mPhysXMonitor;

	FieldSamplerManager*	mManager;

	PxU32							mForceSampleBatchBufferSize;
	PxU32							mForceSampleBatchBufferPos;
	Array <FieldSamplerQuery*>		mForceSampleBatchQuery;
	Array <NiFieldSamplerQueryData> mForceSampleBatchQueryData;
	Array <PxVec4>					mForceSampleBatchPosition;
	Array <PxVec4>					mForceSampleBatchVelocity;
	Array <PxF32>					mForceSampleBatchMass;

	friend class ModuleFieldSampler;
	friend class FieldSamplerManager;
};

class FieldSamplerSceneCPU : public FieldSamplerScene
{
public:
	FieldSamplerSceneCPU(ModuleFieldSampler& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list);
	~FieldSamplerSceneCPU();

protected:
	virtual FieldSamplerManager* createManager();

};

#if defined(APEX_CUDA_SUPPORT)
class FieldSamplerSceneGPU : public FieldSamplerScene, public CudaModuleScene
{
public:
	FieldSamplerSceneGPU(ModuleFieldSampler& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list);
	~FieldSamplerSceneGPU();

	void* getHeadCudaObj()
	{
		return CudaModuleScene::getHeadCudaObj();
	}

//CUDA module objects
#include "../cuda/include/fieldsampler.h"

	PxCudaContextManager* getCudaContext() const
	{
		return mCtxMgr;
	}

protected:
	virtual FieldSamplerManager* createManager();

	/* keep a convenience pointer to the cuda context manager */
	PxCudaContextManager* mCtxMgr;
};
#endif

}
}
} // end namespace physx::apex

#endif
