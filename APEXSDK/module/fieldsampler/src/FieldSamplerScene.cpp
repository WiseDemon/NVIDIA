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

#include "FieldSamplerScene.h"
#include "FieldSamplerManager.h"
#include "FieldSamplerQuery.h"
#include "FieldSamplerPhysXMonitor.h"
#include "NiApexScene.h"
#include "NiApexRenderDebug.h"
#include "ModulePerfScope.h"

#include "NxFromPx.h"


#if defined(APEX_CUDA_SUPPORT)
#include "PxGpuTask.h"
#include "ApexCudaSource.h"
#endif

#include "NxLock.h"

namespace physx
{
namespace apex
{
namespace fieldsampler
{

FieldSamplerScene::FieldSamplerScene(ModuleFieldSampler& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list)
	: mModule(&module)
	, mApexScene(&scene)
	, mDebugRender(debugRender)
	, mManager(NULL)
	, mPhysXScene(NULL)
	, mForceSampleBatchBufferPos(0)
	, mForceSampleBatchBufferSize(0)
{
	list.add(*this);		// Add self to module's list of FieldSamplerScenes
}

FieldSamplerScene::~FieldSamplerScene()
{
}

void FieldSamplerScene::visualize()
{
#ifndef WITHOUT_DEBUG_VISUALIZE
	mDebugRender->pushRenderState();
	// This is using the new debug rendering
	mDebugRender->popRenderState();
#endif
}

void FieldSamplerScene::destroy()
{
#if NX_SDK_VERSION_MAJOR == 3
	PX_DELETE(mPhysXMonitor);
#endif
	PX_DELETE(mManager);

	removeAllActors();
	mApexScene->moduleReleased(*this);
	delete this;
}

NiFieldSamplerManager* FieldSamplerScene::getManager()
{
	if (mManager == NULL)
	{
		mManager = createManager();
		PX_ASSERT(mManager != NULL);
	}
	return mManager;
}


#if NX_SDK_VERSION_MAJOR == 2
void FieldSamplerScene::setModulePhysXScene(NxScene* /*s*/)
{
}
#elif NX_SDK_VERSION_MAJOR == 3
void FieldSamplerScene::setModulePhysXScene(PxScene* s)
{
	if (s)
	{
		mPhysXMonitor->setPhysXScene(s);
	}
	mPhysXScene = s;
}
#endif


void FieldSamplerScene::submitTasks(PxF32 /*elapsedTime*/, PxF32 /*substepSize*/, PxU32 /*numSubSteps*/)
{
#if NX_SDK_VERSION_MAJOR == 3
	physx::PxTaskManager* tm;
	{
		NX_READ_LOCK(*mApexScene);
		tm = mApexScene->getTaskManager();
	}
	tm->submitNamedTask(&mPhysXMonitorFetchTask, FSST_PHYSX_MONITOR_FETCH);
	tm->submitNamedTask(&mPhysXMonitorLoadTask, FSST_PHYSX_MONITOR_LOAD);
#endif
	if (mManager != NULL)
	{
		mManager->submitTasks();
	}
}

void FieldSamplerScene::setTaskDependencies()
{
#if NX_SDK_VERSION_MAJOR == 3
	for (PxU32 i = 0; i < mForceSampleBatchQuery.size(); i++)
	{
		if (mForceSampleBatchQuery[i] && mForceSampleBatchQueryData[i].count > 0)
		{
			static_cast<NiFieldSamplerQuery*>(mForceSampleBatchQuery[i])->submitFieldSamplerQuery(mForceSampleBatchQueryData[i], mApexScene->getTaskManager()->getNamedTask(FSST_PHYSX_MONITOR_LOAD));
			mForceSampleBatchQueryData[i].count = 0;
		}
		mForceSampleBatchBufferPos = 0;
	}
	if (mPhysXMonitor->isEnable())
	{
		mPhysXMonitor->update();
	}
#endif
	if (mManager != NULL)
	{
		mManager->setTaskDependencies();
	}

#if NX_SDK_VERSION_MAJOR == 3
	// Just in case one of the scene conditions doesn't set a bounding dependency, let's not let these dangle
	PxTaskManager* tm;
	{
		NX_READ_LOCK(*mApexScene);
		tm = mApexScene->getTaskManager();
	}
	mPhysXMonitorFetchTask.finishBefore(tm->getNamedTask(AST_PHYSX_FETCH_RESULTS));
	mPhysXMonitorLoadTask.finishBefore(tm->getNamedTask(AST_PHYSX_FETCH_RESULTS));
#endif
}

void FieldSamplerScene::fetchResults()
{
	if (mManager != NULL)
	{
		mManager->fetchResults();
	}
}

#if NX_SDK_VERSION_MAJOR == 3
void FieldSamplerScene::enablePhysXMonitor(bool enable)
{
	PX_UNUSED(enable);
	mPhysXMonitor->enablePhysXMonitor(enable);
}

void FieldSamplerScene::setPhysXFilterData(physx::PxFilterData filterData)
{
	mPhysXMonitor->setPhysXFilterData(filterData);
}


PxU32 FieldSamplerScene::createForceSampleBatch(PxU32 maxCount, const physx::PxFilterData filterData)
{
	mForceSampleBatchBufferSize += maxCount;
	mForceSampleBatchPosition.resize(mForceSampleBatchBufferSize);
	mForceSampleBatchVelocity.resize(mForceSampleBatchBufferSize);
	mForceSampleBatchMass.resize(mForceSampleBatchBufferSize);

	NiFieldSamplerQueryDesc desc;
	desc.maxCount = maxCount;
	desc.samplerFilterData = filterData;
	//SceneInfo* sceneInfo = DYNAMIC_CAST(SceneInfo*)(mSceneList.getResource(i));
	//NiFieldSamplerScene* niFieldSamplerScene = sceneInfo->getSceneWrapper()->getNiFieldSamplerScene();
	//desc.ownerFieldSamplerScene = this;
	

	PxU32 id = 0;
	while (id < mForceSampleBatchQuery.size() && mForceSampleBatchQuery[id]) 
	{
		id++;
	}
	if (id == mForceSampleBatchQuery.size())
	{
		mForceSampleBatchQuery.pushBack(0);
		NiFieldSamplerQueryData data;
		data.count = 0;
		mForceSampleBatchQueryData.pushBack(data);
	}
	mForceSampleBatchQuery[id] = static_cast<FieldSamplerQuery*>(mManager->createFieldSamplerQuery(desc));
	return id;
}


void FieldSamplerScene::releaseForceSampleBatch(PxU32 batchId)
{
	if (batchId < mForceSampleBatchQuery.size() && mForceSampleBatchQuery[batchId])
	{
		mForceSampleBatchBufferSize -= mForceSampleBatchQuery[batchId]->getQueryDesc().maxCount;
		mForceSampleBatchPosition.resize(mForceSampleBatchBufferSize);
		mForceSampleBatchVelocity.resize(mForceSampleBatchBufferSize);
		mForceSampleBatchMass.resize(mForceSampleBatchBufferSize);

		mForceSampleBatchQuery[batchId]->release();
		mForceSampleBatchQuery[batchId] = 0;
	}
}


void FieldSamplerScene::submitForceSampleBatch(	PxU32 batchId, PxVec4* forces, const PxU32 forcesStride,
								const PxVec3* positions, const PxU32 positionsStride,
								const PxVec3* velocities, const PxU32 velocitiesStride,
								const PxF32* mass, const PxU32 massStride,
								const PxU32* indices, const PxU32 numIndices)
{
	PX_UNUSED(forcesStride);
	PX_ASSERT(forcesStride == sizeof(PxVec4));
	PX_ASSERT(indices);
	if (batchId >= mForceSampleBatchQuery.size() || mForceSampleBatchQuery[batchId] == 0) return;

	PxU32 maxIndices = indices[numIndices - 1] + 1; //supposed that indices are sorted
	for (PxU32 i = 0; i < maxIndices; i++)
	{
		mForceSampleBatchPosition[mForceSampleBatchBufferPos + i] = *(PxVec4*)((PxU8*)positions + i * positionsStride);
		mForceSampleBatchVelocity[mForceSampleBatchBufferPos + i] = *(PxVec4*)((PxU8*)velocities + i * velocitiesStride);
		mForceSampleBatchMass[mForceSampleBatchBufferPos + i] = *(PxF32*)((PxU8*)mass + i * massStride);
	}

	NiFieldSamplerQueryData& data = mForceSampleBatchQueryData[batchId];
	data.count = numIndices;
	data.isDataOnDevice = false;
	data.positionStrideBytes = sizeof(PxVec4);
	data.velocityStrideBytes = sizeof(PxVec4);
	data.massStrideBytes = massStride ? sizeof(PxF32) : 0;
	data.pmaInMass = (PxF32*)&mForceSampleBatchMass[mForceSampleBatchBufferPos];
	data.pmaInPosition = (PxF32*)&mForceSampleBatchPosition[mForceSampleBatchBufferPos];
	data.pmaInVelocity = (PxF32*)&mForceSampleBatchVelocity[mForceSampleBatchBufferPos];
	data.pmaInIndices = (PxU32*)indices;
	data.pmaOutField = forces;
	data.timeStep = getApexScene().getPhysXSimulateTime();

	mForceSampleBatchBufferPos += maxIndices;
}

#endif

/******************************** CPU Version ********************************/


FieldSamplerSceneCPU::FieldSamplerSceneCPU(ModuleFieldSampler& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list) :
	FieldSamplerScene(module, scene, debugRender, list)
{
#if NX_SDK_VERSION_MAJOR == 3
	mPhysXMonitor = PX_NEW(FieldSamplerPhysXMonitor)(*this);
#endif
}

FieldSamplerSceneCPU::~FieldSamplerSceneCPU()
{
}

FieldSamplerManager* FieldSamplerSceneCPU::createManager()
{
	return PX_NEW(FieldSamplerManagerCPU)(this);
}

/******************************** GPU Version ********************************/

#if defined(APEX_CUDA_SUPPORT)

FieldSamplerSceneGPU::FieldSamplerSceneGPU(ModuleFieldSampler& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list)
	: FieldSamplerScene(module, scene, debugRender, list)
	, CudaModuleScene(scene, *mModule, APEX_CUDA_TO_STR(APEX_CUDA_MODULE_PREFIX))
{
#if NX_SDK_VERSION_MAJOR == 3
	mPhysXMonitor = PX_NEW(FieldSamplerPhysXMonitor)(*this);
#endif
	{
		physx::PxGpuDispatcher* gd = mApexScene->getTaskManager()->getGpuDispatcher();
		PX_ASSERT(gd != NULL);
		mCtxMgr = gd->getCudaContextManager();
		physx::PxScopedCudaLock _lock_(*mCtxMgr);

//CUDA module objects
#include "../cuda/include/fieldsampler.h"
	}
}

FieldSamplerSceneGPU::~FieldSamplerSceneGPU()
{
	CudaModuleScene::destroy(*mApexScene);
}

FieldSamplerManager* FieldSamplerSceneGPU::createManager()
{
	return PX_NEW(FieldSamplerManagerGPU)(this);
}

#endif

}
}
} // end namespace physx::apex

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
