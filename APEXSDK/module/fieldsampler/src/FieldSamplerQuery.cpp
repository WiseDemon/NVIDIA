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
#include "FieldSamplerQuery.h"
#include "FieldSamplerManager.h"
#include "FieldSamplerWrapper.h"
#include "FieldSamplerSceneWrapper.h"
#include "FieldBoundaryWrapper.h"

#include "NiApexScene.h"

#if defined(APEX_CUDA_SUPPORT)
#include "PxGpuTask.h"
#endif

#include "FieldSamplerCommon.h"


namespace physx
{
namespace apex
{
namespace fieldsampler
{


FieldSamplerQuery::FieldSamplerQuery(const NiFieldSamplerQueryDesc& desc, NxResourceList& list, FieldSamplerManager* manager)
	: mManager(manager)
	, mQueryDesc(desc)
	, mAccumVelocity(manager->getApexScene(), NV_ALLOC_INFO("mAccumVelocity", PARTICLES))
	, mOnStartCallback(NULL)
	, mOnFinishCallback(NULL)
{
	list.add(*this);
}

void FieldSamplerQuery::release()
{
	if (mInRelease)
	{
		return;
	}
	mInRelease = true;
	destroy();
}

void FieldSamplerQuery::destroy()
{
	delete this;
}


FieldSamplerQuery::SceneInfo* FieldSamplerQuery::findSceneInfo(FieldSamplerSceneWrapper* sceneWrapper) const
{
	for (physx::PxU32 i = 0; i < mSceneList.getSize(); ++i)
	{
		SceneInfo* sceneInfo = DYNAMIC_CAST(SceneInfo*)(mSceneList.getResource(i));
		if (sceneInfo->getSceneWrapper() == sceneWrapper)
		{
			return sceneInfo;
		}
	}
	return NULL;
}


bool FieldSamplerQuery::addFieldSampler(FieldSamplerWrapper* fieldSamplerWrapper)
{
	const NiFieldSamplerDesc& fieldSamplerDesc = fieldSamplerWrapper->getNiFieldSamplerDesc();
	float multiplier = 1.0f;
#if NX_SDK_VERSION_MAJOR == 2
	bool result = mManager->getFieldSamplerGroupsFiltering()(
	                  mQueryDesc.samplerFilterData, fieldSamplerDesc.samplerFilterData
	              );
#else
	bool result = mManager->getFieldSamplerGroupsFiltering(mQueryDesc.samplerFilterData, fieldSamplerDesc.samplerFilterData, multiplier);
#endif
	if (result)
	{
		FieldSamplerSceneWrapper* sceneWrapper = fieldSamplerWrapper->getFieldSamplerSceneWrapper();
		SceneInfo* sceneInfo = findSceneInfo(sceneWrapper);
		if (sceneInfo == NULL)
		{
			sceneInfo = createSceneInfo(sceneWrapper);
		}
		sceneInfo->addFieldSampler(fieldSamplerWrapper, multiplier);
	}
	return result;
}

bool FieldSamplerQuery::removeFieldSampler(FieldSamplerWrapper* fieldSamplerWrapper)
{
	FieldSamplerSceneWrapper* sceneWrapper = fieldSamplerWrapper->getFieldSamplerSceneWrapper();
	SceneInfo* sceneInfo = findSceneInfo(sceneWrapper);
	return (sceneInfo != NULL) ? sceneInfo->removeFieldSampler(fieldSamplerWrapper) : false;
}

void FieldSamplerQuery::clearAllFieldSamplers()
{
	for (physx::PxU32 i = 0; i < mSceneList.getSize(); ++i)
	{
		SceneInfo* sceneInfo = DYNAMIC_CAST(SceneInfo*)(mSceneList.getResource(i));
		sceneInfo->clearAllFieldSamplers();
	}
}

void FieldSamplerQuery::submitFieldSamplerQuery(const NiFieldSamplerQueryData& data, PxTask* task, PxTask* readyTask)
{
	PX_UNUSED(readyTask);
	for (physx::PxU32 i = 0; i < mSceneList.getSize(); ++i)
	{
		SceneInfo* sceneInfo = DYNAMIC_CAST(SceneInfo*)(mSceneList.getResource(i));
		NiFieldSamplerScene* niFieldSamplerScene = sceneInfo->getSceneWrapper()->getNiFieldSamplerScene();
		const physx::PxTask* fieldSamplerReadyTask = niFieldSamplerScene->onSubmitFieldSamplerQuery(data, readyTask);
		if (fieldSamplerReadyTask != 0)
		{
			task->startAfter(fieldSamplerReadyTask->getTaskID());
		}
	}
}

void FieldSamplerQuery::update()
{
	mPrimarySceneList.clear();
	mSecondarySceneList.clear();

	for (physx::PxU32 i = 0; i < mSceneList.getSize(); ++i)
	{
		SceneInfo* sceneInfo = DYNAMIC_CAST(SceneInfo*)(mSceneList.getResource(i));
		sceneInfo->update();

		if (sceneInfo->getEnabledFieldSamplerCount() > 0 && (sceneInfo->getSceneWrapper()->getNiFieldSamplerScene() != mQueryDesc.ownerFieldSamplerScene))
		{
			((sceneInfo->getSceneWrapper()->getNiFieldSamplerSceneDesc().isPrimary) ? mPrimarySceneList : mSecondarySceneList).pushBack(sceneInfo);
		}
	}
}

bool FieldSamplerQuery::SceneInfo::update()
{
	mEnabledFieldSamplerCount = 0;
	for (physx::PxU32 i = 0; i < mFieldSamplerArray.size(); ++i)
	{
		if (mFieldSamplerArray[i].mFieldSamplerWrapper->isEnabled())
		{
			++mEnabledFieldSamplerCount;
		}
		if (mFieldSamplerArray[i].mFieldSamplerWrapper->isEnabledChanged())
		{
			mFieldSamplerArrayChanged = true;
		}
	}

	if (mFieldSamplerArrayChanged)
	{
		mFieldSamplerArrayChanged = false;
		return true;
	}
	return false;
}
/******************************** CPU Version ********************************/
class TaskExecute : public physx::PxTask, public physx::UserAllocated
{
public:
	TaskExecute(FieldSamplerQueryCPU* query) : mQuery(query) {}

	const char* getName() const
	{
		return "FieldSamplerQueryCPU::TaskExecute";
	}
	void run()
	{
		mQuery->execute();
	}

protected:
	FieldSamplerQueryCPU* mQuery;
};

FieldSamplerQueryCPU::FieldSamplerQueryCPU(const NiFieldSamplerQueryDesc& desc, NxResourceList& list, FieldSamplerManager* manager)
	: FieldSamplerQuery(desc, list, manager)
{
	mTaskExecute = PX_NEW(TaskExecute)(this);

	mExecuteCount = 256;
}

FieldSamplerQueryCPU::~FieldSamplerQueryCPU()
{
	delete mTaskExecute;
}

PxTaskID FieldSamplerQueryCPU::submitFieldSamplerQuery(const NiFieldSamplerQueryData& data, PxTaskID taskID)
{
	PX_ASSERT(data.isDataOnDevice == false);
	PX_ASSERT(data.count <= mQueryDesc.maxCount);
	if (data.count == 0)
	{
		return taskID;
	}
	mQueryData = data;

	mResultField.resize(mExecuteCount);
	mWeights.resize(mExecuteCount);
	mAccumVelocity.reserve(mQueryDesc.maxCount);

	physx::PxTaskManager* tm = mManager->getApexScene().getTaskManager();
	tm->submitUnnamedTask(*mTaskExecute);

	FieldSamplerQuery::submitFieldSamplerQuery(data, mTaskExecute, NULL);

	mTaskExecute->finishBefore(taskID);
	return mTaskExecute->getTaskID();
}

void FieldSamplerQueryCPU::execute()
{
	if (mOnStartCallback)
	{
		(*mOnStartCallback)(NULL);
	}

	NiFieldSampler::ExecuteData executeData;
	
	executeData.position		= mQueryData.pmaInPosition;
	executeData.velocity		= mQueryData.pmaInVelocity;
	executeData.mass			= mQueryData.pmaInMass;//  + massOffset;
	executeData.resultField		= mResultField.begin();
	executeData.positionStride	= mQueryData.positionStrideBytes;
	executeData.velocityStride	= mQueryData.velocityStrideBytes;
	executeData.massStride		= mQueryData.massStrideBytes;
	executeData.indicesMask		= 0;

	PxU32 beginIndex;
	PxU32* indices = &beginIndex;
	if (mQueryData.pmaInIndices)
	{
		indices = mQueryData.pmaInIndices;
		executeData.indicesMask = ~executeData.indicesMask;
	}

	for (physx::PxU32 executeOffset = 0; executeOffset < mQueryData.count; executeOffset += mExecuteCount)
	{
		const PxU32 positionStride = mQueryData.positionStrideBytes / 4;
		const PxU32 velocityStride = mQueryData.velocityStrideBytes / 4;
		const PxU32 massStride = mQueryData.massStrideBytes / 4;
		//const PxU32 offset = executeOffset * stride;
		//const PxU32 massOffset = executeOffset * massStride;

		beginIndex = executeOffset;
		executeData.count        = physx::PxMin(mExecuteCount, mQueryData.count - executeOffset);
		executeData.indices		 = indices + (executeOffset & executeData.indicesMask);

		PxVec4* accumField = (PxVec4*)(mQueryData.pmaOutField);
		PxVec4* accumVelocity = mAccumVelocity.getPtr() + executeOffset;
		//clear accum
		for (physx::PxU32 i = 0; i < executeData.count; ++i)
		{
			PxU32 j = executeData.indices[i & executeData.indicesMask] + (i & ~executeData.indicesMask);
			accumField[j] = physx::PxVec4(0.0f);
			accumVelocity[i] = physx::PxVec4(0.0f);
		}
		for (physx::PxU32 sceneIdx = 0; sceneIdx < mPrimarySceneList.size(); ++sceneIdx)
		{
			executeScene(mPrimarySceneList[sceneIdx], executeData, accumField, accumVelocity, positionStride, velocityStride, massStride);
		}

		//setup weights for secondary scenes
		for (physx::PxU32 i = 0; i < executeData.count; ++i)
		{
			PxU32 j = executeData.indices[i & executeData.indicesMask] + (i & ~executeData.indicesMask);
			accumField[j].w = accumVelocity[i].w;
			accumVelocity[i].w = 0.0f;
		}
		for (physx::PxU32 sceneIdx = 0; sceneIdx < mSecondarySceneList.size(); ++sceneIdx)
		{
			executeScene(mSecondarySceneList[sceneIdx], executeData, accumField, accumVelocity, positionStride, velocityStride, massStride);
		}

		//compose accum field
		for (physx::PxU32 i = 0; i < executeData.count; ++i)
		{
			PxU32 j = executeData.indices[i & executeData.indicesMask] + (i & ~executeData.indicesMask);
			PxF32 blend = accumField[j].w;
			PxF32 velW = accumVelocity[i].w;
			PxF32 weight = blend + velW * (1 - blend);
			if (weight >= VELOCITY_WEIGHT_THRESHOLD)
			{
				PxVec3 result = accumField[j].getXYZ();
				const PxVec3& velocity = *(physx::PxVec3*)(executeData.velocity + j * velocityStride);
				result += (accumVelocity[i].getXYZ() - weight * velocity);
				accumField[j] = PxVec4(result, 0);
			}
		}
	}

	if (mOnFinishCallback)
	{
		(*mOnFinishCallback)(NULL);
	}
}

void FieldSamplerQueryCPU::executeScene(const SceneInfo* sceneInfo, 
										const NiFieldSampler::ExecuteData& executeData, 
										PxVec4* accumField, 
										PxVec4* accumVelocity, 
										PxU32 positionStride, 
										PxU32 velocityStride, 
										PxU32 massStride)
{
	FieldSamplerExecuteArgs execArgs;
	execArgs.elapsedTime = mQueryData.timeStep;
	execArgs.totalElapsedMS = mManager->getApexScene().getTotalElapsedMS();

	const physx::Array<FieldSamplerInfo>& fieldSamplerArray = sceneInfo->getFieldSamplerArray();
	for (physx::PxU32 fieldSamplerIdx = 0; fieldSamplerIdx < fieldSamplerArray.size(); ++fieldSamplerIdx)
	{
		const FieldSamplerWrapperCPU* fieldSampler = DYNAMIC_CAST(FieldSamplerWrapperCPU*)(fieldSamplerArray[fieldSamplerIdx].mFieldSamplerWrapper);
		if (fieldSampler->isEnabled())
		{
			const physx::PxF32 multiplier = fieldSamplerArray[fieldSamplerIdx].mMultiplier;
			PX_UNUSED(multiplier);

			const NiFieldSamplerDesc& desc = fieldSampler->getNiFieldSamplerDesc();
			if (desc.cpuSimulationSupport)
			{
				const NiFieldShapeDesc& shapeDesc = fieldSampler->getNiFieldSamplerShape();
				PX_ASSERT(shapeDesc.weight >= 0.0f && shapeDesc.weight <= 1.0f);

				for (physx::PxU32 i = 0; i < executeData.count; ++i)
				{
					mWeights[i] = 0;
				}

				physx::PxU32 boundaryCount = fieldSampler->getFieldBoundaryCount();
				for (physx::PxU32 boundaryIndex = 0; boundaryIndex < boundaryCount; ++boundaryIndex)
				{
					FieldBoundaryWrapper* fieldBoundaryWrapper = fieldSampler->getFieldBoundaryWrapper(boundaryIndex);

					const physx::Array<NiFieldShapeDesc>& fieldShapes = fieldBoundaryWrapper->getFieldShapes();
					for (PxU32 shapeIndex = 0; shapeIndex < fieldShapes.size(); ++shapeIndex)
					{
						const NiFieldShapeDesc& boundaryShapeDesc = fieldShapes[shapeIndex];
						PX_ASSERT(boundaryShapeDesc.weight >= 0.0f && boundaryShapeDesc.weight <= 1.0f);

						for (physx::PxU32 i = 0; i < executeData.count; ++i)
						{
							PxU32 j = executeData.indices[i & executeData.indicesMask] + (i & ~executeData.indicesMask);
							physx::PxVec3* pos = (physx::PxVec3*)(executeData.position + j * positionStride);
							const PxF32 excludeWeight = evalFade(evalDistInShape(boundaryShapeDesc, *pos), 0.0f) * boundaryShapeDesc.weight;
							mWeights[i] = physx::PxMax(mWeights[i], excludeWeight);
						}
					}
				}

				for (physx::PxU32 i = 0; i < executeData.count; ++i)
				{
					PxU32 j = executeData.indices[i & executeData.indicesMask] + (i & ~executeData.indicesMask);
					physx::PxVec3* pos = (physx::PxVec3*)(executeData.position + j * positionStride);
					const PxF32 includeWeight = evalFade(evalDistInShape(shapeDesc, *pos), desc.boundaryFadePercentage) * shapeDesc.weight;
					const PxF32 excludeWeight = mWeights[i];
					mWeights[i] = includeWeight * (1.0f - excludeWeight);
#if FIELD_SAMPLER_MULTIPLIER == FIELD_SAMPLER_MULTIPLIER_WEIGHT
					mWeights[i] *= multiplier;
#endif
				}

				//execute field
				fieldSampler->getNiFieldSampler()->executeFieldSampler(executeData);

#if FIELD_SAMPLER_MULTIPLIER == FIELD_SAMPLER_MULTIPLIER_VALUE
				const physx::PxF32 multiplier = fieldSamplerArray[fieldSamplerIdx].mMultiplier;
				for (physx::PxU32 i = 0; i < executeData.count; ++i)
				{
					executeData.resultField[i] *= multiplier;
				}
#endif

				//accum field
				switch (desc.type)
				{
				case NiFieldSamplerType::FORCE:
					for (physx::PxU32 i = 0; i < executeData.count; ++i)
					{
						PxU32 j = executeData.indices[i & executeData.indicesMask] + (i & ~executeData.indicesMask);
						execArgs.position = *(physx::PxVec3*)(executeData.position + j * positionStride);
						execArgs.velocity = *(physx::PxVec3*)(executeData.velocity + j * velocityStride);
						execArgs.mass     = *(executeData.mass + massStride * j);

						accumFORCE(execArgs, executeData.resultField[i], mWeights[i], accumField[j], accumVelocity[i]);
					}
					break;
				case NiFieldSamplerType::ACCELERATION:
					for (physx::PxU32 i = 0; i < executeData.count; ++i)
					{
						PxU32 j = executeData.indices[i & executeData.indicesMask] + (i & ~executeData.indicesMask);
						execArgs.position = *(physx::PxVec3*)(executeData.position + j * positionStride);
						execArgs.velocity = *(physx::PxVec3*)(executeData.velocity + j * velocityStride);
						execArgs.mass     = *(executeData.mass + massStride * j);

						accumACCELERATION(execArgs, executeData.resultField[i], mWeights[i], accumField[j], accumVelocity[i]);
					}
					break;
				case NiFieldSamplerType::VELOCITY_DRAG:
					for (physx::PxU32 i = 0; i < executeData.count; ++i)
					{
						PxU32 j = executeData.indices[i & executeData.indicesMask] + (i & ~executeData.indicesMask);
						execArgs.position = *(physx::PxVec3*)(executeData.position + j * positionStride);
						execArgs.velocity = *(physx::PxVec3*)(executeData.velocity + j * velocityStride);
						execArgs.mass     = *(executeData.mass + massStride * j);

						accumVELOCITY_DRAG(execArgs, desc.dragCoeff, executeData.resultField[i], mWeights[i], accumField[j], accumVelocity[i]);
					}
					break;
				case NiFieldSamplerType::VELOCITY_DIRECT:
					for (physx::PxU32 i = 0; i < executeData.count; ++i)
					{
						PxU32 j = executeData.indices[i & executeData.indicesMask] + (i & ~executeData.indicesMask);
						execArgs.position = *(physx::PxVec3*)(executeData.position + j * positionStride);
						execArgs.velocity = *(physx::PxVec3*)(executeData.velocity + j * velocityStride);
						execArgs.mass     = *(executeData.mass + massStride * j);

						accumVELOCITY_DIRECT(execArgs, executeData.resultField[i], mWeights[i], accumField[j], accumVelocity[i]);
					}
					break;
				};
			}
		}
	}

}


/******************************** GPU Version ********************************/
#if defined(APEX_CUDA_SUPPORT)

class FieldSamplerQueryLaunchTask : public PxGpuTask, public physx::UserAllocated
{
public:
	FieldSamplerQueryLaunchTask(FieldSamplerQueryGPU* query) : mQuery(query) {}
	const char* getName() const
	{
		return "FieldSamplerQueryLaunchTask";
	}
	void         run()
	{
		PX_ALWAYS_ASSERT();
	}
	bool         launchInstance(CUstream stream, int kernelIndex)
	{
		return mQuery->launch(stream, kernelIndex);
	}
	physx::PxGpuTaskHint::Enum getTaskHint() const
	{
		return physx::PxGpuTaskHint::Kernel;
	}

protected:
	FieldSamplerQueryGPU* mQuery;
};

class FieldSamplerQueryPrepareTask : public physx::PxTask, public physx::UserAllocated
{
public:
	FieldSamplerQueryPrepareTask(FieldSamplerQueryGPU* query) : mQuery(query) {}

	const char* getName() const
	{
		return "FieldSamplerQueryPrepareTask";
	}
	void run()
	{
		mQuery->prepare();
	}

protected:
	FieldSamplerQueryGPU* mQuery;
};

class FieldSamplerQueryCopyTask : public PxGpuTask, public physx::UserAllocated
{
public:
	FieldSamplerQueryCopyTask(FieldSamplerQueryGPU* query) : mQuery(query) {}
	const char* getName() const
	{
		return "FieldSamplerQueryCopyTask";
	}
	void         run()
	{
		PX_ALWAYS_ASSERT();
	}
	bool         launchInstance(CUstream stream, int kernelIndex)
	{
		return mQuery->copy(stream, kernelIndex);
	}
	physx::PxGpuTaskHint::Enum getTaskHint() const
	{
		return physx::PxGpuTaskHint::Kernel;
	}

protected:
	FieldSamplerQueryGPU* mQuery;
};

class FieldSamplerQueryFetchTask : public physx::PxTask, public physx::UserAllocated
{
public:
	FieldSamplerQueryFetchTask(FieldSamplerQueryGPU* query) : mQuery(query) {}

	const char* getName() const
	{
		return "FieldSamplerQueryFetchTask";
	}
	void run()
	{
		mQuery->fetch();
	}

protected:
	FieldSamplerQueryGPU* mQuery;
};


FieldSamplerQueryGPU::FieldSamplerQueryGPU(const NiFieldSamplerQueryDesc& desc, NxResourceList& list, FieldSamplerManager* manager)
	: FieldSamplerQueryCPU(desc, list, manager)
	, mPositionMass(manager->getApexScene(), NV_ALLOC_INFO("mPositionMass", PARTICLES))
	, mVelocity(manager->getApexScene(), NV_ALLOC_INFO("mVelocity", PARTICLES))
	, mAccumField(manager->getApexScene(), NV_ALLOC_INFO("mAccumField", PARTICLES))
	, mCopyQueue(*manager->getApexScene().getTaskManager()->getGpuDispatcher())
{
	mTaskLaunch   = PX_NEW(FieldSamplerQueryLaunchTask)(this);
	mTaskPrepare  = PX_NEW(FieldSamplerQueryPrepareTask)(this);
	mTaskCopy     = PX_NEW(FieldSamplerQueryCopyTask)(this);
	mTaskFetch    = PX_NEW(FieldSamplerQueryFetchTask)(this);
}

FieldSamplerQueryGPU::~FieldSamplerQueryGPU()
{
	PX_DELETE(mTaskFetch);
	PX_DELETE(mTaskCopy);
	PX_DELETE(mTaskPrepare);
	PX_DELETE(mTaskLaunch);
}

PxTaskID FieldSamplerQueryGPU::submitFieldSamplerQuery(const NiFieldSamplerQueryData& data, PxTaskID taskID)
{
	PX_ASSERT(data.count <= mQueryDesc.maxCount);
	if (data.count == 0)
	{
		return taskID;
	}
	mQueryData = data;

	if (!data.isDataOnDevice)
	{
		bool isWorkOnCPU = true;
		// try to find FieldSampler which has no CPU implemntation (Turbulence for example)
		for (physx::PxU32 sceneIdx = 0; (sceneIdx < mPrimarySceneList.size() + mSecondarySceneList.size()) && isWorkOnCPU; ++sceneIdx)
		{
			const physx::Array<FieldSamplerInfo>& fsArray = sceneIdx < mPrimarySceneList.size() 
				? mPrimarySceneList[sceneIdx]->getFieldSamplerArray() 
				: mSecondarySceneList[sceneIdx-mPrimarySceneList.size()]->getFieldSamplerArray();
			for (physx::PxU32 fsIdx = 0; fsIdx < fsArray.size() && isWorkOnCPU; fsIdx++)
			{
				if (fsArray[fsIdx].mFieldSamplerWrapper->isEnabled())
				{
					isWorkOnCPU = fsArray[fsIdx].mFieldSamplerWrapper->getNiFieldSamplerDesc().cpuSimulationSupport;
				}
			}
		}

		// if all FSs can work on CPU we will execute FieldSamplerQuery on CPU
		if (isWorkOnCPU)
		{
			return FieldSamplerQueryCPU::submitFieldSamplerQuery(data, taskID);
		}

		mPositionMass.reserve(mQueryDesc.maxCount, ApexMirroredPlace::CPU_GPU);
		mVelocity.reserve(mQueryDesc.maxCount, ApexMirroredPlace::CPU_GPU);
		mAccumField.reserve(mQueryDesc.maxCount, ApexMirroredPlace::CPU_GPU);
	}
	mAccumVelocity.reserve(mQueryDesc.maxCount, ApexMirroredPlace::CPU_GPU);
	
	// if data on device or some FS can't work on CPU we will launch FieldSamplerQuery on GPU
	physx::PxTaskManager* tm = mManager->getApexScene().getTaskManager();
	tm->submitUnnamedTask(*mTaskLaunch, physx::PxTaskType::TT_GPU);

	if (data.isDataOnDevice)
	{
		FieldSamplerQuery::submitFieldSamplerQuery(data, mTaskLaunch, NULL);

		mTaskLaunch->finishBefore(taskID);
		return mTaskLaunch->getTaskID();
	}
	else
	{
		NiFieldSamplerQueryData data4Device;
		data4Device.timeStep = data.timeStep;
		data4Device.count = data.count;
		data4Device.isDataOnDevice = true;
		data4Device.positionStrideBytes = sizeof(physx::PxVec4);
		data4Device.velocityStrideBytes = sizeof(physx::PxVec4);
		data4Device.massStrideBytes = sizeof(physx::PxVec4);
		data4Device.pmaInPosition = (PxF32*)mPositionMass.getGpuPtr();
		data4Device.pmaInVelocity = (PxF32*)mVelocity.getGpuPtr();
		data4Device.pmaInMass = &mPositionMass.getGpuPtr()->w;
		data4Device.pmaOutField = mAccumField.getGpuPtr();
		data4Device.pmaInIndices = 0;

		FieldSamplerQuery::submitFieldSamplerQuery(data4Device, mTaskLaunch, mTaskCopy);

		tm->submitUnnamedTask(*mTaskPrepare);
		tm->submitUnnamedTask(*mTaskCopy, physx::PxTaskType::TT_GPU);
		tm->submitUnnamedTask(*mTaskFetch);

		mTaskPrepare->finishBefore(mTaskCopy->getTaskID());
		mTaskCopy->finishBefore(mTaskLaunch->getTaskID());
		mTaskLaunch->finishBefore(mTaskFetch->getTaskID());
		mTaskFetch->finishBefore(taskID);
		return mTaskPrepare->getTaskID();
	}
}

void FieldSamplerQueryGPU::prepare()
{
	const PxU32 positionStride = mQueryData.positionStrideBytes / sizeof(PxF32);
	const PxU32 velocityStride = mQueryData.velocityStrideBytes / sizeof(PxF32);
	const PxU32 massStride = mQueryData.massStrideBytes / sizeof(PxF32);
	for (PxU32 idx = 0; idx < mQueryData.count; idx++)
	{
		mPositionMass[idx] = physx::PxVec4(*(PxVec3*)(mQueryData.pmaInPosition + idx * positionStride), *(mQueryData.pmaInMass + idx * massStride));
		mVelocity[idx] = physx::PxVec4(*(PxVec3*)(mQueryData.pmaInVelocity + idx * velocityStride), 0.f);
	}
}

void FieldSamplerQueryGPU::fetch()
{
	for (PxU32 idx = 0; idx < mQueryData.count; idx++)
	{
		mQueryData.pmaOutField[idx] = mAccumField[idx];
	}
}

bool FieldSamplerQueryGPU::copy(CUstream stream, int kernelIndex)
{
	if (kernelIndex == 0)
	{
		mCopyQueue.reset(stream, 4);
		mPositionMass.copyHostToDeviceQ(mCopyQueue, mQueryData.count);
		mVelocity.copyHostToDeviceQ(mCopyQueue, mQueryData.count);
		mCopyQueue.flushEnqueued();
	}
	return false;
}

bool FieldSamplerQueryGPU::launch(CUstream stream, int kernelIndex)
{
	FieldSamplerPointsKernelArgs args;
	args.elapsedTime        = mQueryData.timeStep;
	args.totalElapsedMS     = mManager->getApexScene().getTotalElapsedMS();
	if (mQueryData.isDataOnDevice)
	{
		args.positionMass   = (float4*)mQueryData.pmaInPosition;
		args.velocity       = (float4*)mQueryData.pmaInVelocity;
		args.accumField     = (float4*)mQueryData.pmaOutField;
	}
	else
	{
		args.positionMass   = (float4*)mPositionMass.getGpuPtr();
		args.velocity       = (float4*)mVelocity.getGpuPtr();
		args.accumField     = (float4*)mAccumField.getGpuPtr();
	}
	args.accumVelocity      = (float4*)mAccumVelocity.getGpuPtr();

	NiFieldSamplerPointsKernelLaunchData launchData;
	launchData.stream       = stream;
	launchData.kernelType   = FieldSamplerKernelType::POINTS;
	launchData.kernelArgs   = &args;
	launchData.threadCount  = mQueryData.count;
	launchData.memRefSize   = mQueryData.count;

	if (kernelIndex == 0 && mOnStartCallback)
	{
		(*mOnStartCallback)(stream);
	}

	if (kernelIndex == 0)
	{
		CUDA_OBJ(clearKernel)(stream, mQueryData.count,
		                      createApexCudaMemRef(args.accumField, launchData.memRefSize, ApexCudaMemFlags::OUT),
		                      createApexCudaMemRef(args.accumVelocity, launchData.memRefSize, ApexCudaMemFlags::OUT));
		return true;
	}
	--kernelIndex;

	const PxU32 bothSceneCount = mPrimarySceneList.size() + mSecondarySceneList.size();
	if (kernelIndex < (int) bothSceneCount)
	{
		SceneInfo* sceneInfo = (kernelIndex < (int) mPrimarySceneList.size()) 
			? mPrimarySceneList[(physx::PxU32)kernelIndex] 
			: mSecondarySceneList[(physx::PxU32)kernelIndex - mPrimarySceneList.size()];
		SceneInfoGPU* sceneInfoGPU = DYNAMIC_CAST(SceneInfoGPU*)(sceneInfo);

		launchData.kernelMode = FieldSamplerKernelMode::DEFAULT;
		if (kernelIndex == (int) mPrimarySceneList.size() - 1)
		{
			launchData.kernelMode = FieldSamplerKernelMode::FINISH_PRIMARY;
		}
		if ((kernelIndex == (int) bothSceneCount - 1))
		{
			launchData.kernelMode = FieldSamplerKernelMode::FINISH_SECONDARY;
		}

		FieldSamplerSceneWrapperGPU* sceneWrapper = DYNAMIC_CAST(FieldSamplerSceneWrapperGPU*)(sceneInfo->getSceneWrapper());

		launchData.queryParamsHandle = sceneInfoGPU->getQueryParamsHandle();
		launchData.paramsExArrayHandle = sceneInfoGPU->getParamsHandle();
		launchData.fieldSamplerArray = &sceneInfo->getFieldSamplerArray();
		launchData.activeFieldSamplerCount = sceneInfo->getEnabledFieldSamplerCount();

		sceneWrapper->getNiFieldSamplerScene()->launchFieldSamplerCudaKernel(launchData);
		return true;
	}
	kernelIndex -= bothSceneCount;

	if (kernelIndex == 0)
	{
		CUDA_OBJ(composeKernel)(stream, mQueryData.count,
								createApexCudaMemRef(args.accumField, launchData.memRefSize, ApexCudaMemFlags::IN_OUT),
								createApexCudaMemRef((const float4*)args.accumVelocity, launchData.memRefSize, ApexCudaMemFlags::IN),
								createApexCudaMemRef(args.velocity, launchData.memRefSize, ApexCudaMemFlags::IN),
								args.elapsedTime);
		return true;
	}
	--kernelIndex;

	if (!mQueryData.isDataOnDevice)
	{
		mAccumField.copyDeviceToHostQ(mCopyQueue, mQueryData.count);
		mCopyQueue.flushEnqueued();

		physx::PxTaskManager* tm = mManager->getApexScene().getTaskManager();
		tm->getGpuDispatcher()->addCompletionPrereq(*mTaskFetch);
	}

	if (mOnFinishCallback)
	{
		(*mOnFinishCallback)(stream);
	}
	return false;
}

FieldSamplerQueryGPU::SceneInfoGPU::SceneInfoGPU(NxResourceList& list, FieldSamplerQuery* query, FieldSamplerSceneWrapper* sceneWrapper)
	: SceneInfo(list, query, sceneWrapper)
	, mConstMemGroup(DYNAMIC_CAST(FieldSamplerSceneWrapperGPU*)(sceneWrapper)->getConstStorage())
{
	APEX_CUDA_CONST_MEM_GROUP_SCOPE(mConstMemGroup);

	mQueryParamsHandle.alloc(_storage_);
}

bool FieldSamplerQueryGPU::SceneInfoGPU::update()
{
	if (FieldSamplerQuery::SceneInfo::update())
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(mConstMemGroup);

		FieldSamplerParamsExArray paramsExArray;
		mParamsExArrayHandle.allocOrFetch(_storage_, paramsExArray);
		if (paramsExArray.resize(_storage_, mEnabledFieldSamplerCount))
		{
			for (physx::PxU32 i = 0, enabledIdx = 0; i < mFieldSamplerArray.size(); ++i)
			{
				FieldSamplerWrapperGPU* fieldSamplerWrapper = DYNAMIC_CAST(FieldSamplerWrapperGPU*)(mFieldSamplerArray[i].mFieldSamplerWrapper);
				if (fieldSamplerWrapper->isEnabled())
				{
					FieldSamplerParamsEx fsParamsEx;
					fsParamsEx.paramsHandle = fieldSamplerWrapper->getParamsHandle();
					fsParamsEx.multiplier = mFieldSamplerArray[i].mMultiplier;
					PX_ASSERT(enabledIdx < mEnabledFieldSamplerCount);
					paramsExArray.updateElem(_storage_, fsParamsEx, enabledIdx++);
				}
			}
			mParamsExArrayHandle.update(_storage_, paramsExArray);
		}
		return true;
	}
	return false;
}

physx::PxVec3 FieldSamplerQueryGPU::executeFieldSamplerQueryOnGrid(const NiFieldSamplerQueryGridData& data)
{
	FieldSamplerGridKernelArgs args;

	args.numX           = data.numX;
	args.numY           = data.numY;
	args.numZ           = data.numZ;

	args.gridToWorld    = data.gridToWorld;

	args.mass           = data.mass;
	args.elapsedTime    = data.timeStep;
	args.cellSize		= data.cellSize;
	args.totalElapsedMS = mManager->getApexScene().getTotalElapsedMS();

	NiFieldSamplerGridKernelLaunchData launchData;
	launchData.stream       = data.stream;
	launchData.kernelType   = FieldSamplerKernelType::GRID;
	launchData.kernelArgs   = &args;
	launchData.threadCountX	= data.numX;
	launchData.threadCountY	= data.numY;
	launchData.threadCountZ	= data.numZ;
	launchData.accumArray	= data.resultVelocity;


	{
		APEX_CUDA_SURFACE_SCOPE_BIND(surfRefGridAccum, *launchData.accumArray, ApexCudaMemFlags::OUT);

		CUDA_OBJ(clearGridKernel)(data.stream, launchData.threadCountX, launchData.threadCountY, launchData.threadCountZ,
								  args.numX, args.numY, args.numZ);
	}

	physx::PxVec3 velocity(0.0f);
	for (physx::PxU32 i = 0; i < mSecondarySceneList.size(); ++i)
	{
		SceneInfoGPU* sceneInfo = DYNAMIC_CAST(SceneInfoGPU*)(mSecondarySceneList[i]);
		FieldSamplerSceneWrapperGPU* sceneWrapper = DYNAMIC_CAST(FieldSamplerSceneWrapperGPU*)(sceneInfo->getSceneWrapper());

		launchData.activeFieldSamplerCount = 0;

		const physx::Array<FieldSamplerInfo>& fieldSamplerArray = sceneInfo->getFieldSamplerArray();
		for (physx::PxU32 fieldSamplerIdx = 0; fieldSamplerIdx < fieldSamplerArray.size(); ++fieldSamplerIdx)
		{
			const FieldSamplerWrapperGPU* wrapper = static_cast<const FieldSamplerWrapperGPU* >( fieldSamplerArray[fieldSamplerIdx].mFieldSamplerWrapper );
			if (wrapper->isEnabled())
			{
				switch (wrapper->getNiFieldSamplerDesc().gridSupportType)
				{
					case NiFieldSamplerGridSupportType::SINGLE_VELOCITY:
					{
						const NiFieldSampler* fieldSampler = wrapper->getNiFieldSampler();
						velocity += fieldSampler->queryFieldSamplerVelocity();
					}
					break;
					case NiFieldSamplerGridSupportType::VELOCITY_PER_CELL:
					{
						launchData.activeFieldSamplerCount += 1;
					}
					break;
					default:
						break;
				}
			}
		}

		if (launchData.activeFieldSamplerCount > 0)
		{
			launchData.queryParamsHandle = sceneInfo->getQueryParamsHandle();
			launchData.paramsExArrayHandle = sceneInfo->getParamsHandle();
			launchData.fieldSamplerArray = &sceneInfo->getFieldSamplerArray();
			launchData.kernelMode = FieldSamplerKernelMode::DEFAULT;

			sceneWrapper->getNiFieldSamplerScene()->launchFieldSamplerCudaKernel(launchData);
		}
	}
	return velocity;
}


#endif

}
}
} // end namespace physx::apex

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
