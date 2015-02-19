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
#include "PsArray.h"
#include "ApexInterface.h"
#include "NiApexScene.h"
#include "ModuleIofx.h"
#include "IofxManager.h"
#include "IofxSceneCPU.h"
#include "IofxAsset.h"
#include "IosObjectData.h"
#include "IofxRenderData.h"

#include "IofxActorCPU.h"

#ifdef APEX_TEST
#include "IofxManagerTestData.h"
#endif

#ifdef APEX_CUDA_SUPPORT
#include "ApexCuda.h" // APEX_CUDA_MEM_ALIGN_UP_32BIT
#include "ApexMirroredArray.h"
#if ENABLE_TEST
#include "IofxTestManagerGPU.h"
#define IOFX_MANAGER_GPU IofxTestManagerGPU
#else
#include "IofxManagerGPU.h"
#define IOFX_MANAGER_GPU IofxManagerGPU
#endif
#endif

#define BASE_SPRITE_SEMANTICS (1<<NxRenderSpriteSemantic::POSITION) | \
	(1<<NxRenderSpriteSemantic::VELOCITY) | \
	(1<<NxRenderSpriteSemantic::LIFE_REMAIN)

#define BASE_MESH_SEMANTICS   (1<<NxRenderInstanceSemantic::POSITION) | \
	(1<<NxRenderInstanceSemantic::ROTATION_SCALE) | \
	(1<<NxRenderInstanceSemantic::VELOCITY_LIFE)

namespace physx
{
namespace apex
{
namespace iofx
{

#pragma warning(disable: 4355) // 'this' : used in base member initializer list


IofxManager::IofxManager(IofxScene& scene, const NiIofxManagerDesc& desc, bool isMesh)
	: mPostUpdateTaskID(0)
	, mIofxScene(&scene)
	, mIosAssetName(desc.iosAssetName)
	, mWorkingIosData(NULL)
	, mResultIosData(NULL)
	, mStagingIosData(NULL)
	, mRenderIosData(NULL)
	, mTargetSemantics(0)
	, mInteropState(INTEROP_OFF)
	, mResultReadyState(RESULT_WAIT_FOR_NEW)
	, positionMass(*scene.mApexScene, NV_ALLOC_INFO("positionMass", PARTICLES))
	, velocityLife(*scene.mApexScene, NV_ALLOC_INFO("velocityLife", PARTICLES))
	, collisionNormalFlags(*scene.mApexScene, NV_ALLOC_INFO("collisionNormalFlags", PARTICLES))
	, density(*scene.mApexScene, NV_ALLOC_INFO("density", PARTICLES))
	, actorIdentifiers(*scene.mApexScene, NV_ALLOC_INFO("actorIdentifiers", PARTICLES))
	, inStateToInput(*scene.mApexScene, NV_ALLOC_INFO("inStateToInput", PARTICLES))
	, outStateToInput(*scene.mApexScene, NV_ALLOC_INFO("outStateToInput", PARTICLES))
	, userData(*scene.mApexScene, NV_ALLOC_INFO("userData", PARTICLES))
	, pubStateSize(0)
	, privStateSize(0)
	, mStateSwap(false)
	, mTotalElapsedTime(0)
	, mIsMesh(isMesh)
	, mDistanceSortingEnabled(false)
	, mCudaIos(desc.iosOutputsOnDevice)
	, mCudaModifiers(false)
	, mCudaPipeline(NULL)
	, mSimulateTask(*this)
	, mCopyQueue(*scene.mApexScene->getTaskManager()->getGpuDispatcher())
	, mLastNumObjects(0)
	, mLastMaxInputID(0)
	, mOnStartCallback(NULL)
	, mOnFinishCallback(NULL)
#ifdef APEX_TEST
	, mTestData(NULL)
#endif
{
	scene.mActorManagers.add(*this);

	mBounds.setEmpty();

	mInStateOffset = 0;
	mOutStateOffset = desc.maxObjectCount;

	// The decision whether to use GPU IOFX Modifiers is separate from whether the IOS
	// outputs will come from the GPU or not
#if defined(APEX_CUDA_SUPPORT)
	physx::PxGpuDispatcher* gd = scene.mApexScene->getTaskManager()->getGpuDispatcher();
	if (gd && gd->getCudaContextManager()->contextIsValid() && !scene.mModule->mCudaDisabled)
	{
		mCudaModifiers = true;
		// detect interop
		if (gd->getCudaContextManager()->getInteropMode() != PxCudaInteropMode::NO_INTEROP && !scene.mModule->mInteropDisabled)
		{
			mInteropState = INTEROP_WAIT_FOR_RENDER_ALLOC;
		}
		const PxU32 dataCount = (mInteropState != INTEROP_OFF) ? 3u : 2u;
		for (PxU32 i = 0 ; i < dataCount ; i++)
		{
			IofxOutputData* outputData = mIsMesh ? static_cast<IofxOutputData*>(PX_NEW(IofxOutputDataMesh)()) : PX_NEW(IofxOutputDataSprite)();
			IosObjectGpuData* gpuIosData = PX_NEW(IosObjectGpuData)(i, outputData);
			mObjData.pushBack(gpuIosData);
		}

		mOutStateOffset = APEX_CUDA_MEM_ALIGN_UP_32BIT(desc.maxObjectCount);
		mCudaPipeline = PX_NEW(IOFX_MANAGER_GPU)(*mIofxScene->mApexScene, desc, *this);
	}
	else
#endif
	{
		for (PxU32 i = 0 ; i < 2 ; i++)
		{
			IofxOutputData* outputData = mIsMesh ? static_cast<IofxOutputData*>(PX_NEW(IofxOutputDataMesh)()) : PX_NEW(IofxOutputDataSprite)();
			IosObjectCpuData* cpuIosData = PX_NEW(IosObjectCpuData)(i, outputData);
			mObjData.pushBack(cpuIosData);
		}
	}

	mWorkingIosData = mObjData[0];
	mResultIosData = mObjData[1];

	// Create & Assign Shared Render Data
	if (mIsMesh)
	{
		mSharedRenderData = PX_NEW(IofxSharedRenderDataMesh)(0);
	}
	else
	{
		mSharedRenderData = PX_NEW(IofxSharedRenderDataSprite)(0);
	}

#if defined(APEX_CUDA_SUPPORT)
	if (mInteropState != INTEROP_OFF)
	{
		mStagingIosData = mObjData[2];

		mInteropRenderData.resize( mObjData.size() );
		for (PxU32 i = 0 ; i < mObjData.size() ; i++)
		{
			IofxSharedRenderData* renderData;
			if (mIsMesh)
			{
				renderData = PX_NEW(IofxSharedRenderDataMesh)(i + 1);
			}
			else
			{
				renderData = PX_NEW(IofxSharedRenderDataSprite)(i + 1);
			}
			renderData->setUseInterop(true);

			mInteropRenderData[i] = renderData;
			mObjData[i]->renderData = renderData;
		}
	}
	else
#endif
	{
		for (PxU32 i = 0 ; i < mObjData.size() ; i++)
		{
			mObjData[i]->renderData = mSharedRenderData;
		}
	}

	ApexMirroredPlace::Enum place = ApexMirroredPlace::CPU;
#if defined(APEX_CUDA_SUPPORT)
	if (mCudaIos || mCudaModifiers)
	{
		place =  ApexMirroredPlace::CPU_GPU;
	}
#endif
	{
		positionMass.setSize(desc.maxInputCount, place);
		velocityLife.setSize(desc.maxInputCount, place);
		if (desc.iosSupportsCollision)
		{
			collisionNormalFlags.setSize(desc.maxInputCount, place);
		}
		if (desc.iosSupportsDensity)
		{
			density.setSize(desc.maxInputCount, place);
		}
		actorIdentifiers.setSize(desc.maxInputCount, place);
		inStateToInput.setSize(desc.maxInStateCount, place);
		outStateToInput.setSize(desc.maxObjectCount, place);

		if (desc.iosSupportsUserData)
		{
			userData.setSize(desc.maxInputCount, place);
		}

		mSimBuffers.pmaPositionMass = &positionMass;
		mSimBuffers.pmaVelocityLife = &velocityLife;
		mSimBuffers.pmaCollisionNormalFlags = desc.iosSupportsCollision ? &collisionNormalFlags : NULL;
		mSimBuffers.pmaDensity = desc.iosSupportsDensity ? &density : NULL;
		mSimBuffers.pmaActorIdentifiers = &actorIdentifiers;
		mSimBuffers.pmaInStateToInput = &inStateToInput;
		mSimBuffers.pmaOutStateToInput = &outStateToInput;
		mSimBuffers.pmaUserData = desc.iosSupportsUserData ? &userData : NULL;
	}

	if (!mCudaModifiers)
	{
		mOutputToState.resize(desc.maxObjectCount);
	}

	/* Initialize IOS object data structures */
	for (PxU32 i = 0 ; i < mObjData.size() ; i++)
	{
		mObjData[i]->pmaPositionMass = mSimBuffers.pmaPositionMass;
		mObjData[i]->pmaVelocityLife = mSimBuffers.pmaVelocityLife;
		mObjData[i]->pmaCollisionNormalFlags = mSimBuffers.pmaCollisionNormalFlags;
		mObjData[i]->pmaDensity = mSimBuffers.pmaDensity;
		mObjData[i]->pmaActorIdentifiers = mSimBuffers.pmaActorIdentifiers;
		mObjData[i]->pmaInStateToInput = mSimBuffers.pmaInStateToInput;
		mObjData[i]->pmaOutStateToInput = mSimBuffers.pmaOutStateToInput;
		mObjData[i]->pmaUserData = mSimBuffers.pmaUserData;

		mObjData[i]->iosAssetName = desc.iosAssetName;
		mObjData[i]->iosOutputsOnDevice = desc.iosOutputsOnDevice;
		mObjData[i]->iosSupportsDensity = desc.iosSupportsDensity;
		mObjData[i]->iosSupportsCollision = desc.iosSupportsCollision;
		mObjData[i]->iosSupportsUserData = desc.iosSupportsUserData;
		mObjData[i]->maxObjectCount = desc.maxObjectCount;
		mObjData[i]->maxInputCount = desc.maxInputCount;
		mObjData[i]->maxInStateCount = desc.maxInStateCount;
	}
}

IofxManager::~IofxManager()
{
	for (PxU32 i = 0; i < pubState.slices.size(); ++i)
	{
		delete pubState.slices[i];
	}

	for (PxU32 i = 0; i < privState.slices.size(); ++i)
	{
		delete privState.slices[i];
	}
}

void IofxManager::destroy()
{
#if defined(APEX_CUDA_SUPPORT)
	if (mCudaPipeline)
	{
		mCudaPipeline->release();
	}
	for (PxU32 i = 0 ; i < mInteropRenderData.size() ; i++)
	{
		PX_DELETE(mInteropRenderData[i]);
	}
#endif
	if (mSharedRenderData != NULL)
	{
		PX_DELETE(mSharedRenderData);
	}
	for (PxU32 i = 0 ; i < mObjData.size() ; i++)
	{
		PX_DELETE(mObjData[i]);
	}

	delete this;
}


void IofxManager::release()
{
	mIofxScene->releaseIofxManager(this);
}

#if !defined(APEX_CUDA_SUPPORT)
/* Stubs for console builds */
void IofxManager::fillMapUnmapArraysForInterop(physx::Array<CUgraphicsResource> &, physx::Array<CUgraphicsResource> &) {}
void IofxManager::mapBufferResults(bool, bool) {}
#endif

void IofxManager::prepareRenderResources()
{
	if (mResultReadyState != RESULT_READY)
	{
		mRenderIosData = NULL;
		return;
	}
	mResultReadyState = RESULT_WAIT_FOR_NEW;

	mRenderIosData = mResultIosData;
#if defined(APEX_CUDA_SUPPORT)
	if (mInteropState == INTEROP_FAILED)
	{
		//interop has failed, so it is unsafe to use new render data
		//just return and hope that old render data is still ok
		mRenderIosData = NULL;
		return;
	}
	if (mInteropState == INTEROP_READY)
	{
		physx::swap(mStagingIosData, mResultIosData);

		if (mRenderIosData->renderData->getBufferIsMapped())
		{
			APEX_INTERNAL_ERROR("IofxManager: CUDA Interop Error - render data is still mapped to CUDA memory!");
			PX_ASSERT(0);
		}
	}
	else
#endif
	{
		mRenderIosData->renderData->alloc(mRenderIosData, NULL);
	}

	bool bNeedUpdate = false;

	// mLiveRenderVolumesLock is allready locked in IofxScene::prepareRenderResources
	for (PxU32 i = 0 ; i < mIofxScene->mLiveRenderVolumes.size() ; i++)
	{
		ApexRenderVolume* vol = mIofxScene->mLiveRenderVolumes[i];
		// all render volumes are allready locked in IofxScene::prepareRenderResources

		PxU32 iofxActorCount;
		NxIofxActor* const* iofxActorList = vol->getIofxActorList(iofxActorCount);
		for (PxU32 iofxActorIndex = 0; iofxActorIndex < iofxActorCount; ++iofxActorIndex)
		{
			IofxActor* iofxActor = DYNAMIC_CAST(IofxActor*)( iofxActorList[iofxActorIndex] );
			if (&iofxActor->mMgr == this)
			{
				bNeedUpdate |= iofxActor->prepareRenderResources(mRenderIosData);
			}
		}
	}

	if (bNeedUpdate)
	{
		mRenderIosData->prepareRenderDataUpdate();
	}
}

void IofxManager::postPrepareRenderResources()
{
	if (mRenderIosData != NULL)
	{
		mRenderIosData->executeRenderDataUpdate();
		mRenderIosData = NULL;
	}
}


PxF32 IofxManager::getObjectRadius() const
{
	return mObjData[0] ? mObjData[0]->radius : 0.0f;
}

void IofxManager::setSimulationParameters(PxF32 radius, const PxVec3& up, PxF32 gravity, PxF32 restDensity)
{
	/* Initialize IOS object data structures */
	for (PxU32 i = 0 ; i < mObjData.size() ; i++)
	{
		mObjData[i]->radius = radius;
		mObjData[i]->upVector = up;
		mObjData[i]->gravity = gravity;
		mObjData[i]->restDensity = restDensity;
	}
}

void IofxManager::createSimulationBuffers(NiIosBufferDesc& outDesc)
{
	outDesc = mSimBuffers;
}

/* Called by owning IOS actor during simulation startup, only if
 * the IOS is going to simulate this frame, so it is safe to submit
 * tasks from here. postUpdateTaskID is the ID for an IOS task
 * that should run after IOFX modifiers.  If the IOFX Manager adds
 * no dependencies, postUpdateTaskID task will run right after
 * updateEffectsData() returns.  If updateEffectsData() will be completely
 * synchronous, it is safe to return 0 here.
 */
PxTaskID IofxManager::getUpdateEffectsTaskID(PxTaskID postUpdateTaskID)
{
	physx::PxTaskManager* tm = mIofxScene->mApexScene->getTaskManager();
	mPostUpdateTaskID = postUpdateTaskID;
	if (mCudaModifiers)
	{
		return mCudaPipeline->launchGpuTasks();
	}
	else
	{
		tm->submitUnnamedTask(mSimulateTask);
		mSimulateTask.finishBefore(tm->getNamedTask(AST_PHYSX_FETCH_RESULTS));
		return mSimulateTask.getTaskID();
	}
}


void TaskUpdateEffects::run()
{
	setProfileStat((PxU16) mOwner.mWorkingIosData->numParticles);
	mOwner.cpuModifiers();
}

/// \brief Called by IOS actor before TaskUpdateEffects is scheduled to run
void IofxManager::updateEffectsData(PxF32 deltaTime, PxU32 numObjects, PxU32 maxInputID, PxU32 maxStateID, void* extraData)
{
	PX_PROFILER_PLOT((PxU32)numObjects, "IofxManagerUpdateEffectsData");

	PX_ASSERT(maxStateID >= maxInputID && maxInputID >= numObjects);

	mLastNumObjects = numObjects;
	mLastMaxInputID = maxInputID;

	if (mCudaIos && !mCudaModifiers)
	{
#if defined(APEX_CUDA_SUPPORT)
		/* Presumably, updateEffectsData() is being called from a DtoH GPU task */
		mCopyQueue.reset(CUstream(extraData), 8);
		positionMass.copyDeviceToHostQ(mCopyQueue, maxInputID);
		velocityLife.copyDeviceToHostQ(mCopyQueue, maxInputID);
		if (mWorkingIosData->iosSupportsCollision)
		{
			collisionNormalFlags.copyDeviceToHostQ(mCopyQueue, maxInputID);
		}
		if (mWorkingIosData->iosSupportsDensity)
		{
			density.copyDeviceToHostQ(mCopyQueue, maxInputID);
		}
		if (mWorkingIosData->iosSupportsUserData)
		{
			userData.copyDeviceToHostQ(mCopyQueue, maxInputID);
		}
		actorIdentifiers.copyDeviceToHostQ(mCopyQueue, maxInputID);
		inStateToInput.copyDeviceToHostQ(mCopyQueue, maxStateID);
		mCopyQueue.flushEnqueued();

		mIofxScene->mApexScene->getTaskManager()->getGpuDispatcher()->addCompletionPrereq(mSimulateTask);
#else
		PX_ALWAYS_ASSERT();
#endif
	}

	/* Data from the IOS */
	mWorkingIosData->maxInputID = maxInputID;
	mWorkingIosData->maxStateID = maxStateID;
	mWorkingIosData->numParticles = numObjects;

	/* Data from the scene */
	mWorkingIosData->eyePosition = mIofxScene->mApexScene->getEyePosition();
	mWorkingIosData->eyeDirection = mIofxScene->mApexScene->getEyeDirection();

	PxMat44 viewMtx = mIofxScene->mApexScene->getViewMatrix();
	PxMat44 projMtx = mIofxScene->mApexScene->getProjMatrix();
	mWorkingIosData->eyeAxisX = PxVec3(viewMtx.column0.x, viewMtx.column1.x, viewMtx.column2.x);
	mWorkingIosData->eyeAxisY = PxVec3(viewMtx.column0.y, viewMtx.column1.y, viewMtx.column2.y);
	mWorkingIosData->zNear = physx::PxAbs(projMtx.column3.z / projMtx.column2.z);

	mWorkingIosData->deltaTime = deltaTime;
	// TODO: Convert into PxU32 elapsed milliseconds
	mTotalElapsedTime = numObjects ? mTotalElapsedTime + mWorkingIosData->deltaTime : 0;
	mWorkingIosData->elapsedTime = mTotalElapsedTime;

	/* IOFX data */
	mWorkingIosData->writeBufferCalled = false;

	if (mCudaModifiers)
	{
		mCudaPipeline->launchPrep(); // calls allocOutputs
	}
	else
	{
		//wait for outputData copy to render resource
		mWorkingIosData->waitForRenderDataUpdate();

		mWorkingIosData->allocOutputs();
	}
}

void IofxManager::cpuModifiers()
{
	if (!mCudaIos && mOnStartCallback)
	{
		(*mOnStartCallback)(NULL);
	}
	PxU32 maxInputID, maxStateID, numObjects;

	maxInputID = mWorkingIosData->maxInputID;
	maxStateID = mWorkingIosData->maxStateID;
	numObjects = mWorkingIosData->numParticles;

	PX_UNUSED(numObjects);

	/* Swap state buffer pointers */

	IosObjectCpuData* md = DYNAMIC_CAST(IosObjectCpuData*)(mWorkingIosData);

	md->inPubState = mStateSwap ? &pubState.a[0] : &pubState.b[0];
	md->outPubState = mStateSwap ? &pubState.b[0] : &pubState.a[0];

	md->inPrivState = mStateSwap ? &privState.a[0] : &privState.b[0];
	md->outPrivState = mStateSwap ? &privState.b[0] : &privState.a[0];

	swapStates();

	/* Sort sprites */

	if (!mIsMesh)
	{
		DYNAMIC_CAST(IosObjectCpuData*)(mWorkingIosData)->sortingKeys =
			mDistanceSortingEnabled ? &mSortingKeys.front() : NULL;
	}

	/* Volume Migration (1 pass) */

	mCountPerActor.clear();
	mCountPerActor.resize(mActorTable.size() * mVolumeTable.size(), 0);
	for (PxU32 input = 0 ; input < maxInputID ; input++)
	{
		NiIofxActorID& id = mWorkingIosData->pmaActorIdentifiers->get(input);
		if (id.getActorClassID() == NiIofxActorID::INV_ACTOR || id.getActorClassID() >= mActorClassTable.size())
		{
			id.set(NiIofxActorID::NO_VOLUME, NiIofxActorID::INV_ACTOR);
		}
		else
		{
			const PxVec3& pos = mWorkingIosData->pmaPositionMass->get(input).getXYZ();
			PxU32 curPri = 0;
			PxU16 curVID = NiIofxActorID::NO_VOLUME;

			for (PxU16 i = 0 ; i < mVolumeTable.size() ; i++)
			{
				VolumeData& vd = mVolumeTable[ i ];
				if (vd.vol == NULL)
				{
					continue;
				}

				const PxU32 bit = mActorClassTable.size() * i + id.getActorClassID();

				// This volume owns this particle if:
				//  1. The volume bounds contain the particle
				//  2. The volume affects the particle's IOFX Asset
				//  3. This volume has the highest priority or was the previous owner
				if (vd.mBounds.contains(pos) &&
				    (mVolumeActorClassBitmap[ bit >> 5 ] & (1u << (bit & 31))) &&
				    (curVID == NiIofxActorID::NO_VOLUME || vd.mPri > curPri || (vd.mPri == curPri && id.getVolumeID() == i)))
				{
					curVID = i;
					curPri = vd.mPri;
				}
			}

			id.setVolumeID(curVID);
		}

		// Count particles in each actor
		if (id.getVolumeID() != NiIofxActorID::NO_VOLUME)
		{
			const PxU32 actorID = mActorClassTable[ id.getActorClassID() ].actorID;
			PX_ASSERT(actorID < mActorTable.size());
			const PxU32 aid = id.getVolumeID() * mActorTable.size() + actorID;
			++mCountPerActor[aid];
		}
	}

	/* Prefix sum */
	mStartPerActor.clear();
	mStartPerActor.resize(mCountPerActor.size(), 0);
	PxU32 sum = 0;
	for (PxU32 i = 0 ; i < mStartPerActor.size() ; i++)
	{
		mStartPerActor[ i ] = sum;
		sum += mCountPerActor[ i ];
	}

	IosObjectCpuData* objData = DYNAMIC_CAST(IosObjectCpuData*)(mWorkingIosData);
	objData->outputToState = &mOutputToState.front();

	/* Generate outputToState (1 pass) */
	mBuildPerActor.clear();
	mBuildPerActor.resize(mStartPerActor.size(), 0);
	PxU32 homeless = 0;
	for (PxU32 state = 0 ; state < maxStateID ; state++)
	{
		PxU32 input = objData->pmaInStateToInput->get(state);
		if (input == NiIosBufferDesc::NOT_A_PARTICLE)
		{
			continue;
		}

		input &= ~NiIosBufferDesc::NEW_PARTICLE_FLAG;

		const NiIofxActorID id = objData->pmaActorIdentifiers->get(input);
		if (id.getVolumeID() == NiIofxActorID::NO_VOLUME)
		{
			objData->pmaOutStateToInput->get(sum + homeless) = input;
			++homeless;
		}
		else
		{
			PX_ASSERT(id.getActorClassID() != NiIofxActorID::INV_ACTOR && id.getActorClassID() < mActorClassTable.size());
			const PxU32 actorID = mActorClassTable[ id.getActorClassID() ].actorID;
			PX_ASSERT(actorID < mActorTable.size());
			const PxU32 aid = id.getVolumeID() * mActorTable.size() + actorID;
			objData->outputToState[ mStartPerActor[aid] + mBuildPerActor[ aid ]++ ] = state;
		}
	}

	/* Step IOFX Actors */
	PxU32 aid = 0;
	PxTaskManager* tm = mIofxScene->mApexScene->getTaskManager();
	for (PxU32 i = 0 ; i < mVolumeTable.size() ; i++)
	{
		VolumeData& d = mVolumeTable[ i ];
		if (d.vol == 0)
		{
			aid += mActorTable.size();
			continue;
		}

		for (PxU32 j = 0 ; j < mActorTable.size(); j++)
		{
			if (d.mActors[ j ] == DEFERRED_IOFX_ACTOR && mActorTable[ j ] != NULL &&
			        (mIofxScene->mModule->mDeferredDisabled || mCountPerActor[ aid ]))
			{
				IofxActor* iofxActor = PX_NEW(IofxActorCPU)(mActorTable[j]->getRenderAsset(), mIofxScene, *this);
				if (d.vol->addIofxActor(*iofxActor))
				{
					d.mActors[ j ] = iofxActor;

					initIofxActor(iofxActor, j, d.vol);
				}
				else
				{
					iofxActor->release();
				}
			}

			IofxActorCPU* cpuIofx = DYNAMIC_CAST(IofxActorCPU*)(d.mActors[ j ]);
			if (cpuIofx && cpuIofx != DEFERRED_IOFX_ACTOR)
			{
				if (mCountPerActor[ aid ])
				{
					ObjectRange range;
					range.objectCount = mCountPerActor[ aid ];
					range.startIndex = mStartPerActor[ aid ];
					PX_ASSERT(range.startIndex + range.objectCount <= numObjects);

					cpuIofx->mWorkingRange = range;

					cpuIofx->mModifierTask.setContinuation(*tm, tm->getTaskFromID(mPostUpdateTaskID));
					cpuIofx->mModifierTask.removeReference();
				}
				else
				{
					cpuIofx->mWorkingVisibleCount = 0;
					cpuIofx->mWorkingRange.objectCount = 0;
					cpuIofx->mWorkingRange.startIndex = 0;
					cpuIofx->mWorkingBounds.setEmpty();
				}
			}

			aid++;
		}
	}

	if (!mCudaIos && mOnFinishCallback)
	{
		(*mOnFinishCallback)(NULL);
	}
#if APEX_TEST
	if (mTestData != NULL)
	{
		mTestData->mIsCPUTest = true;
		mTestData->mOutStateToInput.resize(objData->pmaOutStateToInput->getSize());
		mTestData->mInStateToInput.resize(objData->pmaInStateToInput->getSize());
		for (PxU32 i = 0; i < objData->pmaOutStateToInput->getSize(); i++)
		{
			mTestData->mOutStateToInput[i] = objData->pmaOutStateToInput->get(i);
		}
		for (PxU32 i = 0; i < objData->pmaInStateToInput->getSize(); i++)
		{
			mTestData->mInStateToInput[i] = objData->pmaInStateToInput->get(i);
		}
		mTestData->mMaxInputID = objData->maxInputID;
		mTestData->mMaxStateID = objData->maxStateID;
		mTestData->mNumParticles = objData->numParticles;

		mTestData->mCountPerActor.resize(mCountPerActor.size());
		mTestData->mStartPerActor.resize(mStartPerActor.size());
		for (PxU32 i = 0; i < mCountPerActor.size(); i++)
		{
			mTestData->mCountPerActor[i] = mCountPerActor[i];
		}
		for (PxU32 i = 0; i < mStartPerActor.size(); i++)
		{
			mTestData->mStartPerActor[i] = mStartPerActor[i];
		}
	}
#endif

}

void IofxManager::outputHostToDevice(physx::PxGpuCopyDescQueue& copyQueue)
{
	if (mCudaIos && !mCudaModifiers)
	{
#if defined(APEX_CUDA_SUPPORT)
		actorIdentifiers.copyHostToDeviceQ(copyQueue, mLastMaxInputID);
		outStateToInput.copyHostToDeviceQ(copyQueue, mLastNumObjects);
#else
		PX_ALWAYS_ASSERT();
#endif
	}
}


void IofxManager::submitTasks()
{
	/* Discover new volumes, removed volumes */
	for (PxU32 i = 0 ; i < mVolumeTable.size() ; i++)
	{
		mVolumeTable[ i ].mFlags = 0;
	}

	for (PxU32 i = 0 ; i < mIofxScene->mLiveRenderVolumes.size() ; i++)
	{
		getVolumeID(mIofxScene->mLiveRenderVolumes[ i ]);
	}

	for (PxU32 i = 0 ; i < mVolumeTable.size() ; i++)
	{
		if (mVolumeTable[ i ].mFlags == 0)
		{
			mVolumeTable[ i ].vol = 0;
		}
	}

	/* Trim Volume, ActorID and ActorClassID tables */
	while (mVolumeTable.size() && mVolumeTable.back().vol == 0)
	{
		mVolumeTable.popBack();
	}

	if (!mActorTable.empty())
	{
		PxI32 lastValidID = -1;
		for (PxI32 cur = (physx::PxI32)mActorTable.size() - 1; cur >= 0; --cur)
		{
			if (mActorTable[(physx::PxU32)cur] != NULL)
			{
				lastValidID = cur;
				break;
			}
		}
		if (lastValidID == -1)
		{
			mActorTable.clear();
		}
		else
		{
			mActorTable.resize((physx::PxU32)lastValidID + 1);
		}
	}

	if (!mActorClassTable.empty())
	{
		PxI32 lastValidID = -1;
		for (PxU32 cur = 0; cur < mActorClassTable.size(); cur += mActorClassTable[ cur ].count)
		{
			if (mActorClassTable[ cur ].client != NULL)
			{
				lastValidID = (physx::PxI32)cur;
			}
		}
		if (lastValidID == -1)
		{
			mActorClassTable.clear();
		}
		else
		{
			mActorClassTable.resize((physx::PxU32)(lastValidID + mActorClassTable[ (physx::PxU32)lastValidID ].count));
		}
	}

	const physx::PxU32 volumeActorClassBitmapSize = (mVolumeTable.size() * mActorClassTable.size() + 31) >> 5;
	mVolumeActorClassBitmap.resize(volumeActorClassBitmapSize);
	for (PxU32 i = 0 ; i < volumeActorClassBitmapSize ; i++)
	{
		mVolumeActorClassBitmap[ i ] = 0;
	}

	/* Add new IofxActors as necessary */
	for (PxU32 i = 0 ; i < mVolumeTable.size() ; i++)
	{
		VolumeData& d = mVolumeTable[ i ];

		// First, ensure per-volume actor array can hold all ClassIDs
		d.mActors.resize(mActorTable.size(), 0);

		if (d.vol == NULL)
		{
			continue;
		}

		d.mBounds = d.vol->getOwnershipBounds();
		d.mPri = d.vol->getPriority();

		for (PxU32 cur = 0; cur < mActorTable.size(); ++cur)
		{
			if (mActorTable[ cur ] != NULL)
			{
				if (!d.mActors[ cur ])
				{
					d.mActors[ cur ] = DEFERRED_IOFX_ACTOR;
				}
			}
		}

		d.vol->renderDataLock(); // for safety during affectsIofxAsset() calls
		for (PxU32 cur = 0; cur < mActorClassTable.size(); ++cur)
		{
			const ActorClassData& acd = mActorClassTable[ cur ];
			if (acd.client != NULL && acd.actorID < mActorTable.size())
			{
				IofxAsset* iofxAsset = acd.client->getAssetSceneInst()->getAsset();
				if (iofxAsset && d.vol->affectsIofxAsset(*iofxAsset))
				{
					const PxU32 bit = mActorClassTable.size() * i + cur;
					mVolumeActorClassBitmap[ bit >> 5 ] |= (1u << (bit & 31));
				}
			}
		}
		d.vol->renderDataUnLock(); // for safety during affectsIofxAsset() calls
	}

	PxU32 targetSemantics = 0;
	mDistanceSortingEnabled = false;
	{
		for (AssetHashMap_t::Iterator it = mAssetHashMap.getIterator(); !it.done(); ++it)
		{
			IofxAsset* iofxAsset = it->first;
			IofxAssetSceneInst* iofxAssetSceneInst = it->second;

			targetSemantics |= iofxAssetSceneInst->getSemantics();
			if (!mDistanceSortingEnabled && iofxAsset->isSortingEnabled())
			{
				mDistanceSortingEnabled = true;
				if (!mCudaModifiers)
				{
					mSortingKeys.resize(mOutputToState.size());
				}
			}
		}
	}
	mTargetSemantics = targetSemantics;

	if (mCudaModifiers)
	{
		mCudaPipeline->submitTasks();
	}
	else
	{
		mWorkingIosData->updateSemantics(mTargetSemantics, false);
	}

	if (!addedAssets.empty())
	{
		/* Calculate state sizes required by new assets */
		PxU32 newPubStateSize = 0, newPrivStateSize = 0;
		for (PxU32 i = 0; i < addedAssets.size(); ++i)
		{
			newPubStateSize = PxMax(newPubStateSize, addedAssets[i]->getPubStateSize());
			newPrivStateSize = PxMax(newPrivStateSize, addedAssets[i]->getPrivStateSize());
		}

		PxU32 maxObjectCount = outStateToInput.getSize(),
			totalCount = mOutStateOffset + maxObjectCount;

		// Allocate data for pubstates
		while (newPubStateSize > pubStateSize)
		{
			pubStateSize += sizeof(IofxSlice);

			SliceArray* slice = new SliceArray(*mIofxScene->mApexScene, NV_ALLOC_INFO("slice", PARTICLES));

#if defined(APEX_CUDA_SUPPORT)
			if (mCudaModifiers)
			{
				//slice->reserve(totalCount, ApexMirroredPlace::GPU); Recalculated on GPU
			}
			else
#endif
			{
				slice->reserve(totalCount, ApexMirroredPlace::CPU);
			}

			pubState.slices.pushBack(slice);

			IofxSlice* p;
#if defined(APEX_CUDA_SUPPORT)
			p = mCudaModifiers
				? pubState.slices.back()->getGpuPtr()
				: pubState.slices.back()->getPtr();
#else
			p = pubState.slices.back()->getPtr();
#endif
			pubState.a.pushBack(p + mInStateOffset);
			pubState.b.pushBack(p + mOutStateOffset);
		}

		// Allocate data for privstates
		while (newPrivStateSize > privStateSize)
		{
			privStateSize += sizeof(IofxSlice);

			SliceArray* slice = new SliceArray(*mIofxScene->mApexScene, NV_ALLOC_INFO("slice", PARTICLES));

#if defined(APEX_CUDA_SUPPORT)
			if (mCudaModifiers)
			{
				slice->reserve(totalCount, ApexMirroredPlace::GPU);
			}
			else
#endif
			{
				slice->reserve(totalCount, ApexMirroredPlace::CPU);
			}

			privState.slices.pushBack(slice);

			IofxSlice* p;
#if defined(APEX_CUDA_SUPPORT)
			p = mCudaModifiers 
				? privState.slices.back()->getGpuPtr()
				: privState.slices.back()->getPtr();
#else
			p = privState.slices.back()->getPtr();
#endif
			privState.a.pushBack(p + mInStateOffset);
			privState.b.pushBack(p + mOutStateOffset);
		}

		addedAssets.clear();
	}
}

void IofxManager::swapStates()
{
	mStateSwap = !mStateSwap;
	swap(mInStateOffset, mOutStateOffset);
}

void IofxManager::fetchResults()
{
	if (!mPostUpdateTaskID)
	{
		return;
	}
	mPostUpdateTaskID = 0;

	if (mCudaModifiers)
	{
		mCudaPipeline->fetchResults();
	}
	else
	{
		for (PxU32 i = 0 ; i < mVolumeTable.size() ; i++)
		{
			VolumeData& d = mVolumeTable[ i ];
			for (PxU32 j = 0 ; j < mActorTable.size() ; j++)
			{
				IofxActorCPU* cpuIofx = DYNAMIC_CAST(IofxActorCPU*)(d.mActors[ j ]);
				if (cpuIofx && cpuIofx != DEFERRED_IOFX_ACTOR)
				{
					cpuIofx->mResultBounds = cpuIofx->mWorkingBounds;
					cpuIofx->mResultRange = cpuIofx->mWorkingRange;
					cpuIofx->mResultVisibleCount = cpuIofx->mWorkingVisibleCount;
				}
			}
		}
	}

	//build bounds
	{
		mBounds.setEmpty();
		for (PxU32 i = 0 ; i < mVolumeTable.size() ; i++)
		{
			VolumeData& d = mVolumeTable[ i ];
			for (PxU32 j = 0 ; j < mActorTable.size() ; j++)
			{
				IofxActor* iofx = d.mActors[ j ];
				if (iofx && iofx != DEFERRED_IOFX_ACTOR)
				{
					mBounds.include(iofx->mResultBounds);
				}
			}
		}
	}

	//swap ObjectData
	bool bSwapObjectData = true;
	if (mCudaModifiers)
	{
		bSwapObjectData = mCudaPipeline->swapObjectData();
	}
	if (bSwapObjectData)
	{
		physx::swap(mResultIosData, mWorkingIosData);
		mResultReadyState = RESULT_READY;
	}
}

PxBounds3 IofxManager::getBounds() const
{
	return mBounds;
}

PxU32 IofxManager::getActorID(IofxAssetSceneInst* assetSceneInst, PxU16 meshID)
{
	IofxAsset* iofxAsset = assetSceneInst->getAsset();

	NxApexAsset* renderAsset = NULL;
	if (mIsMesh)
	{
		const char* rmName = iofxAsset->getMeshAssetName(meshID);
		bool isOpaqueMesh = iofxAsset->isOpaqueMesh(meshID);
		renderAsset = iofxAsset->mRenderMeshAssetTracker.getMeshAssetFromName(rmName, isOpaqueMesh);
		if (renderAsset == NULL)
		{
			APEX_INVALID_PARAMETER("IofxManager: ApexRenderMeshAsset with name \"%s\" not found.", rmName);
		}
	}
	else
	{
		const char* mtlName = iofxAsset->getSpriteMaterialName();
		renderAsset = iofxAsset->mSpriteMaterialAssetTracker.getAssetFromName(mtlName);
		if (renderAsset == NULL)
		{
			APEX_INVALID_PARAMETER("IofxManager: SpriteMaterial with name \"%s\" not found.", mtlName);
		}
	}
	PxU32 actorID = PxU32(-1);
	if (renderAsset != NULL)
	{
		for (PxU32 id = 0 ; id < mActorTable.size() ; id++)
		{
			if (mActorTable[id] != NULL)
			{
				if (mActorTable[id]->getRenderAsset() == renderAsset)
				{
					actorID = id;
					break;
				}
			}
			else if (actorID == PxU32(-1))
			{
				actorID = id;
			}
		}
		if (actorID == PxU32(-1))
		{
			actorID = mActorTable.size();
			mActorTable.resize(actorID + 1, NULL);
		}

		IofxActorSceneInst* &actorSceneInst = mActorTable[actorID];
		if (actorSceneInst == NULL)
		{
			actorSceneInst = PX_NEW(IofxActorSceneInst)(renderAsset);
		}
		actorSceneInst->addRef();

		// only add iofxAsset one time, check refCount
		if (assetSceneInst->getRefCount() == 1)
		{
			actorSceneInst->addAssetSceneInst(assetSceneInst);
		}
	}
	return actorID;
}
void IofxManager::releaseActorID(IofxAssetSceneInst* assetSceneInst, PxU32 actorID)
{
	PX_ASSERT(actorID < mActorTable.size());
	IofxActorSceneInst* &actorSceneInst = mActorTable[actorID];
	if (actorSceneInst != NULL)
	{
		PX_ASSERT(actorSceneInst->getRefCount() > 0);
		if (actorSceneInst->removeRef())
		{
			for (PxU16 j = 0 ; j < mVolumeTable.size() ; j++)
			{
				if (mVolumeTable[ j ].vol == NULL)
				{
					continue;
				}

				if (actorID < mVolumeTable[ j ].mActors.size())
				{
					IofxActor* iofxActor = mVolumeTable[ j ].mActors[ actorID ];
					if (iofxActor && iofxActor != DEFERRED_IOFX_ACTOR)
					{
						iofxActor->release();
						//IofxManager::removeActorAtIndex should zero the actor in mActors
						PX_ASSERT(mVolumeTable[ j ].mActors[ actorID ] == 0);
					}
					mVolumeTable[ j ].mActors[ actorID ] = 0;
				}
			}

			PX_DELETE(actorSceneInst);
			actorSceneInst = NULL;
		}
		else
		{
			// only remove iofxAsset one time, check refCount
			if (assetSceneInst->getRefCount() == 1)
			{
				actorSceneInst->removeAssetSceneInst(assetSceneInst);
			}
		}
	}
}

PxU16 IofxManager::getActorClassID(NiIofxManagerClient* iofxClient, PxU16 meshID)
{
	IofxManagerClient* client = static_cast<IofxManagerClient*>(iofxClient);
	if (client != 0)
	{
		const PxU16 actorClassID = PxU16(client->getActorClassID() + meshID);
		PX_ASSERT(actorClassID < mActorClassTable.size());
		PX_ASSERT(mActorClassTable[actorClassID].client == client);
		PX_ASSERT(meshID < mActorClassTable[actorClassID].count);

		const PxU32 actorID = mActorClassTable[actorClassID].actorID;
		if (actorID != PxU32(-1))
		{
			return actorClassID;
		}
		else
		{
			APEX_DEBUG_WARNING("IofxManager: getActorClassID returned invalid actor.");
			return NiIofxActorID::INV_ACTOR;
		}
	}
	else
	{
		APEX_INVALID_PARAMETER("IofxManager: getActorClassID was called with invalid client.");
		return NiIofxActorID::INV_ACTOR;
	}
}

NiIofxManagerClient* IofxManager::createClient(physx::apex::NxIofxAsset* asset, const NiIofxManagerClient::Params& params)
{
	IofxAsset* iofxAsset = static_cast<IofxAsset*>(asset);

	IofxAssetSceneInst* &assetSceneInst = mAssetHashMap[iofxAsset];
	if (assetSceneInst == NULL)
	{
		assetSceneInst = createAssetSceneInst(iofxAsset);
		// Update state sizes later in submitTasks
		addedAssets.pushBack(iofxAsset);
		// increase asset refCount
		//NxResourceProvider* nrp = NiGetApexSDK()->getNamedResourceProvider();
		//nrp->setResource(NX_IOFX_AUTHORING_TYPE_NAME, asset->getName(), asset, true);
	}
	assetSceneInst->addRef();

	//allocate actorClasses
	PxU16 actorClassCount = PxU16( PxMax(1u, iofxAsset->getMeshAssetCount()) );
	PxU32 actorClassID = 0;
	while (actorClassID < mActorClassTable.size())
	{
		ActorClassData& acd = mActorClassTable[ actorClassID ];

		if (acd.client == NULL && actorClassCount <= acd.count)
		{
			/* Make a shim to conver remaining hole */
			PxU16 remains = PxU16(acd.count - actorClassCount);
			for (PxU16 i = 0 ; i < remains ; i++)
			{
				ActorClassData& acd1 = mActorClassTable[ actorClassID + actorClassCount + i ];
				acd1.client = 0;
				acd1.meshid = i;
				acd1.count = remains;
				acd1.actorID = PxU32(-1);
			}
			break;
		}
		actorClassID = actorClassID + acd.count;
	}
	if (actorClassID >= mActorClassTable.size())
	{
		/* Asset is not in table, append it */
		actorClassID = mActorClassTable.size();
		mActorClassTable.resize(actorClassID + actorClassCount);
	}

	IofxManagerClient* client = NULL;
#if defined(APEX_CUDA_SUPPORT)
	if (mCudaModifiers)
	{
		client = mCudaPipeline->createClient(assetSceneInst, actorClassID, params);
	}
	else
#endif
	{
		client = PX_NEW(IofxManagerClient)(assetSceneInst, actorClassID, params);
	}
	PX_ASSERT(client != NULL);

	for (PxU16 i = 0 ; i < actorClassCount ; i++)
	{
		ActorClassData& acd = mActorClassTable[ actorClassID + i ];
		acd.client = client;
		acd.meshid = i;
		acd.count = actorClassCount;
		acd.actorID = getActorID(assetSceneInst, i);
	}

	return static_cast<NiIofxManagerClient*>(client);
}

void IofxManager::releaseClient(NiIofxManagerClient* iofxClient)
{
	// TODO: free unused memory in states

	IofxManagerClient* client = static_cast<IofxManagerClient*>(iofxClient);
	if (client != 0)
	{
		IofxAssetSceneInst* assetSceneInst = client->getAssetSceneInst();
		PxU32 actorClassID = PxU16(client->getActorClassID());

		if (actorClassID < mActorClassTable.size())
		{
			for (PxU16 i = 0 ; i < mActorClassTable[ actorClassID ].count ; i++)
			{
				PxU32 actorID = mActorClassTable[ actorClassID + i ].actorID;
				if (actorID != PxU32(-1))
				{
					releaseActorID(assetSceneInst, actorID);
				}
			}

			// TODO: merge backward hole also
			/* merge consecutive holes */
			PxU32 next = actorClassID + mActorClassTable[ actorClassID ].count;
			while (next < mActorClassTable.size() && mActorClassTable[ next ].client == NULL)
			{
				next = next + mActorClassTable[ next ].count;
			}

			PxU16 count = PxU16(next - actorClassID);
			for (PxU16 i = 0 ; i < count ; i++)
			{
				ActorClassData& acd = mActorClassTable[ actorClassID + i ];
				acd.client = 0;
				acd.meshid = i;
				acd.count = count;
				acd.actorID = PxU32(-1);
			}
		}

		PX_DELETE(client);

		if (assetSceneInst->removeRef())
		{
			// decrease asset refCount
			//NxResourceProvider* nrp = NiGetApexSDK()->getNamedResourceProvider();
			//nrp->releaseResource(NX_IOFX_AUTHORING_TYPE_NAME, ad.asset->getName());

			IofxAsset* iofxAsset = assetSceneInst->getAsset();
			PX_DELETE(assetSceneInst);
			assetSceneInst = NULL;

			mAssetHashMap.erase(iofxAsset);
		}
	}
}

PxU16 IofxManager::getVolumeID(NxApexRenderVolume* vol)
{
	PxI32 hole = -1;
	for (PxU16 i = 0 ; i < mVolumeTable.size() ; i++)
	{
		if (vol == mVolumeTable[ i ].vol)
		{
			mVolumeTable[ i ].mFlags = 1;
			return i;
		}
		else if (hole == -1 && !mVolumeTable[ i ].vol)
		{
			hole = (PxI32) i;
		}
	}
	if (hole == -1)
	{
		mVolumeTable.insert();
		hole = (physx::PxI32)mVolumeTable.size() - 1;
	}
	VolumeData& d = mVolumeTable[ (physx::PxU32)hole ];
	d.vol = DYNAMIC_CAST(ApexRenderVolume*)(vol);
	d.mFlags = 1;
	d.mActors.clear(); //Iofx Actors are released in ApexRenderVolume destructor!
	return (PxU16) hole;
}


void IofxManager::removeActorAtIndex(PxU32 index)
{
	IofxActor* iofx = DYNAMIC_CAST(IofxActor*)(mActorArray[ index ]);

	for (PxU32 i = 0 ; i < mVolumeTable.size() ; i++)
	{
		if (mVolumeTable[ i ].vol == iofx->mRenderVolume)
		{
			PX_ASSERT(iofx == mVolumeTable[ i ].mActors[ iofx->mActorID ]);
			mVolumeTable[ i ].mActors[ iofx->mActorID ] = 0;
		}
	}

	ApexContext::removeActorAtIndex(index);
}

IofxAssetSceneInst* IofxManager::createAssetSceneInst(NxIofxAsset* nxAsset)
{
	IofxAsset* asset = DYNAMIC_CAST(IofxAsset*)( nxAsset );

	PxU32 semantics = physx::PxU32(mIsMesh ? BASE_MESH_SEMANTICS : BASE_SPRITE_SEMANTICS);
	if( mObjData[0]->iosSupportsDensity ) {
		semantics |= mIsMesh ? (PxU32)NxRenderInstanceSemantic::DENSITY : (PxU32)NxRenderSpriteSemantic::DENSITY;
	}
	semantics |= mIsMesh ? asset->getMeshSemanticsBitmap() : asset->getSpriteSemanticsBitmap();

	IofxAssetSceneInst* assetSceneInst = 0;
#if defined(APEX_CUDA_SUPPORT)
	if (mCudaModifiers)
	{
		assetSceneInst = mCudaPipeline->createAssetSceneInst(asset, semantics);
	}
	else
#endif
	{
		assetSceneInst = PX_NEW(IofxAssetSceneInstCPU)(asset, semantics, mIofxScene);
	}
	PX_ASSERT(assetSceneInst != 0);
	return assetSceneInst;
}

void IofxManager::initIofxActor(IofxActor* iofxActor, PxU32 actorID, ApexRenderVolume* renderVolume)
{
	iofxActor->addSelfToContext(*this);
	iofxActor->mActorID = actorID;
	iofxActor->mRenderVolume = renderVolume;
	iofxActor->mSemantics = 0;
	iofxActor->mDistanceSortingEnabled = false;

	PX_ASSERT(mActorTable[actorID] != NULL);
	const physx::Array<IofxAssetSceneInst*>& iofxAssets = mActorTable[actorID]->getAssetSceneInstArray();
	for (PxU32 k = 0; k < iofxAssets.size(); ++k)
	{
		IofxAssetSceneInst* assetSceneInst = iofxAssets[k];

		iofxActor->mSemantics |= assetSceneInst->getSemantics();
		iofxActor->mDistanceSortingEnabled |= assetSceneInst->getAsset()->isSortingEnabled();
	}

}

#ifdef APEX_TEST
IofxManagerTestData* IofxManager::createTestData()
{
	mTestData = new IofxManagerTestData();
	return mTestData;
}

void IofxManager::copyTestData() const
{
	if (mTestData == NULL)
	{
		return;
	}

	mTestData->mNOT_A_PARTICLE = NiIosBufferDesc::NOT_A_PARTICLE;
	mTestData->mNEW_PARTICLE_FLAG = NiIosBufferDesc::NEW_PARTICLE_FLAG;
	mTestData->mSTATE_ID_MASK = STATE_ID_MASK;
}
void IofxManager::clearTestData()
{
	mTestData->mInStateToInput.reset();
	mTestData->mOutStateToInput.reset();
	mTestData->mCountPerActor.reset();
	mTestData->mStartPerActor.reset();
	mTestData->mPositionMass.reset();
	mTestData->mSortedActorIDs.reset();
	mTestData->mSortedStateIDs.reset();
	mTestData->mActorStart.reset();
	mTestData->mActorEnd.reset();
	mTestData->mActorVisibleEnd.reset();
	mTestData->mMinBounds.reset();
	mTestData->mMaxBounds.reset();
}
#endif

}
}
} // end namespace physx::apex
