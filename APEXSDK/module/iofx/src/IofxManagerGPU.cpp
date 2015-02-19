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
#include "NxApexDefs.h"

#if defined(APEX_CUDA_SUPPORT)

#include "NiApexSDK.h"
#include "NiApexScene.h"
#include "Modifier.h"
#include "NxIofxActor.h"
#include "IofxManagerGPU.h"
#include "IofxAsset.h"
#include "IofxSceneGPU.h"

#include "ModuleIofx.h"
#include "IofxActorGPU.h"

#include "PxGpuTask.h"
#include "ApexCutil.h"

#include "RandStateHelpers.h"

#include "IofxRenderData.h"

#ifdef APEX_TEST
#include "IofxManagerTestData.h"
#endif

#define CUDA_OBJ(name) SCENE_CUDA_OBJ(mIofxScene, name)

namespace physx
{
namespace apex
{
namespace iofx
{

class IofxAssetSceneInstGPU : public IofxAssetSceneInst
{
public:
	IofxAssetSceneInstGPU(IofxAsset* asset, PxU32 semantics, IofxScene* scene)
		: IofxAssetSceneInst(asset, semantics)
		, _constMemGroup(SCENE_CUDA_OBJ(*scene, modifierStorage))
	{
		_totalRandomCount = 0;

		APEX_CUDA_CONST_MEM_GROUP_SCOPE(_constMemGroup)

		_storage_.alloc(_assetParamsHandle);
		AssetParams assetParams;
		buildModifierList(assetParams.spawnModifierList, _asset->mSpawnModifierStack);
		buildModifierList(assetParams.continuousModifierList, _asset->mContinuousModifierStack);
		_storage_.update(_assetParamsHandle, assetParams);
	}
	virtual ~IofxAssetSceneInstGPU() {}

	InplaceHandle<AssetParams> getAssetParamsHandle() const
	{
		return _assetParamsHandle;
	}

private:

	void buildModifierList(ModifierList& list, const ModifierStack& stack)
	{
		InplaceStorage& _storage_ = _constMemGroup.getStorage();

		class Mapper : public ModifierParamsMapperGPU
		{
		public:
			InplaceStorage* storage;

			InplaceHandleBase paramsHandle;
			physx::PxU32 paramsRandomCount;

			virtual InplaceStorage& getStorage()
			{
				return *storage;
			}

			virtual void  onParams(InplaceHandleBase handle, physx::PxU32 randomCount)
			{
				paramsHandle = handle;
				paramsRandomCount = randomCount;
			}

		} mapper;
		mapper.storage = &_storage_;

		list.resize(_storage_, stack.size());

		PxU32 index = 0;
		for (ModifierStack::ConstIterator it = stack.begin(); it != stack.end(); ++it)
		{
			PxU32 type = (*it)->getModifierType();
			//NxU32 usage = (*it)->getModifierUsage();
			//if ((usage & usageStage) == usageStage && (usage & usageClass) == usageClass)
			{
				const Modifier* modifier = Modifier::castFrom(*it);
				modifier->mapParamsGPU(mapper);

				ModifierListElem listElem;
				listElem.type = type;
				listElem.paramsHandle = mapper.paramsHandle;
				list.updateElem(_storage_, listElem, index);

				_totalRandomCount += mapper.paramsRandomCount;
			}
			++index;
		}
	}

	ApexCudaConstMemGroup		_constMemGroup;
	InplaceHandle<AssetParams>	_assetParamsHandle;
	physx::PxU32				_totalRandomCount;
};

class IofxManagerClientGPU : public IofxManagerClient
{
public:
	IofxManagerClientGPU(IofxAssetSceneInst* assetSceneInst, PxU32 actorClassID, const NiIofxManagerClient::Params& params, IofxScene* scene)
		: IofxManagerClient(assetSceneInst, actorClassID, params)
		, _constMemGroup(SCENE_CUDA_OBJ(*scene, modifierStorage))
	{
		setParamsGPU();
	}

	InplaceHandle<ClientParams> getClientParamsHandle() const
	{
		return _clientParamsHandle;
	}

	// NiIofxManagerClient interface
	virtual void setParams(const NiIofxManagerClient::Params& params)
	{
		IofxManagerClient::setParams(params);
		setParamsGPU();
	}

private:
	void setParamsGPU()
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(_constMemGroup)

		ClientParams clientParams;
		if (_clientParamsHandle.allocOrFetch(_storage_, clientParams))
		{
			clientParams.assetParamsHandle = static_cast<IofxAssetSceneInstGPU*>(_assetSceneInst)->getAssetParamsHandle();
		}
		clientParams.objectScale = _params.objectScale;
		_clientParamsHandle.update(_storage_, clientParams);
	}

	ApexCudaConstMemGroup		_constMemGroup;
	InplaceHandle<ClientParams>	_clientParamsHandle;
};


IofxManagerClient* IofxManagerGPU::createClient(IofxAssetSceneInst* assetSceneInst, PxU32 actorClassID, const NiIofxManagerClient::Params& params)
{
	return PX_NEW(IofxManagerClientGPU)(assetSceneInst, actorClassID, params, &mIofxScene);
}

IofxAssetSceneInst* IofxManagerGPU::createAssetSceneInst(IofxAsset* asset,PxU32 semantics)
{
	return PX_NEW(IofxAssetSceneInstGPU)(asset, semantics, &mIofxScene);
}

class IofxManagerLaunchTask : public physx::PxGpuTask, public physx::UserAllocated
{
public:
	IofxManagerLaunchTask(IofxManagerGPU* actor) : mActor(actor) {}
	const char* getName() const
	{
		return "IofxManagerLaunchTask";
	}
	void         run()
	{
		PX_ALWAYS_ASSERT();
	}
	bool         launchInstance(CUstream stream, int kernelIndex)
	{
		return mActor->cudaLaunch(stream, kernelIndex);
	}
	physx::PxGpuTaskHint::Enum getTaskHint() const
	{
		return physx::PxGpuTaskHint::Kernel;
	}

protected:
	IofxManagerGPU* mActor;
};

IofxManagerGPU::IofxManagerGPU(NiApexScene& scene, const NiIofxManagerDesc& desc, IofxManager& mgr, const ApexMirroredPlace::Enum defaultPlace)
	: mManager(mgr)
	, mIofxScene(*mgr.mIofxScene)
	, mCopyQueue(*scene.getTaskManager()->getGpuDispatcher())
	, mDefaultPlace(defaultPlace)
	, mCuSpawnScale(scene)
	, mCuSpawnSeed(scene)
	, mCuBlockPRNGs(scene)
	, mCuSortedActorIDs(scene)
	, mCuSortedStateIDs(scene)
	, mCuSortTempKeys(scene)
	, mCuSortTempValues(scene)
	, mCuSortTemp(scene)
	, mCuMinBounds(scene)
	, mCuMaxBounds(scene)
	, mCuTempMinBounds(scene)
	, mCuTempMaxBounds(scene)
	, mCuTempActorIDs(scene)
	, mCuActorStart(scene)
	, mCuActorEnd(scene)
	, mCuActorVisibleEnd(scene)
	, mCurSeed(0)
	, mTargetBufDevPtr(NULL)
	, mCountActorIDs(0)
	, mNumberVolumes(0)
	, mNumberActorClasses(0)
	, mEmptySimulation(false)
	, mVolumeConstMemGroup(CUDA_OBJ(migrationStorage))
	, mRemapConstMemGroup(CUDA_OBJ(remapStorage))
	, mModifierConstMemGroup(CUDA_OBJ(modifierStorage))
	, mOutputToBuffer(false)
	, mInteropStateCopy(mgr.mInteropState)
{
	mTaskLaunch = PX_NEW(IofxManagerLaunchTask)(this);

	const PxU32 maxObjectCount = desc.maxObjectCount;
	const PxU32 maxInStateCount = desc.maxInStateCount;
	PxU32 usageClass = 0;
	PxU32 blockSize = MAX_THREADS_PER_BLOCK;

	if (mManager.mIsMesh)
	{
		usageClass = ModifierUsage_Mesh;
		//blockSize = CUDA_OBJ(meshModifiersKernel).getBlockDim().x;
	}
	else
	{
		usageClass = ModifierUsage_Sprite;
		//blockSize = CUDA_OBJ(spriteModifiersKernel).getBlockDim().x;
	}

	mCuSpawnScale.reserve(mManager.mOutStateOffset + maxObjectCount, ApexMirroredPlace::GPU);
	mCuSpawnSeed.reserve(mManager.mOutStateOffset + maxObjectCount, ApexMirroredPlace::GPU);

	mCuSortedActorIDs.reserve(maxInStateCount, defaultPlace);
	mCuSortedStateIDs.reserve(maxInStateCount, defaultPlace);

	mCuSortTempKeys.reserve(maxInStateCount, ApexMirroredPlace::GPU);
	mCuSortTempValues.reserve(maxInStateCount, ApexMirroredPlace::GPU);
	mCuSortTemp.reserve(MAX_BOUND_BLOCKS * NEW_SORT_KEY_DIGITS, ApexMirroredPlace::GPU);

	mCuTempMinBounds.reserve(WARP_SIZE * 2, ApexMirroredPlace::GPU);
	mCuTempMaxBounds.reserve(WARP_SIZE * 2, ApexMirroredPlace::GPU);
	mCuTempActorIDs.reserve(WARP_SIZE * 2, ApexMirroredPlace::GPU);

	// alloc volumeConstMem
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(mVolumeConstMemGroup)

		mVolumeParamsArrayHandle.alloc(_storage_);
		mActorClassIDBitmapArrayHandle.alloc(_storage_);
	}

	// alloc remapConstMem
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(mRemapConstMemGroup)

		mActorIDRemapArrayHandle.alloc(_storage_);
	}

	// alloc modifierConstMem
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(mModifierConstMemGroup)

		mClientParamsHandleArrayHandle.alloc(_storage_);

		if (mManager.mIsMesh)
		{
			mMeshOutputLayoutHandle.alloc(_storage_);
		}
		else
		{
			mSpriteOutputLayoutHandle.alloc(_storage_);
		}
	}

	InitDevicePRNGs(scene, blockSize, mRandThreadLeap, mRandGridLeap, mCuBlockPRNGs);
}

void IofxManagerGPU::release()
{
	delete this;
}

IofxManagerGPU::~IofxManagerGPU()
{
	delete mTaskLaunch;
}


void IofxManagerGPU::submitTasks()
{
	if (mInteropStateCopy <= IofxManager::INTEROP_FAILED)
	{
		mManager.mWorkingIosData->updateSemantics(mManager.mTargetSemantics, false);
	}

	mNumberActorClasses = mManager.mActorClassTable.size();
	mNumberVolumes = mManager.mVolumeTable.size();
	mCountActorIDs = mManager.mActorTable.size() * mNumberVolumes;

	// update volumeConstMem
	if (mNumberVolumes)
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(mVolumeConstMemGroup)

		VolumeParamsArray volumeParamsArray;
		_storage_.fetch(mVolumeParamsArrayHandle, volumeParamsArray);
		volumeParamsArray.resize(_storage_, mNumberVolumes);
		_storage_.update(mVolumeParamsArrayHandle, volumeParamsArray);


		ActorClassIDBitmapArray actorClassIDBitmapArray;
		_storage_.fetch(mActorClassIDBitmapArrayHandle, actorClassIDBitmapArray);
		actorClassIDBitmapArray.resize(_storage_, mManager.mVolumeActorClassBitmap.size());
		_storage_.update(mActorClassIDBitmapArrayHandle, actorClassIDBitmapArray);

		actorClassIDBitmapArray.updateRange(_storage_, &mManager.mVolumeActorClassBitmap.front(), actorClassIDBitmapArray.getSize());

		for (PxU32 i = 0 ; i < mNumberVolumes ; i++)
		{
			VolumeParams volumeParams;
			IofxManager::VolumeData& vd = mManager.mVolumeTable[ i ];
			if (vd.vol)
			{
				volumeParams.bounds = vd.mBounds;
				volumeParams.priority = vd.mPri;
			}
			else
			{
				volumeParams.bounds.setEmpty();
				volumeParams.priority = 0;
			}
			volumeParamsArray.updateElem(_storage_, volumeParams, i);
		}
	}
	else
	{
		APEX_DEBUG_WARNING("IofxManager: There is no render volume!");
	}

	// update remapConstMem
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(mRemapConstMemGroup)

		ActorIDRemapArray actorIDRemapArray;
		_storage_.fetch(mActorIDRemapArrayHandle, actorIDRemapArray);
		actorIDRemapArray.resize(_storage_, mNumberActorClasses);
		for (physx::PxU32 i = 0 ; i < mNumberActorClasses ; ++i)
		{
			actorIDRemapArray.updateElem(_storage_, mManager.mActorClassTable[i].actorID, i);
		}
		_storage_.update(mActorIDRemapArrayHandle, actorIDRemapArray);
	}

	// update modifierConstMem
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(mModifierConstMemGroup)

		ClientParamsHandleArray clientParamsHandleArray;
		_storage_.fetch(mClientParamsHandleArrayHandle, clientParamsHandleArray);
		clientParamsHandleArray.resize(_storage_, mNumberActorClasses);
		for (physx::PxU32 i = 0 ; i < mNumberActorClasses ; ++i)
		{
			InplaceHandle<ClientParams> clientParamsHandle;
			IofxManagerClientGPU* clientGPU = static_cast<IofxManagerClientGPU*>(mManager.mActorClassTable[i].client);
			if (clientGPU != NULL)
			{
				clientParamsHandle = clientGPU->getClientParamsHandle();
			}
			clientParamsHandleArray.updateElem(_storage_, clientParamsHandle, i);
		}
		_storage_.update(mClientParamsHandleArrayHandle, clientParamsHandleArray);

		if (mManager.mIsMesh)
		{
			MeshOutputLayout meshOutputLayout;

			IosObjectGpuData* mWorkingData = DYNAMIC_CAST(IosObjectGpuData*)(mManager.mWorkingIosData);
			IofxOutputDataMesh* meshOutputData = DYNAMIC_CAST(IofxOutputDataMesh*)(mWorkingData->outputData);
			const NxUserRenderInstanceBufferDesc& instanceBufferDesc = meshOutputData->getVertexDesc();

			mOutputDWords = instanceBufferDesc.stride >> 2;
			meshOutputLayout.stride = instanceBufferDesc.stride;
			::memcpy(meshOutputLayout.offsets, instanceBufferDesc.semanticOffsets, sizeof(meshOutputLayout.offsets));

			_storage_.update(mMeshOutputLayoutHandle, meshOutputLayout);
		}
		else
		{
			SpriteOutputLayout spriteOutputLayout;

			IosObjectGpuData* mWorkingData = DYNAMIC_CAST(IosObjectGpuData*)(mManager.mWorkingIosData);
			IofxOutputDataSprite* spriteOutputData = DYNAMIC_CAST(IofxOutputDataSprite*)(mWorkingData->outputData);
			const NxUserRenderSpriteBufferDesc& spriteBufferDesc = spriteOutputData->getVertexDesc();
			
			mOutputDWords = spriteBufferDesc.stride >> 2;
			spriteOutputLayout.stride = spriteBufferDesc.stride;
			::memcpy(spriteOutputLayout.offsets, spriteBufferDesc.semanticOffsets, sizeof(spriteOutputLayout.offsets));

			_storage_.update(mSpriteOutputLayoutHandle, spriteOutputLayout);
		}
	}

}


#pragma warning(push)
#pragma warning(disable:4312) // conversion from 'CUdeviceptr' to 'PxU32 *' of greater size

PxTaskID IofxManagerGPU::launchGpuTasks()
{
	physx::PxTaskManager* tm = mIofxScene.mApexScene->getTaskManager();
	tm->submitUnnamedTask(*mTaskLaunch, PxTaskType::TT_GPU);
	mTaskLaunch->finishBefore(mManager.mPostUpdateTaskID);
	return mTaskLaunch->getTaskID();
}

void IofxManagerGPU::launchPrep()
{
	IosObjectGpuData* mWorkingData = DYNAMIC_CAST(IosObjectGpuData*)(mManager.mWorkingIosData);

	if (!mWorkingData->numParticles)
	{
		mEmptySimulation = true;
		return;
	}

	mCurSeed = static_cast<PxU32>(mIofxScene.mApexScene->getSeed());
	mTargetBufDevPtr = 0;
	mOutputToBuffer = true;
	mTargetTextureCount = 0;

	physx::PxTaskManager* tm = mIofxScene.mApexScene->getTaskManager();
	physx::PxCudaContextManager* ctx = tm->getGpuDispatcher()->getCudaContextManager();
	if (mInteropStateCopy > IofxManager::INTEROP_FAILED)
	{
		bool bInteropFailed = false;
		if (mInteropStateCopy == IofxManager::INTEROP_READY)
		{
			bInteropFailed = true;
			if (mWorkingData->renderData->getBufferIsMapped())
			{
				physx::PxScopedCudaLock s(*ctx);

				CUdeviceptr renderableDevicePtr;
				if ( mWorkingData->renderData->resolveResourceList(renderableDevicePtr, mTargetTextureCount, mTargetCuArrayList) )
				{
					mTargetBufDevPtr = reinterpret_cast<PxU32 *>(renderableDevicePtr);
					PX_ASSERT( mTargetBufDevPtr != NULL || mTargetTextureCount > 0 );

					if (!mManager.mIsMesh)
					{
						IofxSharedRenderDataSprite* renderDataSprite = static_cast<IofxSharedRenderDataSprite*>(mWorkingData->renderData);

						//alloc/release TextureBuffers
						for( PxU32 i = 0; i < mTargetTextureCount; ++i ) {
							mTargetTextureBufferList[i].alloc( ctx, renderDataSprite->getTextureDesc(i), mTargetCuArrayList[i] );
						}
						for( PxU32 i = mTargetTextureCount; i < NxUserRenderSpriteBufferDesc::MAX_SPRITE_TEXTURES; ++i ) {
							mTargetTextureBufferList[i].release();
						}
					}

					//we've resolved interop mapped memory!
					bInteropFailed = false;
				}
			}
		}

		if (bInteropFailed)
		{
			APEX_INTERNAL_ERROR("IofxManager: CUDA Interop Error - failed to resolve mapped CUDA memory!");
			//this case will be handles later in swapObjectData() on fetchResult!
		}
		else
		{
			//interop is ok, disable output to buffer
			mOutputToBuffer = false;
		}
	}

	if (mOutputToBuffer)
	{
		//wait for outputData copy to render resource
		mWorkingData->waitForRenderDataUpdate();

		physx::PxScopedCudaLock s(*ctx);

		mWorkingData->allocOutputs(ctx);

		if (!mManager.mIsMesh)
		{
			IofxOutputDataSprite* spriteOutputData = DYNAMIC_CAST(IofxOutputDataSprite*)(mWorkingData->outputData);
			mTargetTextureCount = spriteOutputData->getTextureCount();

			for( PxU32 i = 0; i < mTargetTextureCount; ++i ) {
				mTargetTextureBufferList[i].alloc( ctx, spriteOutputData->getVertexDesc().textureDescs[i] );
			}
			for( PxU32 i = mTargetTextureCount; i < NxUserRenderSpriteBufferDesc::MAX_SPRITE_TEXTURES; ++i ) {
				mTargetTextureBufferList[i].release();
			}
		}

		if (mTargetTextureCount == 0)
		{
			mTargetOutputBuffer.realloc(mWorkingData->outputData->getDefaultBuffer().getCapacity(), ctx);
			mTargetBufDevPtr = static_cast<PxU32*>( mTargetOutputBuffer.getGpuPtr() );
		}

		if (mWorkingData->outputDWords == 0)
		{
			PX_ALWAYS_ASSERT();
			mEmptySimulation = true;
			return;
		}
	}
	else
	{
		mWorkingData->writeBufferCalled = true;

		mWorkingData->outputData->getDefaultBuffer().release();
		mTargetOutputBuffer.release();
	}

	const physx::PxU32 numActorIDValues = mCountActorIDs + 2;
	mCuActorStart.setSize(numActorIDValues, ApexMirroredPlace::CPU_GPU);
	mCuActorEnd.setSize(numActorIDValues, ApexMirroredPlace::CPU_GPU);
	mCuActorVisibleEnd.setSize(numActorIDValues, ApexMirroredPlace::CPU_GPU);
	mCuMinBounds.setSize(numActorIDValues, ApexMirroredPlace::CPU_GPU);
	mCuMaxBounds.setSize(numActorIDValues, ApexMirroredPlace::CPU_GPU);

	mCuSortedActorIDs.setSize(mWorkingData->maxStateID, mDefaultPlace);
	mCuSortedStateIDs.setSize(mWorkingData->maxStateID, mDefaultPlace);

	mManager.positionMass.setSize(mWorkingData->maxInputID, ApexMirroredPlace::CPU_GPU);
	mManager.velocityLife.setSize(mWorkingData->maxInputID, ApexMirroredPlace::CPU_GPU);
	mManager.actorIdentifiers.setSize(mWorkingData->maxInputID, ApexMirroredPlace::CPU_GPU);
	mManager.inStateToInput.setSize(mWorkingData->maxStateID, ApexMirroredPlace::CPU_GPU);
	mManager.outStateToInput.setSize(mWorkingData->numParticles, ApexMirroredPlace::CPU_GPU);
	if (mWorkingData->iosSupportsCollision)
	{
		mManager.collisionNormalFlags.setSize(mWorkingData->maxInputID, ApexMirroredPlace::CPU_GPU);
	}
	if (mWorkingData->iosSupportsDensity)
	{
		mManager.density.setSize(mWorkingData->maxInputID, ApexMirroredPlace::CPU_GPU);
	}
	if (mWorkingData->iosSupportsUserData)
	{
		mManager.userData.setSize(mWorkingData->maxInputID, ApexMirroredPlace::CPU_GPU);
	}

	mEmptySimulation = false;
}

#pragma warning(pop)


///
PX_INLINE PxU32 getHighestBitShift(physx::PxU32 x)
{
	PX_ASSERT(isPowerOfTwo(x));
	return highestSetBit(x);
}

void IofxManagerGPU::cudaLaunchRadixSort(CUstream stream, unsigned int numElements, unsigned int keyBits, unsigned int startBit, bool useSyncKernels)
{
	if (useSyncKernels)
	{
		//we use OLD Radix Sort on Tesla (SM < 2), because it is faster
		CUDA_OBJ(radixSortSyncKernel)(
			stream, numElements,
			mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr(),
			mCuSortTempKeys.getGpuPtr(), mCuSortTempValues.getGpuPtr(),
			mCuSortTemp.getGpuPtr(), keyBits, startBit
		);
	}
	else
	{
#if 1
		//NEW Radix Sort
		unsigned int totalThreads = (numElements + NEW_SORT_VECTOR_SIZE - 1) / NEW_SORT_VECTOR_SIZE;
		if (CUDA_OBJ(newRadixSortBlockKernel).isSingleBlock(totalThreads))
		{
			//launch just a single block for small sizes
			CUDA_OBJ(newRadixSortBlockKernel)(
				stream, APEX_CUDA_SINGLE_BLOCK_LAUNCH,
				numElements, keyBits, startBit,
				mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr()
			);
		}
		else
		{
			for (unsigned int bit = startBit; bit < startBit + keyBits; bit += RADIX_SORT_NBITS)
			{
				physx::PxU32 gridSize = 
					CUDA_OBJ(newRadixSortStepKernel)(
						stream, totalThreads,
						numElements, bit,
						mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr(),
						mCuSortTempKeys.getGpuPtr(), mCuSortTempValues.getGpuPtr(),
						mCuSortTemp.getGpuPtr(),
						1, 0
					);

				//launch just a single block
				CUDA_OBJ(newRadixSortStepKernel)(
					stream, APEX_CUDA_SINGLE_BLOCK_LAUNCH,
					numElements, bit,
					mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr(),
					mCuSortTempKeys.getGpuPtr(), mCuSortTempValues.getGpuPtr(),
					mCuSortTemp.getGpuPtr(),
					2, gridSize
				);

				CUDA_OBJ(newRadixSortStepKernel)(
					stream, totalThreads,
					numElements, bit,
					mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr(),
					mCuSortTempKeys.getGpuPtr(), mCuSortTempValues.getGpuPtr(),
					mCuSortTemp.getGpuPtr(),
					3, 0
				);

				mCuSortedActorIDs.swapGpuPtr(mCuSortTempKeys);
				mCuSortedStateIDs.swapGpuPtr(mCuSortTempValues);
			}
		}
#else
		//OLD Radix Sort
		for (unsigned int startBit = 0; startBit < keyBits; startBit += RADIX_SORT_NBITS)
		{
			int gridSize = 
				CUDA_OBJ(radixSortStep1Kernel)(
					stream, numElements,
					mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr(),
					mCuSortTempKeys.getGpuPtr(), mCuSortTempValues.getGpuPtr(),
					mCuSortTemp.getGpuPtr(), startBit
				);

			//launch just 1 block
			CUDA_OBJ(radixSortStep2Kernel)(
				stream, CUDA_OBJ(radixSortStep2Kernel).getBlockDim().x,
				mCuSortTemp.getGpuPtr(), gridSize
			);

			CUDA_OBJ(radixSortStep3Kernel)(
				stream, numElements,
				mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr(),
				mCuSortTempKeys.getGpuPtr(), mCuSortTempValues.getGpuPtr(),
				mCuSortTemp.getGpuPtr(), startBit
			);
		}
#endif
	}
}

bool IofxManagerGPU::cudaLaunch(CUstream stream, int kernelIndex)
{
	physx::PxTaskManager* tm = mIofxScene.mApexScene->getTaskManager();

	if (mEmptySimulation)
	{
		return false;
	}

	const physx::PxU32 numActorIDValues = mCountActorIDs + 2;
	//value <  mCountActorIDs     - valid particle with volume
	//value == mCountActorIDs     - homeless particle (no volume or invalid actor class)
	//value == mCountActorIDs + 1 - NOT_A_PARTICLE


	IofxSceneGPU* sceneGPU = static_cast<IofxSceneGPU*>(&mIofxScene);
	bool useSyncKernels = !sceneGPU->getGpuDispatcher()->getCudaContextManager()->supportsArchSM20();

	IosObjectGpuData* mWorkingData = DYNAMIC_CAST(IosObjectGpuData*)(mManager.mWorkingIosData);

	switch (kernelIndex)
	{
	case 0:
		if (mManager.mOnStartCallback)
		{
			(*mManager.mOnStartCallback)(stream);
		}
		mCopyQueue.reset(stream, 24);
		if (!mManager.mCudaIos && mWorkingData->maxInputID > 0)
		{
			mManager.positionMass.copyHostToDeviceQ(mCopyQueue);
			mManager.velocityLife.copyHostToDeviceQ(mCopyQueue);
			mManager.actorIdentifiers.copyHostToDeviceQ(mCopyQueue);
			mManager.inStateToInput.copyHostToDeviceQ(mCopyQueue);
			if (mWorkingData->iosSupportsCollision)
			{
				mManager.collisionNormalFlags.copyHostToDeviceQ(mCopyQueue);
			}
			if (mWorkingData->iosSupportsDensity)
			{
				mManager.density.copyHostToDeviceQ(mCopyQueue);
			}
			if (mWorkingData->iosSupportsUserData)
			{
				mManager.userData.copyHostToDeviceQ(mCopyQueue);
			}
			mCopyQueue.flushEnqueued();
		}
		break;

	case 1:
		/* Volume Migration (input space) */
		CUDA_OBJ(volumeMigrationKernel)(stream,
		                                PxMax(mWorkingData->maxInputID, numActorIDValues),
										mVolumeConstMemGroup.getStorage().mappedHandle(mVolumeParamsArrayHandle),
		                                mVolumeConstMemGroup.getStorage().mappedHandle(mActorClassIDBitmapArrayHandle),
		                                mNumberActorClasses, mNumberVolumes, numActorIDValues,
		                                mManager.actorIdentifiers.getGpuPtr(), mWorkingData->maxInputID,
		                                (const float4*)mManager.positionMass.getGpuPtr(),
		                                mCuActorStart.getGpuPtr(), mCuActorEnd.getGpuPtr(), mCuActorVisibleEnd.getGpuPtr()
		                               );
		break;

	case 2:
		{
			APEX_CUDA_TEXTURE_SCOPE_BIND(texRefRemapPositions,      mManager.positionMass)
			APEX_CUDA_TEXTURE_SCOPE_BIND(texRefRemapActorIDs,       mManager.actorIdentifiers)
			APEX_CUDA_TEXTURE_SCOPE_BIND(texRefRemapInStateToInput, mManager.inStateToInput)

			/* if mDistanceSortingEnabled, sort on camera distance first, else directly make ActorID keys */
			CUDA_OBJ(makeSortKeys)(stream, mWorkingData->maxStateID,
								   mManager.inStateToInput.getGpuPtr(), mWorkingData->maxInputID,
								   mManager.mActorTable.size(), mCountActorIDs,
								   mRemapConstMemGroup.getStorage().mappedHandle(mActorIDRemapArrayHandle),
								   (const float4*)mManager.positionMass.getGpuPtr(), mManager.mDistanceSortingEnabled,
								   mWorkingData->eyePosition, mWorkingData->eyeDirection, mWorkingData->zNear,
								   mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr());

			if (mManager.mDistanceSortingEnabled)
			{
				cudaLaunchRadixSort(stream, mWorkingData->maxStateID, 32, 0, useSyncKernels);

				/* Generate ActorID sort keys, using distance sorted stateID values */
				CUDA_OBJ(remapKernel)(stream, mWorkingData->maxStateID,
									  mManager.inStateToInput.getGpuPtr(), mWorkingData->maxInputID,
									  mManager.mActorTable.size(), mCountActorIDs,
									  mRemapConstMemGroup.getStorage().mappedHandle(mActorIDRemapArrayHandle),
									  mCuSortedStateIDs.getGpuPtr(), mCuSortedActorIDs.getGpuPtr());
			}
		}
		break;

	case 3:
		/* ActorID Sort (output state space) */
		// input: mCuSortedActorIDs == actorIDs, in distance sorted order
		// input: mCuSortedStateIDs == stateIDs, in distance sorted order

		// output: mCuSortedActorIDs == sorted ActorIDs
		// output: mCuSortedStateIDs == output-to-input state
		{
			//SortedActorIDs could contain values from 0 to mCountActorIDs + 1 (included),
			//so keybits should cover at least mCountActorIDs + 2 numbers
			PxU32 keybits = 0;
			while ((1U << keybits) < numActorIDValues)
			{
				++keybits;
			}

			cudaLaunchRadixSort(stream, mWorkingData->maxStateID, keybits, 0, useSyncKernels);
		}
		break;

	case 4:
		/* Per-IOFX actor particle range detection */
		CUDA_OBJ(actorRangeKernel)(stream, mWorkingData->maxStateID,
		                           mCuSortedActorIDs.getGpuPtr(), mCountActorIDs,
		                           mCuActorStart.getGpuPtr(), mCuActorEnd.getGpuPtr(), mCuActorVisibleEnd.getGpuPtr(),
								   mCuSortedStateIDs.getGpuPtr()
		                          );
		break;

	case 5:
		/* Modifiers (output state space) */
		{
			PX_PROFILER_PERF_SCOPE("IofxManagerGPUModifiers");
			ModifierCommonParams commonParams = mWorkingData->getCommonParams();

			APEX_CUDA_TEXTURE_SCOPE_BIND(texRefPositionMass,     mManager.positionMass)
			APEX_CUDA_TEXTURE_SCOPE_BIND(texRefVelocityLife,     mManager.velocityLife)
			APEX_CUDA_TEXTURE_SCOPE_BIND(texRefInStateToInput,   mManager.inStateToInput)
			APEX_CUDA_TEXTURE_SCOPE_BIND(texRefStateSpawnSeed,   mCuSpawnSeed)
			APEX_CUDA_TEXTURE_SCOPE_BIND(texRefStateSpawnScale,  mCuSpawnScale)

			APEX_CUDA_TEXTURE_SCOPE_BIND(texRefActorIDs,         mManager.actorIdentifiers)

			if (mWorkingData->iosSupportsCollision)
			{
				CUDA_OBJ(texRefCollisionNormalFlags).bindTo(mManager.collisionNormalFlags);
			}
			if (mWorkingData->iosSupportsDensity)
			{
				CUDA_OBJ(texRefDensity).bindTo(mManager.density);
			}
			if (mWorkingData->iosSupportsUserData)
			{
				CUDA_OBJ(texRefUserData).bindTo(mManager.userData);
			}

			PRNGInfo rand;
			rand.g_stateSpawnSeed = mCuSpawnSeed.getGpuPtr();
			rand.g_randBlock = mCuBlockPRNGs.getGpuPtr();
			rand.randGrid = mRandGridLeap;
			rand.randThread = mRandThreadLeap;
			rand.seed = mCurSeed;

			if (mManager.mIsMesh)
			{
				// 3x3 matrix => 9 float scalars => 3 slices

				APEX_CUDA_TEXTURE_SCOPE_BIND(texRefMeshPrivState0, *mManager.privState.slices[0]);
				APEX_CUDA_TEXTURE_SCOPE_BIND(texRefMeshPrivState1, *mManager.privState.slices[1]);
				APEX_CUDA_TEXTURE_SCOPE_BIND(texRefMeshPrivState2, *mManager.privState.slices[2]);

				MeshPrivateStateArgs meshPrivStateArgs;
				meshPrivStateArgs.g_state[0] = mManager.privState.a[0];
				meshPrivStateArgs.g_state[1] = mManager.privState.a[1];
				meshPrivStateArgs.g_state[2] = mManager.privState.a[2];

				CUDA_OBJ(meshModifiersKernel)(ApexKernelConfig(MAX_SMEM_BANKS * mOutputDWords, WARP_SIZE * physx::PxMax<physx::PxU32>(mOutputDWords, 4)), 
											  stream, mWorkingData->numParticles,
											  mManager.mInStateOffset, mManager.mOutStateOffset,
											  mModifierConstMemGroup.getStorage().mappedHandle(mClientParamsHandleArrayHandle),
											  commonParams,
											  mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr(),
											  mManager.outStateToInput.getGpuPtr(),
											  meshPrivStateArgs, mCuSpawnScale.getGpuPtr(),
											  rand, mTargetBufDevPtr,
											  mModifierConstMemGroup.getStorage().mappedHandle(mMeshOutputLayoutHandle)
											 );
			}
			else
			{
				// 1 float scalar => 1 slice

				APEX_CUDA_TEXTURE_SCOPE_BIND(texRefSpritePrivState0, *mManager.privState.slices[0]);

				SpritePrivateStateArgs spritePrivStateArgs;
				spritePrivStateArgs.g_state[0] = mManager.privState.a[0];

				//IofxSharedRenderDataSprite* renderDataSprite = static_cast<IofxSharedRenderDataSprite*>(mWorkingData->renderData);
				//IofxOutputDataSprite* outputDataSprite = static_cast<IofxOutputDataSprite*>(mWorkingData->outputData);

				//PxU32 targetTextureCount = mOutputToBuffer ? outputDataSprite->getTextureCount() : renderDataSprite->getTextureCount();
				//const NxUserRenderSpriteTextureDesc* targetTextureDescArray = mOutputToBuffer ? outputDataSprite->getTextureDescArray() : renderDataSprite->getTextureDescArray();

				if (mTargetTextureCount > 0)
				{
					SpriteTextureOutputLayout outputLayout;
					outputLayout.textureCount = mTargetTextureCount;
					for (PxU32 i = 0; i < outputLayout.textureCount; ++i)
					{
						outputLayout.textureData[i].layout = static_cast<PxU16>(mTargetTextureBufferList[i].getLayout());

						PxU32 width = mTargetTextureBufferList[i].getWidth();
						PxU32 pitch = mTargetTextureBufferList[i].getPitch();
						//width should be a power of 2 and a multiply of WARP_SIZE
						PX_ASSERT(isPowerOfTwo(width));
						PX_ASSERT(isPowerOfTwo(pitch));
						PX_ASSERT((width & (WARP_SIZE - 1)) == 0);
						outputLayout.textureData[i].widthShift = static_cast<PxU8>(highestSetBit(width));
						outputLayout.textureData[i].pitchShift = static_cast<PxU8>(highestSetBit(pitch));

						outputLayout.texturePtr[i] = mTargetTextureBufferList[i].getPtr();
					}

					CUDA_OBJ(spriteTextureModifiersKernel)(stream, mWorkingData->numParticles,
														   mManager.mInStateOffset, mManager.mOutStateOffset,
														   mModifierConstMemGroup.getStorage().mappedHandle(mClientParamsHandleArrayHandle),
														   commonParams,
														   mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr(),
														   mManager.outStateToInput.getGpuPtr(),
														   spritePrivStateArgs, mCuSpawnScale.getGpuPtr(),
														   rand, outputLayout
														  );

				}
				else
				{
					CUDA_OBJ(spriteModifiersKernel)(ApexKernelConfig(MAX_SMEM_BANKS * mOutputDWords, WARP_SIZE * physx::PxMax<physx::PxU32>(mOutputDWords, 4)),
													stream, mWorkingData->numParticles,
													mManager.mInStateOffset, mManager.mOutStateOffset,
													mModifierConstMemGroup.getStorage().mappedHandle(mClientParamsHandleArrayHandle),
													commonParams,
													mCuSortedActorIDs.getGpuPtr(), mCuSortedStateIDs.getGpuPtr(),
													mManager.outStateToInput.getGpuPtr(),
													spritePrivStateArgs, mCuSpawnScale.getGpuPtr(),
													rand, mTargetBufDevPtr,
													mModifierConstMemGroup.getStorage().mappedHandle(mSpriteOutputLayoutHandle)
												   );
				}
			}

			if (mWorkingData->iosSupportsCollision)
			{
				CUDA_OBJ(texRefCollisionNormalFlags).unbind();
			}
			if (mWorkingData->iosSupportsDensity)
			{
				CUDA_OBJ(texRefDensity).unbind();
			}
			if (mWorkingData->iosSupportsUserData)
			{
				CUDA_OBJ(texRefUserData).unbind();
			}
		}
		break;

	case 6:
		if (mCountActorIDs > 0)
		{
			/* Per-IOFX actor BBox generation */
			APEX_CUDA_TEXTURE_SCOPE_BIND(texRefBBoxPositions, mManager.positionMass)

			if (useSyncKernels)
			{
				CUDA_OBJ(bboxSyncKernel)(
					stream, mWorkingData->numParticles,
					mCuSortedActorIDs.getGpuPtr(),
					mManager.outStateToInput.getGpuPtr(),
					(const float4*)mManager.positionMass.getGpuPtr(),
					(float4*)mCuMinBounds.getGpuPtr(), (float4*)mCuMaxBounds.getGpuPtr(),
					mCuTempActorIDs.getGpuPtr(),
					(float4*)mCuTempMinBounds.getGpuPtr(), (float4*)mCuTempMaxBounds.getGpuPtr()
				);
			}
			else
			{
				physx::PxU32 bboxGridSize =
					CUDA_OBJ(bboxKernel)(
						stream, mWorkingData->numParticles,
						mCuSortedActorIDs.getGpuPtr(),
						mManager.outStateToInput.getGpuPtr(),
						(const float4*)mManager.positionMass.getGpuPtr(),
						(float4*)mCuMinBounds.getGpuPtr(), (float4*)mCuMaxBounds.getGpuPtr(),
						mCuTempActorIDs.getGpuPtr(),
						(float4*)mCuTempMinBounds.getGpuPtr(), (float4*)mCuTempMaxBounds.getGpuPtr(),
						1, 0
					);

				CUDA_OBJ(bboxKernel)(
					stream, APEX_CUDA_SINGLE_BLOCK_LAUNCH,
					mCuSortedActorIDs.getGpuPtr(),
					mManager.outStateToInput.getGpuPtr(),
					(const float4*)mManager.positionMass.getGpuPtr(),
					(float4*)mCuMinBounds.getGpuPtr(), (float4*)mCuMaxBounds.getGpuPtr(),
					mCuTempActorIDs.getGpuPtr(),
					(float4*)mCuTempMinBounds.getGpuPtr(), (float4*)mCuTempMaxBounds.getGpuPtr(),
					2, bboxGridSize
				);
			}
		}
		break;

	case 7:
		if (mTargetTextureCount > 0)
		{
			if (mOutputToBuffer)
			{
				IofxOutputDataSprite* spriteOutputData = DYNAMIC_CAST(IofxOutputDataSprite*)(mWorkingData->outputData);
				PX_ASSERT(spriteOutputData->getTextureCount() == mTargetTextureCount);

				for (PxU32 i = 0; i < mTargetTextureCount; ++i)
				{
					mTargetTextureBufferList[i].copyDeviceToHostQ(spriteOutputData->getTextureBuffer(i), mCopyQueue);
				}
			}
			else
			{
				for (PxU32 i = 0; i < mTargetTextureCount; ++i)
				{
					mTargetTextureBufferList[i].copyToArray(mTargetCuArrayList[i], stream, mWorkingData->numParticles);
				}
			}
		}
		else
		{
			if (mOutputToBuffer)
			{
				mTargetOutputBuffer.copyDeviceToHostQ(mWorkingData->outputData->getDefaultBuffer(), mCopyQueue);
			}
		}
		if (mCountActorIDs > 0)
		{
			mCuMinBounds.copyDeviceToHostQ(mCopyQueue);
			mCuMaxBounds.copyDeviceToHostQ(mCopyQueue);
		}
		mCuActorStart.copyDeviceToHostQ(mCopyQueue);
		mCuActorEnd.copyDeviceToHostQ(mCopyQueue);
		mCuActorVisibleEnd.copyDeviceToHostQ(mCopyQueue);


		if (mCuSortedActorIDs.cpuPtrIsValid())
		{
			mManager.inStateToInput.copyDeviceToHostQ(mCopyQueue);
			mManager.actorIdentifiers.copyDeviceToHostQ(mCopyQueue);
			mManager.outStateToInput.copyDeviceToHostQ(mCopyQueue);
			mManager.positionMass.copyDeviceToHostQ(mCopyQueue);

			mCuSortedActorIDs.copyDeviceToHostQ(mCopyQueue);
			mCuSortedStateIDs.copyDeviceToHostQ(mCopyQueue);
		}
		else if (!mManager.mCudaIos)
		{
			mManager.actorIdentifiers.copyDeviceToHostQ(mCopyQueue);
			mManager.outStateToInput.copyDeviceToHostQ(mCopyQueue);
		}

		mCopyQueue.flushEnqueued();

		if (mManager.mOnFinishCallback)
		{
			(*mManager.mOnFinishCallback)(stream);
		}

		tm->getGpuDispatcher()->addCompletionPrereq(*tm->getTaskFromID(mManager.mPostUpdateTaskID));
		return false;

	default:
		PX_ALWAYS_ASSERT();
		return false;
	}

	return true;
}

void IofxManagerGPU::fetchResults()
{
	IosObjectGpuData* mWorkingData = DYNAMIC_CAST(IosObjectGpuData*)(mManager.mWorkingIosData);
	PX_UNUSED(mWorkingData);

#ifdef APEX_TEST
	IofxManagerTestData* testData = mManager.mTestData;
	if (testData != NULL)
	{
		testData->mIsGPUTest = true;

		testData->mCountActorIDs = mCountActorIDs;
		testData->mMaxInputID = mWorkingData->maxInputID;
		testData->mMaxStateID = mWorkingData->maxStateID;
		testData->mNumParticles = mWorkingData->numParticles;

		testData->mInStateToInput.resize(mWorkingData->maxStateID);
		testData->mSortedActorIDs.resize(mWorkingData->maxStateID);
		testData->mSortedStateIDs.resize(mWorkingData->maxStateID);

		testData->mOutStateToInput.resize(mWorkingData->numParticles);
		testData->mPositionMass.resize(mWorkingData->numParticles);

		const PxU32 numActorIDValues = mCountActorIDs + 2;
		testData->mMinBounds.resize(numActorIDValues);
		testData->mMaxBounds.resize(numActorIDValues);
		testData->mActorStart.resize(numActorIDValues);
		testData->mActorEnd.resize(numActorIDValues);
		testData->mActorVisibleEnd.resize(numActorIDValues);

		for (PxU32 i = 0; i < mWorkingData->maxStateID; i++)
		{
			testData->mSortedActorIDs[i] = mCuSortedActorIDs[i];
			testData->mSortedStateIDs[i] = mCuSortedStateIDs[i];
			testData->mInStateToInput[i] = mManager.inStateToInput[i];
		}
		for (PxU32 i = 0; i < mWorkingData->numParticles; i++)
		{
			testData->mOutStateToInput[i] = mManager.outStateToInput[i];
			testData->mPositionMass[i] = mManager.positionMass[i];
		}
		for (PxU32 i = 0; i < numActorIDValues; ++i)
		{
			testData->mMinBounds[i] = mCuMinBounds[i];
			testData->mMaxBounds[i] = mCuMaxBounds[i];
			testData->mActorStart[i] = mCuActorStart[i];
			testData->mActorEnd[i] = mCuActorEnd[i];
			testData->mActorVisibleEnd[i] = mCuActorVisibleEnd[i];
		}
	}
#endif

#if 0
	{
		ApexMirroredArray<PxU32> actorID(*mIofxScene.mApexScene);
		ApexMirroredArray<PxVec4> outMinBounds(*mIofxScene.mApexScene);
		ApexMirroredArray<PxVec4> outMaxBounds(*mIofxScene.mApexScene);
		ApexMirroredArray<PxVec4> outDebugInfo(*mIofxScene.mApexScene);
		ApexMirroredArray<PxU32> tmpLastActorID(*mIofxScene.mApexScene);
		tmpLastActorID.setSize(64, ApexMirroredPlace::CPU_GPU);

		const PxU32 NE = 2000;
		actorID.setSize(NE, ApexMirroredPlace::CPU_GPU);

		Array<PxU32> actorCounts;
		actorCounts.reserve(1000);

		PxU32 NA = 0;
		for (PxU32 ie = 0; ie < NE; ++NA)
		{
			PxU32 num_ie = rand(1, 100); // We need to use QDSRand here s.t. seed could be preset during tests!
			PxU32 next_ie = PxMin(ie + num_ie, NE);

			actorCounts.pushBack(next_ie - ie);

			for (; ie < next_ie; ++ie)
			{
				actorID[ie] = NA;
			}
		}
		outMinBounds.setSize(NA, ApexMirroredPlace::CPU_GPU);
		outMaxBounds.setSize(NA, ApexMirroredPlace::CPU_GPU);
		outDebugInfo.setSize(NA, ApexMirroredPlace::CPU_GPU);

		for (PxU32 ia = 0; ia < NA; ++ia)
		{
			outMinBounds[ia].setZero();
			outMaxBounds[ia].setZero();
		}

		physx::PxTaskManager* tm = mIofxScene.mApexScene->getTaskManager();
		physx::PxCudaContextManager* ctx = tm->getGpuDispatcher()->getCudaContextManager();
		physx::PxScopedCudaLock s(*ctx);

		mCopyQueue.reset(0, 4);

		actorID.copyHostToDeviceQ(mCopyQueue);
		outMinBounds.copyHostToDeviceQ(mCopyQueue);
		outMaxBounds.copyHostToDeviceQ(mCopyQueue);
		mCopyQueue.flushEnqueued();

		CUDA_OBJ(bboxKernel2)(0, NE, actorID.getGpuPtr(), NULL, 0, (float4*)outDebugInfo.getGpuPtr(), (float4*)outMinBounds.getGpuPtr(), (float4*)outMaxBounds.getGpuPtr()/*, tmpLastActorID.getGpuPtr()*/);

		outMinBounds.copyDeviceToHostQ(mCopyQueue);
		outMaxBounds.copyDeviceToHostQ(mCopyQueue);
		outDebugInfo.copyDeviceToHostQ(mCopyQueue);
		tmpLastActorID.copyDeviceToHostQ(mCopyQueue);
		mCopyQueue.flushEnqueued();

		CUT_SAFE_CALL(cuCtxSynchronize());

		PxU32 errors = 0;
		PxF32 totCount = 0;
		for (PxU32 ie = 0; ie < NE; ++ie)
		{
			PxU32 id = actorID[ie];
			if (ie == 0 || actorID[ie - 1] != id)
			{
				PxU32 count = actorCounts[id];
				const PxVec4& bounds = outMinBounds[id];
				if (bounds.x != count)
				{
					++errors;
				}
				if (bounds.y != count * 2)
				{
					++errors;
				}
				if (bounds.z != count * 3)
				{
					++errors;
				}
				totCount += count;
			}
		}

	}
#endif

#if 0
	{
		physx::PxTaskManager* tm = mIofxScene.mApexScene->getTaskManager();
		physx::PxCudaContextManager* ctx = tm->getGpuDispatcher()->getCudaContextManager();

		physx::PxScopedCudaLock s(*ctx);

		CUT_SAFE_CALL(cuCtxSynchronize());
	}
#endif
#if DEBUG_GPU
	{
		physx::Array<int> valuesCounters(mWorkingData->maxStateID, 0);
		PxU32 lastKey = PxU32(-1);
		for (PxU32 i = 0; i < mWorkingData->maxStateID; ++i)
		{
			PxU32 currKey = mCuSortedActorIDs.get(i);
			PX_ASSERT(currKey < mCountActorIDs + 2);
			if (lastKey != PxU32(-1))
			{
				PX_ASSERT(lastKey <= currKey);
			}
			if (lastKey != currKey)
			{
				if (mCuActorStart[currKey] != i)
				{
					int temp = 0;
					temp++;
				}
				PX_ASSERT(mCuActorStart[currKey] == i);
				if (lastKey != PxU32(-1))
				{
					if (mCuActorEnd[lastKey] != i)
					{
						int temp = 0;
						temp++;
					}
					PX_ASSERT(mCuActorEnd[lastKey] == i);
				}
			}
			lastKey = currKey;

			PxU32 currValue = (mCuSortedStateIDs.get(i) & STATE_ID_MASK);
			PX_ASSERT(currValue < mWorkingData->maxStateID);
			if (currValue < mWorkingData->maxStateID)
			{
				valuesCounters[currValue] += 1;
			}
		}
		if (lastKey != PxU32(-1))
		{
			PX_ASSERT(mCuActorEnd[lastKey] == mWorkingData->maxStateID);
		}
		for (PxU32 i = 0; i < mWorkingData->maxStateID; ++i)
		{
			PX_ASSERT(valuesCounters[i] == 1);
		}
	}
#endif

	/* Swap input/output state offsets */
	mManager.swapStates();

	if (mEmptySimulation)
	{
		for (PxU32 i = 0 ; i < mNumberVolumes ; i++)
		{
			IofxManager::VolumeData& d = mManager.mVolumeTable[ i ];
			if (d.vol == 0)
			{
				continue;
			}

			for (PxU32 j = 0 ; j < mManager.mActorTable.size() ; j++)
			{
				IofxActor* iofx = d.mActors[ j ];
				if (iofx && iofx != DEFERRED_IOFX_ACTOR)
				{
					iofx->mResultBounds.setEmpty();
					iofx->mResultRange.startIndex = 0;
					iofx->mResultRange.objectCount = 0;
					iofx->mResultVisibleCount = 0;
				}
			}
		}
	}
	else
	{
		PX_ASSERT(mCuActorStart.cpuPtrIsValid() && mCuActorEnd.cpuPtrIsValid());
		if (!mCuActorStart.cpuPtrIsValid() || !mCuActorEnd.cpuPtrIsValid())
		{
			// Workaround for issue seen by a customer
			APEX_INTERNAL_ERROR("Bad cpuPtr in IofxManagerGPU::fetchResults");
			return;
		}
#ifndef NDEBUG
		//check Actor Ranges
		{
			PxU32 totalCount = 0;
			//range with the last index (= mCountActorIDs) contains homeless particles!
			for (PxU32 i = 0 ; i <= mCountActorIDs ; i++)
			{
				const PxU32 rangeStart = mCuActorStart[ i ];
				const PxU32 rangeEnd = mCuActorEnd[ i ];
				const PxU32 rangeVisibleEnd = mCuActorVisibleEnd[ i ];

				PX_ASSERT(rangeStart < mWorkingData->numParticles);
				PX_ASSERT(rangeEnd <= mWorkingData->numParticles);
				PX_ASSERT(rangeStart <= rangeEnd);
				PX_ASSERT(rangeStart <= rangeVisibleEnd && rangeVisibleEnd <= rangeEnd);

				const PxU32 rangeCount = rangeEnd - rangeStart;
				totalCount += rangeCount;
			}
			PX_ASSERT(totalCount == mWorkingData->numParticles);
		}
#endif

		PxU32 aid = 0;
		for (PxU32 i = 0 ; i < mNumberVolumes ; i++)
		{
			IofxManager::VolumeData& d = mManager.mVolumeTable[ i ];
			if (d.vol == 0)
			{
				aid += mManager.mActorTable.size();
				continue;
			}

			for (PxU32 j = 0 ; j < mManager.mActorTable.size() ; j++)
			{
				const PxU32 rangeStart = mCuActorStart[ aid ];
				const PxU32 rangeEnd = mCuActorEnd[ aid ];
				const PxU32 rangeVisibleEnd = mCuActorVisibleEnd[ aid ];

				const PxU32 rangeCount = rangeEnd - rangeStart;
				const PxU32 visibleCount = rangeVisibleEnd - rangeStart;

				if (d.mActors[ j ] == DEFERRED_IOFX_ACTOR && mManager.mActorTable[ j ] != NULL &&
				        (mIofxScene.mModule->mDeferredDisabled || rangeCount))
				{
					IofxActor* iofxActor = PX_NEW(IofxActorGPU)(mManager.mActorTable[j]->getRenderAsset(), &mIofxScene, mManager);
					if (d.vol->addIofxActor(*iofxActor))
					{
						d.mActors[ j ] = iofxActor;

						mManager.initIofxActor(iofxActor, j, d.vol);

						// lock this renderable because the APEX scene will unlock it after this method is called
						iofxActor->renderDataLock();
					}
					else
					{
						iofxActor->release();
					}
				}

				IofxActor* iofxActor = d.mActors[ j ];
				if (iofxActor && iofxActor != DEFERRED_IOFX_ACTOR)
				{
					iofxActor->mResultBounds.setEmpty();
					if (rangeCount > 0)
					{
						iofxActor->mResultBounds.minimum = mCuMinBounds[ aid ].getXYZ();
						iofxActor->mResultBounds.maximum = mCuMaxBounds[ aid ].getXYZ();
					}
					PX_ASSERT(iofxActor->mRenderBounds.isFinite());
					iofxActor->mResultRange.startIndex = rangeStart;
					iofxActor->mResultRange.objectCount = rangeCount;
					iofxActor->mResultVisibleCount = visibleCount;
				}

				aid++;
			}
		}
	}

}

bool IofxManagerGPU::swapObjectData()
{
	bool result = true;
	if (mManager.mInteropState != IofxManager::INTEROP_OFF)
	{
		switch (mManager.mInteropState)
		{
		case IofxManager::INTEROP_WAIT_FOR_RENDER_ALLOC:
			//just do nothing
			result = false;
			break;
		case IofxManager::INTEROP_WAIT_FOR_FETCH_RESULT:
			physx::swap(mManager.mStagingIosData, mManager.mWorkingIosData);
			mManager.mInteropState = IofxManager::INTEROP_READY;
			result = false;
			break;
		case IofxManager::INTEROP_READY:
			if (!mOutputToBuffer)
			{
				result = true;
				break;
			}
			//this is the case when we had an interop failure in launchPrep()
			//go to the next case here!
		case IofxManager::INTEROP_FAILED:
			for (PxU32 i = 0 ; i < mManager.mObjData.size() ; i++)
			{
				mManager.mObjData[i]->renderData->release();
				mManager.mObjData[i]->renderData = mManager.mSharedRenderData;
			}
			mManager.mInteropState = IofxManager::INTEROP_OFF;
			result = mOutputToBuffer;
			break;
		default:
			PX_ALWAYS_ASSERT();
		};
		mInteropStateCopy = mManager.mInteropState;
	}
	return result;
}


/**
 * Called from render thread context, just before renderer calls update/dispatch on any IOFX
 * actors.  Map/Unmap render resources as required.  "Mapped" means the graphics buffer has been
 * mapped into our CUDA context where our kernels can write directly into it.
 */
void IofxManager::fillMapUnmapArraysForInterop(physx::Array<CUgraphicsResource> &toMapArray, physx::Array<CUgraphicsResource> &toUnmapArray)
{
	if (mInteropState > INTEROP_FAILED)
	{
		physx::PxGpuDispatcher *gd = mIofxScene->mApexScene->getTaskManager()->getGpuDispatcher();

		const PxU32 targetSemantics = mTargetSemantics;

		if (mInteropState == INTEROP_WAIT_FOR_RENDER_ALLOC && targetSemantics != 0)
		{
			mResultIosData->updateSemantics(targetSemantics, true);
			mResultIosData->renderData->alloc( mResultIosData, gd->getCudaContextManager() );
			bool bFailed = !mResultIosData->renderData->addResourcesToArray(toMapArray);
			if (!bFailed)
			{
				mStagingIosData->updateSemantics(targetSemantics, true);
				mStagingIosData->renderData->alloc( mStagingIosData, gd->getCudaContextManager() );
				bFailed |= !mStagingIosData->renderData->addResourcesToArray(toMapArray);
			}
			if (bFailed)
			{
				mInteropState = INTEROP_FAILED;
			}
		}
		else if (mResultReadyState == RESULT_READY)
		{
			PX_ASSERT(mInteropState == INTEROP_READY);

			PX_ASSERT(mResultIosData->renderData->getBufferIsMapped() == true);
			bool bFailed = !mResultIosData->renderData->addResourcesToArray(toUnmapArray);
			if (!bFailed)
			{
				PX_ASSERT(mStagingIosData->renderData->getBufferIsMapped() == false);
				mStagingIosData->updateSemantics(targetSemantics, true);
				mStagingIosData->renderData->alloc( mStagingIosData, gd->getCudaContextManager() );
				bFailed |= !mStagingIosData->renderData->addResourcesToArray(toMapArray);
			}
			if (bFailed)
			{
				mInteropState = INTEROP_FAILED;
			}
		}
	}
}


void IofxManager::mapBufferResults(bool mapSuccess, bool unmapSuccess)
{
	if (mInteropState > INTEROP_FAILED)
	{
		if (mInteropState == INTEROP_WAIT_FOR_RENDER_ALLOC && mResultIosData->outputSemantics != 0)
		{
			if (mapSuccess)
			{
				mResultIosData->renderData->setBufferIsMapped(true);
				mStagingIosData->renderData->setBufferIsMapped(true);

				mInteropState = INTEROP_WAIT_FOR_FETCH_RESULT;
			}
			else
			{
				mInteropState = INTEROP_FAILED;
			}
		}
		else if (mResultReadyState == RESULT_READY)
		{
			PX_ASSERT(mInteropState == INTEROP_READY);

			if (unmapSuccess)
			{
				mResultIosData->renderData->setBufferIsMapped(false);
			}
			if (mapSuccess)
			{
				mStagingIosData->renderData->setBufferIsMapped(true);
			}
			if (!(mapSuccess && unmapSuccess))
			{
				mInteropState = INTEROP_FAILED;
			}
		}
	}
}

}
}
} // namespace physx::apex

#endif
