/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __IOFX_MANAGER_H__
#define __IOFX_MANAGER_H__

#include "PsArray.h"
#include "PsHashMap.h"
#include "ApexInterface.h"
#include "NiApexScene.h"
#include "NiIofxManager.h"
#include "ApexActor.h"

#include "ModifierData.h"

#include "PxTask.h"
#include "ApexMirroredArray.h"

namespace physx
{
namespace apex
{
namespace iofx
{

class IofxScene;
class IofxManager;
class IosObjectBaseData;
class IofxAsset;
class IofxActor;
class ApexRenderVolume;
class IofxSharedRenderData;

class TaskUpdateEffects : public physx::PxTask
{
public:
	TaskUpdateEffects(IofxManager& owner) : mOwner(owner) {}
	const char* getName() const
	{
		return "IofxManager::UpdateEffects";
	}
	void run();
protected:
	IofxManager& mOwner;

private:
	TaskUpdateEffects& operator=(const TaskUpdateEffects&);
};

class IofxAsset;

class IofxSceneInst : public physx::UserAllocated
{
public:
	virtual ~IofxSceneInst() {}

	PxU32 getRefCount() const
	{
		return _refCount;
	}
	void addRef()
	{
		++_refCount;
	}
	bool removeRef()
	{
		PX_ASSERT(_refCount > 0);
		return (--_refCount == 0);
	}

protected:
	IofxSceneInst() : _refCount(0) {}

private:
	PxU32		_refCount;
};

class IofxAssetSceneInst : public IofxSceneInst
{
public:
	IofxAssetSceneInst(IofxAsset* asset, PxU32 semantics)
		: _asset(asset), _semantics(semantics)
	{
	}
	virtual ~IofxAssetSceneInst()
	{
	}

	PX_INLINE IofxAsset* getAsset() const
	{
		return _asset;
	}
	PX_INLINE PxU32 getSemantics() const
	{
		return _semantics;
	}

protected:
	IofxAsset*	_asset;
	PxU32		_semantics;
};

class IofxManagerClient : public NiIofxManagerClient, public physx::UserAllocated
{
public:
	IofxManagerClient(IofxAssetSceneInst* assetSceneInst, PxU32 actorClassID, const NiIofxManagerClient::Params& params)
		: _assetSceneInst(assetSceneInst), _actorClassID(actorClassID), _params(params)
	{
	}
	virtual ~IofxManagerClient()
	{
	}

	PX_INLINE IofxAssetSceneInst* getAssetSceneInst() const
	{
		PX_ASSERT(_assetSceneInst != NULL);
		return _assetSceneInst;
	}

	PX_INLINE PxU32 getActorClassID() const
	{
		return _actorClassID;
	}

	PX_INLINE const NiIofxManagerClient::Params& getParams() const
	{
		return _params;
	}

	// NiIofxManagerClient interface
	virtual void getParams(NiIofxManagerClient::Params& params) const
	{
		params = _params;
	}
	virtual void setParams(const NiIofxManagerClient::Params& params)
	{
		_params = params;
	}

protected:
	IofxAssetSceneInst*			_assetSceneInst;
	PxU32						_actorClassID;
	NiIofxManagerClient::Params	_params;
};

class IofxActorSceneInst : public IofxSceneInst
{
public:
	IofxActorSceneInst(NxApexAsset* renderAsset)
		: _renderAsset(renderAsset)
	{
	}
	virtual ~IofxActorSceneInst()
	{
	}

	PX_INLINE NxApexAsset* getRenderAsset() const
	{
		return _renderAsset;
	}

	void addAssetSceneInst(IofxAssetSceneInst* value)
	{
		_assetSceneInstArray.pushBack(value);
	}
	bool removeAssetSceneInst(IofxAssetSceneInst* value)
	{
		return _assetSceneInstArray.findAndReplaceWithLast(value);
	}
	const physx::Array<IofxAssetSceneInst*>& getAssetSceneInstArray() const
	{
		return _assetSceneInstArray;
	}

protected:
	NxApexAsset* _renderAsset;
	physx::Array<IofxAssetSceneInst*> _assetSceneInstArray;
};

class CudaPipeline
{
public:
	virtual ~CudaPipeline() {}
	virtual void release() = 0;
	virtual void fetchResults() = 0;
	virtual void submitTasks() = 0;

	virtual PxTaskID launchGpuTasks() = 0;
	virtual void launchPrep() = 0;

	virtual IofxManagerClient* createClient(IofxAssetSceneInst* assetSceneInst, PxU32 actorClassID, const NiIofxManagerClient::Params& params) = 0;
	virtual IofxAssetSceneInst* createAssetSceneInst(IofxAsset* asset, PxU32 semantics) = 0;

	virtual bool swapObjectData() = 0;
};


class IofxManager : public NiIofxManager, public NxApexResource, public ApexResource, public ApexContext
{
public:
	IofxManager(IofxScene& scene, const NiIofxManagerDesc& desc, bool isMesh);
	~IofxManager();

	void destroy();

	/* Over-ride this ApexContext method to capture IofxActor deletion events */
	void removeActorAtIndex(physx::PxU32 index);

	void createSimulationBuffers(NiIosBufferDesc& outDesc);
	void setSimulationParameters(PxF32 radius, const PxVec3& up, PxF32 gravity, PxF32 restDensity);
	void updateEffectsData(PxF32 deltaTime, PxU32 numObjects, PxU32 maxInputID, PxU32 maxStateID, void* extraData);
	virtual void submitTasks();
	virtual void fetchResults();
	void release();
	void outputHostToDevice(physx::PxGpuCopyDescQueue& copyQueue);
	PxTaskID getUpdateEffectsTaskID(PxTaskID);
	void cpuModifiers();
	PxBounds3 getBounds() const;
	void swapStates();

	PxU32 getActorID(IofxAssetSceneInst* assetSceneInst, PxU16 meshID);
	void releaseActorID(IofxAssetSceneInst* assetSceneInst, PxU32 actorID);

	PxU16 getActorClassID(NiIofxManagerClient* client, PxU16 meshID);

	NiIofxManagerClient* createClient(physx::apex::NxIofxAsset* asset, const NiIofxManagerClient::Params& params);
	void releaseClient(NiIofxManagerClient* client);

	PxU16	getVolumeID(NxApexRenderVolume* vol);
	PX_INLINE PxU32	getSimulatedParticlesCount() const
	{
		return mLastNumObjects;
	}

	PX_INLINE void setOnStartCallback(NiIofxManagerCallback* callback)
	{
		if (mOnStartCallback) 
		{
			PX_DELETE(mOnStartCallback);
		}
		mOnStartCallback = callback;
	}
	PX_INLINE void setOnFinishCallback(NiIofxManagerCallback* callback)
	{
		if (mOnFinishCallback) 
		{
			PX_DELETE(mOnFinishCallback);
		}
		mOnFinishCallback = callback;
	}

	PX_INLINE bool isMesh()
	{
		return mIsMesh;
	}

	physx::PxU32	    getListIndex() const
	{
		return m_listIndex;
	}
	void	            setListIndex(NxResourceList& list, physx::PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}

	PxF32	getObjectRadius() const;

	IofxAssetSceneInst* 	createAssetSceneInst(NxIofxAsset* asset);

	void initIofxActor(IofxActor* iofxActor, PxU32 mActorID, ApexRenderVolume* renderVolume);


	typedef HashMap<IofxAsset*, IofxAssetSceneInst*> AssetHashMap_t;
	AssetHashMap_t				mAssetHashMap;

	physx::Array<IofxActorSceneInst*> mActorTable;

	struct ActorClassData
	{
		IofxManagerClient*	client;  // NULL for empty rows
		PxU16				meshid;
		PxU16				count;
		PxU32				actorID;
	};
	physx::Array<ActorClassData> mActorClassTable;

	struct VolumeData
	{
		ApexRenderVolume* 		 vol;			// NULL for empty rows
		PxBounds3				 mBounds;
		PxU32                    mPri;
		PxU32                    mFlags;
		physx::Array<IofxActor*> mActors; // Indexed by actorClassID
	};
	physx::Array<VolumeData>			mVolumeTable;
	physx::Array<PxU32>					mCountPerActor;
	physx::Array<PxU32>					mStartPerActor;
	physx::Array<PxU32>					mBuildPerActor;
	physx::Array<PxU32>					mOutputToState;
	physx::PxTaskID						mPostUpdateTaskID;

	physx::Array<PxU32>					mSortingKeys;

	physx::Array<PxU32>					mVolumeActorClassBitmap;

	IofxScene*                          mIofxScene;
	physx::Array<IosObjectBaseData*>	mObjData;
	NiIosBufferDesc						mSimBuffers;
	ApexSimpleString					mIosAssetName;

	// reference pointers for IOFX actors, so they know which buffer
	// in in which mode.
	IosObjectBaseData* 					mWorkingIosData;
	IosObjectBaseData* 					mResultIosData;
	IosObjectBaseData*					mStagingIosData;

	IosObjectBaseData*					mRenderIosData;

	enum InteropState
	{
		INTEROP_OFF = 0,
		INTEROP_FAILED,
		INTEROP_WAIT_FOR_RENDER_ALLOC,
		INTEROP_WAIT_FOR_FETCH_RESULT,
		INTEROP_READY,
	};
	InteropState						mInteropState;

	enum ResultReadyState
	{
		RESULT_WAIT_FOR_NEW = 0,
		RESULT_READY,
	};
	ResultReadyState					mResultReadyState;

	volatile physx::PxU32				mTargetSemantics;

	IofxSharedRenderData*				mSharedRenderData;
	physx::Array<IofxSharedRenderData*> mInteropRenderData;

	// Simulation storage, for CPU/GPU IOS
	ApexMirroredArray<PxVec4>			positionMass;
	ApexMirroredArray<PxVec4>			velocityLife;
	ApexMirroredArray<PxVec4>			collisionNormalFlags;
	ApexMirroredArray<PxF32>			density;
	ApexMirroredArray<NiIofxActorID>	actorIdentifiers;
	ApexMirroredArray<PxU32>			inStateToInput;
	ApexMirroredArray<PxU32>			outStateToInput;

	ApexMirroredArray<PxU32>			userData;

	NiIofxManagerCallback*				mOnStartCallback;
	NiIofxManagerCallback*				mOnFinishCallback;

	// Assets that were added on this frame (prior to simulate)
	physx::Array<const IofxAsset*>		addedAssets;

	// Max size of public/private states over active (simulated) assets
	PxU32								pubStateSize, privStateSize;

	// State data (CPU only)

	typedef ApexMirroredArray<IofxSlice> SliceArray;

	struct State
	{
		physx::Array<SliceArray*> slices; // Slices
		physx::Array<IofxSlice*> a, b; // Pointers to slices' halves
	};

	State								pubState;
	State								privState;

	PxU32								mInStateOffset;
	PxU32								mOutStateOffset;
	bool								mStateSwap;

	PxF32                               mTotalElapsedTime;
	bool                                mIsMesh;
	bool                                mDistanceSortingEnabled;
	bool                                mCudaIos;
	bool                                mCudaModifiers;

	void								prepareRenderResources();
	void								postPrepareRenderResources();

	void								fillMapUnmapArraysForInterop(physx::Array<CUgraphicsResource> &, physx::Array<CUgraphicsResource> &);
	void								mapBufferResults(bool, bool);

	CudaPipeline*                       mCudaPipeline;
	TaskUpdateEffects					mSimulateTask;

	PxBounds3							mBounds;

	physx::PxGpuCopyDescQueue     mCopyQueue;

	PxU32								mLastNumObjects;
	PxU32								mLastMaxInputID;
	
#ifdef APEX_TEST
	IofxManagerTestData*				mTestData;

	virtual IofxManagerTestData*		createTestData();
	virtual void						copyTestData() const;
	virtual void						clearTestData();
#endif
};

#define DEFERRED_IOFX_ACTOR ((IofxActor*)(1))

}
}
} // end namespace physx::apex

#endif // __NI_IOFX_MANAGER_H__
