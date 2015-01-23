/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FORCEFIELD_SCENE_H__
#define __FORCEFIELD_SCENE_H__

#include "NxApex.h"

#include "ModuleForceField.h"

#include "ApexInterface.h"
#include "ApexContext.h"
#include "ApexSDKHelpers.h"

#include "NiApexRenderDebug.h"
#include "NiApexSDK.h"
#include "NiModule.h"

#include "DebugRenderParams.h"
#include "ForceFieldDebugRenderParams.h"

#include "PxTask.h"

#include "NiFieldSamplerScene.h"

#if defined(APEX_CUDA_SUPPORT)
#include "ApexCudaWrapper.h"
#include "ApexCuda.h"
#include "CudaModuleScene.h"

#include "../cuda/include/common.h"

#define SCENE_CUDA_OBJ(scene, name) static_cast<ForceFieldSceneGPU*>(scene)->APEX_CUDA_OBJ_NAME(name)
#define CUDA_OBJ(name) SCENE_CUDA_OBJ(mForceFieldScene, name)
#endif

namespace physx
{
namespace apex
{
class NiApexScene;
class DebugRenderParams;
class NiFieldSamplerManager;

namespace forcefield
{
class ModuleForceField;
class ForceFieldActor;
class NxForceFieldActorDesc;


class ForceFieldScene : public NiFieldSamplerScene, public ApexContext, public NxApexResource, public ApexResource
{
public:
	ForceFieldScene(ModuleForceField& module, NiApexScene& scene, NiApexRenderDebug* renderDebug, NxResourceList& list);
	~ForceFieldScene();

	/* NiModuleScene */
	void						updateActors(physx::PxF32 deltaTime);
	void						submitTasks(PxF32 elapsedTime, PxF32 substepSize, PxU32 numSubSteps);
	void						setTaskDependencies();

	virtual void				visualize(void);
	virtual void				visualizeForceFieldForceFields(void);
	virtual void				visualizeForceFieldForces(void);
	virtual void				fetchResults(void);

	virtual void				setModulePhysXScene(PxScene* s);
	virtual PxScene*			getModulePhysXScene() const
	{
		return mPhysXScene;
	}

	virtual physx::PxF32		setResource(physx::PxF32, physx::PxF32, physx::PxF32)
	{
		return 0.0f;
	}
	virtual physx::PxF32		getBenefit(void)
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
	PxU32						getListIndex(void) const
	{
		return m_listIndex;
	}
	void						setListIndex(NxResourceList& list, PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}
	virtual void				release(void)
	{
		mModule->releaseNiModuleScene(*this);
	}

	virtual ForceFieldActor*	createForceFieldActor(const NxForceFieldActorDesc& desc, ForceFieldAsset& asset, NxResourceList& list) = 0;

	NiApexScene& getApexScene() const
	{
		return *mApexScene;
	}

	NiFieldSamplerManager*	getNiFieldSamplerManager();

	/* NiFieldSamplerScene */
	virtual void getFieldSamplerSceneDesc(NiFieldSamplerSceneDesc& ) const
	{
	}

protected:
	void						destroy();

	ModuleForceField* 			mModule;
	NiApexScene*                mApexScene;
	PxScene*                    mPhysXScene;

	NiApexRenderDebug* 			mRenderDebug;

	DebugRenderParams*				mDebugRenderParams;
	ForceFieldDebugRenderParams*	mForceFieldDebugRenderParams;

	NiFieldSamplerManager*		mFieldSamplerManager;

private:
	class TaskUpdate : public physx::PxTask
	{
	public:
		TaskUpdate(ForceFieldScene& owner) : mOwner(owner) {}
		const char* getName() const
		{
			return "ForceFieldScene::Update";
		}
		void run();
	protected:
		ForceFieldScene& mOwner;
	private:
		TaskUpdate& operator=(const  TaskUpdate&);
	};

	TaskUpdate					mUpdateTask;

	friend class ModuleForceField;
	friend class ForceFieldActor;
	friend class TaskUpdate;
};

class ForceFieldSceneCPU : public ForceFieldScene
{
public:
	ForceFieldSceneCPU(ModuleForceField& module, NiApexScene& scene, NiApexRenderDebug* renderDebug, NxResourceList& list);
	~ForceFieldSceneCPU();

	ForceFieldActor*	createForceFieldActor(const NxForceFieldActorDesc& desc, ForceFieldAsset& asset, NxResourceList& list);

	/* NiFieldSamplerScene */

protected:
};

#if defined(APEX_CUDA_SUPPORT)
class ForceFieldSceneGPU : public ForceFieldScene, public CudaModuleScene
{
public:
	ForceFieldSceneGPU(ModuleForceField& module, NiApexScene& scene, NiApexRenderDebug* renderDebug, NxResourceList& list);
	~ForceFieldSceneGPU();

	ForceFieldActor*	createForceFieldActor(const NxForceFieldActorDesc& desc, ForceFieldAsset& asset, NxResourceList& list);

//CUDA module objects
#include "../cuda/include/ForceField.h"

	/* NiFieldSamplerScene */
	virtual ApexCudaConstStorage*	getFieldSamplerCudaConstStorage();
	virtual bool					launchFieldSamplerCudaKernel(const fieldsampler::NiFieldSamplerKernelLaunchData&);

protected:
	/* keep a convenience pointer to the cuda context manager */
	PxCudaContextManager* mCtxMgr;
};
#endif

}
}
} // end namespace physx::apex

#endif
