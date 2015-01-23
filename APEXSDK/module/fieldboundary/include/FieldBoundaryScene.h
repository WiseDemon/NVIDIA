/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FIELDBOUNDARY_SCENE_H__
#define __FIELDBOUNDARY_SCENE_H__

#include "NxApex.h"

#include "ModuleFieldBoundary.h"
#include "NiApexSDK.h"
#include "NiModule.h"
#include "ApexInterface.h"
#include "ApexContext.h"
#include "ApexSDKHelpers.h"

#include "DebugRenderParams.h"
#include "FieldBoundaryDebugRenderParams.h"
#include "PxTask.h"

namespace physx
{
namespace apex
{

class NiApexScene;
class NiFieldSamplerManager;

namespace fieldboundary
{

class ModuleFieldBoundary;

class FieldBoundaryScene : public NiModuleScene, public ApexContext, public NxApexResource, public ApexResource
{
public:
	FieldBoundaryScene(ModuleFieldBoundary& module, NiApexScene& scene, NiApexRenderDebug* renderDebug, NxResourceList& list);
	~FieldBoundaryScene();

	/* NiModuleScene */
	void				updateActors(physx::PxF32 deltaTime);
	void				submitTasks(PxF32 elapsedTime, PxF32 substepSize, PxU32 numSubSteps);

	void				visualize();
	void				fetchResults();
	void				setModulePhysXScene(NxScene* s);
	NxScene*			getModulePhysXScene() const
	{
		return mPhysXScene;
	}
	physx::PxF32		setResource(physx::PxF32, physx::PxF32, physx::PxF32)
	{
		return 0.0f;
	}
	physx::PxF32		getBenefit()
	{
		return 0.0f;
	}
	NxModule*			getNxModule()
	{
		return mModule;
	}

	virtual NxApexSceneStats* getStats()
	{
		return 0;
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
	void				release()
	{
		mModule->releaseNiModuleScene(*this);
	}

	NiFieldSamplerManager* 				getNiFieldSamplerManager();

protected:
	void					destroy();

	ModuleFieldBoundary*	mModule;
	NiApexScene*			mApexScene;
	NiApexRenderDebug*		mRenderDebug;
	NxScene*				mPhysXScene;

	DebugRenderParams*					mDebugRenderParams;
	FieldBoundaryDebugRenderParams*		mFieldBoundaryDebugRenderParams;

	NiFieldSamplerManager* 				mFieldSamplerManager;

private:
	class TaskUpdate : public physx::PxTask
	{
	public:
		TaskUpdate(FieldBoundaryScene& owner) : mOwner(owner) {}
		const char* getName() const
		{
			return "FieldBoundaryScene::Update";
		}
		void run();

	protected:
		FieldBoundaryScene& mOwner;

	private:
		TaskUpdate& operator=(const TaskUpdate&);
	};
	TaskUpdate				mUpdateTask;

	friend class ModuleFieldBoundary;
	friend class FieldBoundaryActor;
	friend class TaskUpdate;
};

}
}
}

#endif
