/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FLUIDIOS_SCENE_H__
#define __FLUIDIOS_SCENE_H__

#include "NxApex.h"
#include "NxModuleFluidIos.h"
#include "NiApexSDK.h"
#include "NiModule.h"
#include "ModuleFluidIos.h"
#include "ApexSharedUtils.h"
#include "ApexSDKHelpers.h"
#include "ApexContext.h"
#include "ApexActor.h"
#include "ModulePerfScope.h"

#include "DebugRenderParams.h"
#include "FluidIosDebugRenderParams.h"

namespace physx
{
namespace apex
{

class NiApexRenderDebug;

namespace nxfluidios
{

class FluidIosScene : public NiModuleScene, public ApexContext, public NxApexResource, public ApexResource
{
public:
	FluidIosScene(ModuleFluidIos& module, NiApexScene& scene, NiApexRenderDebug* renderDebug, NxResourceList& list);
	~FluidIosScene();

	/* NiModuleScene */
	void									submitTasks(PxF32 elapsedTime, PxF32 substepSize, PxU32 numSubSteps);
	void									setTaskDependencies();

	void                                    fetchResults();
	void                                    setModulePhysXScene(NxScene*);
	NxScene*                                getModulePhysXScene() const
	{
		return mPhysXScene;
	}
	void									release()
	{
		mModule->releaseNiModuleScene(*this);
	}
	void									visualize();
	virtual NxModule*						getNxModule()
	{
		return mModule;
	}
	const NxCompartment* 					getCompartment() const;
	const NxCompartment* 					getSPHCompartment() const;

	virtual NxApexSceneStats* getStats()
	{
		return 0;
	}

	/* NxApexResource */
	physx::PxU32							getListIndex() const
	{
		return m_listIndex;
	}
	void                                    setListIndex(NxResourceList& list, physx::PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}

	physx::PxF32							getBenefit();
	physx::PxF32							setResource(physx::PxF32 , physx::PxF32, physx::PxF32);


	NiApexScene*							getApexScene() const
	{
		return mApexScene;
	}

protected:
	ModuleFluidIos*                        mModule;
	NxScene*                                mPhysXScene;
	NiApexScene*                            mApexScene;

	void									destroy();

	physx::PxF32							computeAABBDistanceSquared(const physx::PxBounds3& aabb);

	void									setCompartment(NxCompartment& comp);
	void									setSPHCompartment(NxCompartment& comp);

private:
	NxCompartment* 							mCompartment;
	NxCompartment* 							mSPHCompartment;
	physx::PxF32							mSumBenefit;
	NiApexRenderDebug* 						mRenderDebug;

	DebugRenderParams*						mDebugRenderParams;
	FluidIosDebugRenderParams*				mFluidIosDebugRenderParams;

	friend class FluidIosActor;
	friend class FluidIosAsset;
	friend class ModuleFluidIos;
};

}
}
} // namespace physx::apex

#endif // __FLUIDIOS_SCENE_H__
