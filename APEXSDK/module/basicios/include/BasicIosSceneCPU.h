/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __BASIC_IOS_SCENE_CPU_H__
#define __BASIC_IOS_SCENE_CPU_H__

#if ENABLE_TEST
#include "BasicIosTestScene.h"
#endif
#include "BasicIosScene.h"

namespace physx
{
namespace apex
{
namespace basicios
{

#if ENABLE_TEST
#define BASIC_IOS_SCENE BasicIosTestScene
#else
#define BASIC_IOS_SCENE BasicIosScene
#endif

class BasicIosSceneCPU : public BASIC_IOS_SCENE
{
	class TimerCallback : public NiFieldSamplerCallback, public NiIofxManagerCallback, public physx::UserAllocated
	{
		physx::shdfnd::Time mTimer;
		PxReal mMinTime, mMaxTime;
	public:
		TimerCallback() {}		
		void operator()(void* stream = NULL);
		void reset();
		PxReal getElapsedTime() const;
	};
public:
	BasicIosSceneCPU(ModuleBasicIos& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list);
	~BasicIosSceneCPU();

	virtual BasicIosActor*		createIosActor(NxResourceList& list, BasicIosAsset& asset, physx::apex::NxIofxAsset& iofxAsset);

	virtual void				fetchInjectorParams(PxU32 injectorID, InjectorParams& injParams)
	{
		PX_ASSERT(injectorID < mInjectorParamsArray.size());
		injParams = mInjectorParamsArray[ injectorID ];
	}
	virtual void				updateInjectorParams(PxU32 injectorID, const InjectorParams& injParams)
	{
		PX_ASSERT(injectorID < mInjectorParamsArray.size());
		mInjectorParamsArray[ injectorID ] = injParams;
	}

	void							fetchResults();

protected:
	virtual void setCallbacks(BasicIosActorCPU* actor);
	virtual bool growInjectorStorage(physx::PxU32 newSize)
	{
		mInjectorParamsArray.resize(newSize);
		return true;
	}

private:
	physx::Array<InjectorParams> mInjectorParamsArray;
	TimerCallback					mTimerCallback;

	friend class BasicIosActorCPU;
};

}
}
} // namespace physx::apex

#endif // __BASIC_IOS_SCENE_H__
