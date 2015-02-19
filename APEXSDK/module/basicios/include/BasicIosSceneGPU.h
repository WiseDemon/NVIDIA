/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __BASIC_IOS_SCENE_GPU_H__
#define __BASIC_IOS_SCENE_GPU_H__

#if ENABLE_TEST
#include "BasicIosTestScene.h"
#endif
#include "BasicIosScene.h"

#include "../cuda/include/common.h"

#include "ApexCudaWrapper.h"
#include "CudaModuleScene.h"

#define SCENE_CUDA_OBJ(scene, name) static_cast<BasicIosSceneGPU&>(scene).APEX_CUDA_OBJ_NAME(name)

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

class BasicIosSceneGPU : public BASIC_IOS_SCENE, public CudaModuleScene
{
	class EventCallback : public NiFieldSamplerCallback, public NiIofxManagerCallback, public physx::UserAllocated
	{
		void* mEvent;
	public:
		EventCallback();
		void init();
		virtual ~EventCallback();
		void operator()(void* stream);
		PX_INLINE void* getEvent()
		{
			return mEvent;
		}
		bool mIsCalled;
	};
public:
	BasicIosSceneGPU(ModuleBasicIos& module, NiApexScene& scene, NiApexRenderDebug* debugRender, NxResourceList& list);
	~BasicIosSceneGPU();

	virtual BasicIosActor*		createIosActor(NxResourceList& list, BasicIosAsset& asset, physx::apex::NxIofxAsset& iofxAsset);

	virtual void				fetchInjectorParams(PxU32 injectorID, InjectorParams& injParams);
	virtual void				updateInjectorParams(PxU32 injectorID, const InjectorParams& injParams);

	virtual void				fetchResults();

	void*						getHeadCudaObj()
	{
		return CudaModuleScene::getHeadCudaObj();
	}

//CUDA module objects
#include "../cuda/include/moduleList.h"

protected:
	virtual void setCallbacks(BasicIosActorGPU* actor);
	virtual bool growInjectorStorage(physx::PxU32 newSize);

	void onSimulationStart();

private:
	ApexCudaConstMemGroup				mInjectorConstMemGroup;
	InplaceHandle<InjectorParamsArray>	mInjectorParamsArrayHandle;

	EventCallback						mOnSimulationStart;
	physx::Array<EventCallback*>		mOnStartCallbacks;
	physx::Array<EventCallback*>		mOnFinishCallbacks;

	friend class BasicIosActorGPU;
};

}
}
} // namespace physx::apex

#endif // __BASIC_IOS_SCENE_H__
