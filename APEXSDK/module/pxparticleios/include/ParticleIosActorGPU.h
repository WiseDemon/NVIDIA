/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __PARTICLE_IOS_ACTOR_GPU_H__
#define __PARTICLE_IOS_ACTOR_GPU_H__

#include "NxApex.h"

#include "ParticleIosActor.h"
#include "ParticleIosAsset.h"
#include "NiInstancedObjectSimulation.h"
#include "ParticleIosScene.h"
#include "ApexActor.h"
#include "ApexContext.h"
#include "ApexFIFO.h"
#include "NiFieldSamplerQuery.h"

#include "PxGpuTask.h"

namespace physx
{
namespace apex
{

namespace iofx
{
class NxIofxActor;
class NxApexRenderVolume;
}
	
namespace pxparticleios
{

class ParticleIosActorGPU;

class ParticleIosActorGPU : public ParticleIosActor
{
public:
	ParticleIosActorGPU(NxResourceList&, ParticleIosAsset&, ParticleIosScene&, NxIofxAsset&);
	~ParticleIosActorGPU();

	virtual physx::PxTaskID				submitTasks(physx::PxTaskManager* tm);
	virtual void						setTaskDependencies(physx::PxTaskID taskStartAfterID, physx::PxTaskID taskFinishBeforeID);
	virtual void						fetchResults();

private:
	bool								launch(CUstream stream, int kernelIndex);
	void								trigger();

	CUevent								mCuSyncEvent;
	physx::PxGpuCopyDescQueue			mCopyQueue;

	ApexMirroredArray<physx::PxU32>		mHoleScanSum;
	ApexMirroredArray<physx::PxU32>		mMoveIndices;

	ApexMirroredArray<physx::PxU32>		mTmpReduce;
	ApexMirroredArray<physx::PxU32>		mTmpHistogram;
	ApexMirroredArray<physx::PxU32>		mTmpScan;
	ApexMirroredArray<physx::PxU32>		mTmpScan1;

	ApexMirroredArray<physx::PxU32>		mTmpOutput;	// 0:STATUS_LASTACTIVECOUNT, ...
	ApexMirroredArray<physx::PxU32>		mTmpBoundParams;	// min, max
#if defined(APEX_TEST)
	ApexMirroredArray<physx::PxU32>		mTestMirroredArray;

	ApexCudaConstMemGroup				mTestConstMemGroup;
	InplaceHandle<int>					mTestITHandle;
#endif

	class LaunchTask : public physx::PxGpuTask
	{
	public:
		LaunchTask(ParticleIosActorGPU& actor) : mActor(actor) {}
		const char*	getName() const
		{
			return "ParticleIosActorGPU::LaunchTask";
		}
		void		run()
		{
			PX_ALWAYS_ASSERT();
		}
		bool		launchInstance(CUstream stream, int kernelIndex)
		{
			return mActor.launch(stream, kernelIndex);
		}
		physx::PxGpuTaskHint::Enum getTaskHint() const
		{
			return physx::PxGpuTaskHint::Kernel;
		}

	protected:
		ParticleIosActorGPU& mActor;

	private:
		LaunchTask& operator=(const LaunchTask&);
	};
	class TriggerTask : public physx::PxTask
	{
	public:
		TriggerTask(ParticleIosActorGPU& actor) : mActor(actor) {}

		const char* getName() const
		{
			return "ParticleIosActorGPU::TriggerTask";
		}
		void run()
		{
			mActor.trigger();
		}

	protected:
		ParticleIosActorGPU& mActor;

	private:
		TriggerTask& operator=(const TriggerTask&);
	};


	static PX_CUDA_CALLABLE PX_INLINE PxMat44 inverse(const PxMat44& in);
	static PxReal distance(PxVec4 a, PxVec4 b);

	LaunchTask							mLaunchTask;
	TriggerTask							mTriggerTask;
};

}
}
} // namespace physx::apex

#endif // __PARTICLE_IOS_ACTOR_GPU_H__
