/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __BASIC_IOS_ACTOR_GPU_H__
#define __BASIC_IOS_ACTOR_GPU_H__

#include "NxApex.h"

#if ENABLE_TEST
#include "BasicIosTestActor.h"
#endif
#include "BasicIosActor.h"
#include "BasicIosAsset.h"
#include "NiInstancedObjectSimulation.h"
#include "BasicIosSceneGPU.h"
#include "ApexActor.h"
#include "ApexContext.h"
#include "ApexFIFO.h"
#include "NiFieldSamplerQuery.h"

#include "PxGpuTask.h"

namespace physx
{
namespace apex
{

namespace IOFX
{

class NxIofxActor;
class NxApexRenderVolume;

}

namespace basicios
{

#if ENABLE_TEST
#define BASIC_IOS_ACTOR BasicIosTestActor
#else
#define BASIC_IOS_ACTOR BasicIosActor
#endif

class BasicIosActorGPU : public BASIC_IOS_ACTOR
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	BasicIosActorGPU(NxResourceList&, BasicIosAsset&, BasicIosScene&, physx::apex::NxIofxAsset&, const ApexMirroredPlace::Enum defaultPlace = ApexMirroredPlace::GPU);
	~BasicIosActorGPU();

	virtual void						submitTasks();
	virtual void						setTaskDependencies();
	virtual void						fetchResults();

protected:
	bool								launch(CUstream stream, int kernelIndex);

	physx::PxGpuCopyDescQueue		mCopyQueue;

	ApexMirroredArray<physx::PxU32>		mHoleScanSum;
	ApexMirroredArray<physx::PxU32>		mMoveIndices;

	ApexMirroredArray<physx::PxU32>		mTmpReduce;
	ApexMirroredArray<physx::PxU32>		mTmpHistogram;
	ApexMirroredArray<physx::PxU32>		mTmpScan;
	ApexMirroredArray<physx::PxU32>		mTmpScan1;

	ApexMirroredArray<physx::PxU32>		mTmpOutput;
	ApexMirroredArray<physx::PxU32>		mTmpOutput1;

	class LaunchTask : public physx::PxGpuTask
	{
	public:
		LaunchTask(BasicIosActorGPU& actor) : mActor(actor) {}
		const char* getName() const
		{
			return "BasicIosActorGPU::LaunchTask";
		}
		void         run()
		{
			PX_ALWAYS_ASSERT();
		}
		bool         launchInstance(CUstream stream, int kernelIndex)
		{
			return mActor.launch(stream, kernelIndex);
		}
		physx::PxGpuTaskHint::Enum getTaskHint() const
		{
			return physx::PxGpuTaskHint::Kernel;
		}

	protected:
		BasicIosActorGPU& mActor;

	private:
		LaunchTask& operator=(const LaunchTask&);
	};

	static PX_CUDA_CALLABLE PX_INLINE PxMat44 inverse(const PxMat44& in);
	static PxReal distance(PxVec4 a, PxVec4 b);

	LaunchTask							mLaunchTask;
};

}
}
} // namespace physx::apex

#endif // __BASIC_IOS_ACTOR_GPU_H__
