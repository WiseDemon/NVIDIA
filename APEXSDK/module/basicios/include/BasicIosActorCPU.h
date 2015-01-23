/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __BASIC_IOS_ACTOR_CPU_H__
#define __BASIC_IOS_ACTOR_CPU_H__

#include "NxApex.h"

#if ENABLE_TEST
#include "BasicIosTestActor.h"
#endif
#include "BasicIosActor.h"
#include "BasicIosAsset.h"
#include "NiInstancedObjectSimulation.h"
#include "BasicIosSceneCPU.h"
#include "ApexActor.h"
#include "ApexContext.h"
#include "ApexFIFO.h"
#include "ApexRWLockable.h"
#include "PxTask.h"

namespace physx
{
namespace apex
{

class NxApexRenderVolume;

namespace basicios
{

#if ENABLE_TEST
#define BASIC_IOS_ACTOR BasicIosTestActor
#else
#define BASIC_IOS_ACTOR BasicIosActor
#endif

class BasicIosActorCPU : public BASIC_IOS_ACTOR
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	BasicIosActorCPU(NxResourceList&, BasicIosAsset&, BasicIosScene&, physx::apex::NxIofxAsset&);
	~BasicIosActorCPU();

	virtual void						submitTasks();
	virtual void						setTaskDependencies();
	virtual void						fetchResults();

protected:
	/* Internal utility functions */
	void								simulateParticles();

	static const physx::PxU32 HISTOGRAM_BIN_COUNT = 1024;
	physx::PxU32						computeHistogram(physx::PxU32 dataCount, physx::PxF32 dataMin, physx::PxF32 dataMax, physx::PxU32& bound);

private:
	/* particle data (output to the IOFX actors, and some state) */

	physx::Array<physx::PxU32>			mNewIndices;

	class SimulateTask : public physx::PxTask
	{
	public:
		SimulateTask(BasicIosActorCPU& actor) : mActor(actor) {}

		const char* getName() const
		{
			return "BasicIosActorCPU::SimulateTask";
		}
		void run()
		{
			mActor.simulateParticles();
		}

	protected:
		BasicIosActorCPU& mActor;

	private:
		SimulateTask& operator=(const SimulateTask&);
	};
	SimulateTask						mSimulateTask;

	ApexCpuInplaceStorage				mSimulationStorage;

	friend class BasicIosAsset;
};

}
}
} // namespace physx::apex

#endif // __BASIC_IOS_ACTOR_CPU_H__
