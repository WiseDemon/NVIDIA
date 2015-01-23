/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __PARTICLE_IOS_ACTOR_CPU_H__
#define __PARTICLE_IOS_ACTOR_CPU_H__

#include "NxApex.h"

#include "ParticleIosActor.h"
#include "ParticleIosAsset.h"
#include "NiInstancedObjectSimulation.h"
#include "ParticleIosScene.h"
#include "ApexActor.h"
#include "ApexContext.h"
#include "ApexFIFO.h"

#include "PxTask.h"

namespace physx
{
namespace apex
{

namespace iofx
{
class NxApexRenderVolume;
class NxIofxAsset;
}

namespace pxparticleios
{

class ParticleIosActorCPU : public ParticleIosActor
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ParticleIosActorCPU(NxResourceList&, ParticleIosAsset&, ParticleIosScene&, NxIofxAsset&);
	~ParticleIosActorCPU();

	virtual physx::PxTaskID				submitTasks(physx::PxTaskManager* tm);
	virtual void						setTaskDependencies(physx::PxTaskID taskStartAfterID, physx::PxTaskID taskFinishBeforeID);

private:
	/* Internal utility functions */
	void								simulateParticles();

	static const physx::PxU32 HISTOGRAM_BIN_COUNT = 1024;
	physx::PxU32						computeHistogram(physx::PxU32 dataCount, physx::PxF32 dataMin, physx::PxF32 dataMax, physx::PxU32& bound);

	/* particle data (output to the IOFX actors, and some state) */

	struct NewParticleData
	{
		PxU32  destIndex;
		PxVec3 position;
		PxVec3 velocity;
	};
	physx::Array<physx::PxU32>			mNewIndices;
	physx::Array<physx::PxU32>			mRemovedParticleList;
	physx::Array<NewParticleData>		mAddedParticleList;
	PxParticleExt::IndexPool*			mIndexPool;

	/* Field sampler update velocity */
	physx::Array<physx::PxU32>			mUpdateIndexBuffer;
	physx::Array<physx::PxVec3>			mUpdateVelocityBuffer;

	class SimulateTask : public physx::PxTask
	{
	public:
		SimulateTask(ParticleIosActorCPU& actor) : mActor(actor) {}

		const char* getName() const
		{
			return "ParticleIosActorCPU::SimulateTask";
		}
		void run()
		{
			mActor.simulateParticles();
		}

	protected:
		ParticleIosActorCPU& mActor;

	private:
		SimulateTask& operator=(const SimulateTask&);
	};
	SimulateTask						mSimulateTask;

	ApexCpuInplaceStorage				mSimulationStorage;

	friend class ParticleIosAsset;
};

}
}
} // namespace physx::apex

#endif // __PARTICLE_IOS_ACTOR_CPU_H__
