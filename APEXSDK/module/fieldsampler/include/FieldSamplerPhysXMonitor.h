/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FIELD__SAMPLER_PHYSX_MONITOR_H___
#define __FIELD__SAMPLER_PHYSX_MONITOR_H___

#include "NxApex.h"
#include <PsArray.h>

#include "NiFieldSamplerScene.h"
#include "FieldSamplerPhysXMonitorParams.h"

#if NX_SDK_VERSION_MAJOR == 3
#include <PxScene.h>
#endif

namespace physx
{
namespace apex
{

class NiFieldSamplerQuery;

namespace fieldsampler
{

class FieldSamplerScene;
class FieldSamplerManager;


struct ShapeData : public UserAllocated
{
	PxU32	fdIndex;
	PxU32	rbIndex;
	PxF32	mass;
	PxVec3	pos;
	PxVec3	vel;

	static bool sortPredicate (ShapeData* sd1, ShapeData* sd2)
	{
		if (sd1 == 0) return false;
		if (sd2 == 0) return true;
		return sd1->fdIndex < sd2->fdIndex;
	}
};


class FieldSamplerPhysXMonitor : public UserAllocated
{
private:


public:
	FieldSamplerPhysXMonitor(FieldSamplerScene& scene);
	virtual ~FieldSamplerPhysXMonitor();

#if NX_SDK_VERSION_MAJOR == 3
	virtual void	update();
	virtual void	updatePhysX();

	/* PhysX scene management */
	void	setPhysXScene(PxScene* scene);
	PX_INLINE PxScene*	getPhysXScene() const
	{
		return mScene;
	}

	/* Toggle PhysX Monitor on/off */
	PX_INLINE void enablePhysXMonitor(bool enable)
	{
		mEnable = enable;
	}

	/* Is PhysX Monitor enabled */
	PX_INLINE bool isEnable()
	{
		return mEnable;
	}

	PX_INLINE void FieldSamplerPhysXMonitor::setPhysXFilterData(physx::PxFilterData filterData)
	{
		mFilterData = filterData;
	}

private:
	FieldSamplerPhysXMonitor& operator=(const FieldSamplerPhysXMonitor&);

	void getParticles(physx::PxU32 taskId);
	void updateParticles();
	void getRigidBodies(physx::PxU32 taskId);
	void updateRigidBodies();
	//void getCloth(physx::PxTask& task, bool isDataOnDevice);
	//void updateCloth();

protected:
	void commonInitArray();

	FieldSamplerScene*				mFieldSamplerScene;
	FieldSamplerManager*			mFieldSamplerManager;

	PxScene*						mScene;
	
	FieldSamplerPhysXMonitorParams*	mParams;
	
	PxFilterData					mFilterData;

	//Particles
	physx::PxU32					mNumPS;  //Number of particle systems
	physx::PxU32					mPCount; //Number of particles in buffer
	Array<PxActor*>					mParticleSystems;
	Array<PxF32>					mPSMass;
	Array<PxVec4>					mPSOutField;
	Array<PxVec3>					mOutVelocities;
	Array<PxU32>					mOutIndices;
	Array<PxParticleReadData*>		mParticleReadData;
	Array<NiFieldSamplerQuery*>		mPSFieldSamplerQuery;	
	Array<physx::PxTaskID>			mPSFieldSamplerTaskID;


	//Rigid bodies
	physx::PxU32					mNumRB; //Number of rigid bodies
	Array<PxActor*>					mRBActors;
	Array<ShapeData*>				mRBIndex;
	Array<PxVec4>					mRBInPosition;
	Array<PxVec4>					mRBInVelocity;
	Array<PxVec4>					mRBOutField;
	Array<PxFilterData>				mRBFilterData;
	Array<NiFieldSamplerQuery*>		mRBFieldSamplerQuery;	

	//Enable or disable PhysX Monitor
	bool mEnable;

public:
	class RunAfterActorUpdateTask : public physx::PxTask
	{
	public:
		RunAfterActorUpdateTask(FieldSamplerPhysXMonitor& owner) : mOwner(owner) {}
		const char* getName() const
		{
			return FSST_PHYSX_MONITOR_UPDATE;
		}
		void run()
		{
			mOwner.updatePhysX();
		}

	protected:
		FieldSamplerPhysXMonitor& mOwner;

	private:
		RunAfterActorUpdateTask operator=(const RunAfterActorUpdateTask&);
	};
	RunAfterActorUpdateTask				mTaskRunAfterActorUpdate;
#endif
};


}
}
} // end namespace physx::apex

#endif

