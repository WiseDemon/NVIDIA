/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __ATTRACTOR_FS_ACTOR_H__
#define __ATTRACTOR_FS_ACTOR_H__

#include "BasicFSActor.h"
#include "NxAttractorFSActor.h"
#include "ApexRWLockable.h"
#include "AttractorFSCommon.h"


namespace physx
{
namespace apex
{

class NxRenderMeshActor;

namespace basicfs
{

class AttractorFSAsset;
class BasicFSScene;
class AttractorFSActorParams;

class AttractorFSActor : public BasicFSActor, public NxAttractorFSActor, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	/* NxAttractorFSActor methods */
	AttractorFSActor(const AttractorFSActorParams&, AttractorFSAsset&, NxResourceList&, BasicFSScene&);
	~AttractorFSActor();

	NxBasicFSAsset* 		getAttractorFSAsset() const;

	physx::PxVec3			getCurrentPosition() const
	{
		return mPose.t;
	}
	void					setCurrentPosition(const physx::PxVec3& pos)
	{
		mPose.t = pos;
		mFieldSamplerChanged = true;
	}
	void					setFieldRadius(physx::PxF32 radius)
	{
		mRadius = radius;
		mFieldSamplerChanged = true;
	}
	void					setConstFieldStrength(physx::PxF32 strength);

	void					setVariableFieldStrength(physx::PxF32 strength);

	void					setEnabled(bool isEnabled)
	{
		mFieldSamplerEnabled = isEnabled;
	}

	/* NxApexRenderable, NxApexRenderDataProvider */
	void					updateRenderResources(bool rewriteBuffers, void* userRenderData);
	void					dispatchRenderResources(NxUserRenderer& renderer);

	PxBounds3				getBounds() const
	{
		return ApexRenderable::getBounds();
	}

	void					lockRenderResources()
	{
		ApexRenderable::renderDataLock();
	}
	void					unlockRenderResources()
	{
		ApexRenderable::renderDataUnLock();
	}

	void					getPhysicalLodRange(PxReal& min, PxReal& max, bool& intOnly) const;
	physx::PxF32			getActivePhysicalLod() const;
	void					forcePhysicalLod(PxReal lod);
	/**
	\brief Selectively enables/disables debug visualization of a specific APEX actor.  Default value it true.
	*/
	virtual void setEnableDebugVisualization(bool state)
	{
		ApexActor::setEnableDebugVisualization(state);
	}

	NxApexRenderable*		getRenderable()
	{
		return this;
	}
	NxApexActor*			getNxApexActor()
	{
		return this;
	}

	/* NxApexResource, ApexResource */
	void					release();

	/* NxApexActor, ApexActor */
	void					destroy();
	NxApexAsset*			getOwner() const;

	virtual void			simulate(physx::PxF32 dt);

	virtual void			visualize();

	/* NiFieldSampler */
	virtual bool			updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled);

	///Sets the uniform overall object scale
	virtual void				setCurrentScale(PxF32 scale) 
	{
		mScale = scale;
		mFieldSamplerChanged = true;
	}

	//Retrieves the uniform overall object scale
	virtual PxF32				getCurrentScale(void) const
	{
		return mScale;
	}

protected:
	AttractorFSAsset*		mAsset;

	physx::PxF32			mRadius;

	physx::PxF32			mConstFieldStrength;
	physx::PxF32			mVariableFieldStrength;

	AttractorFSParams		mExecuteParams; 

	physx::Array<physx::PxVec3> mDebugPoints;

	friend class BasicFSScene;
};

class AttractorFSActorCPU : public AttractorFSActor
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	AttractorFSActorCPU(const AttractorFSActorParams&, AttractorFSAsset&, NxResourceList&, BasicFSScene&);
	~AttractorFSActorCPU();

	/* NiFieldSampler */
	virtual void executeFieldSampler(const ExecuteData& data);

private:
};

#if defined(APEX_CUDA_SUPPORT)

class AttractorFSActorGPU : public AttractorFSActorCPU
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	AttractorFSActorGPU(const AttractorFSActorParams&, AttractorFSAsset&, NxResourceList&, BasicFSScene&);
	~AttractorFSActorGPU();

	/* NiFieldSampler */
	virtual bool updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled);

	virtual void getFieldSamplerCudaExecuteInfo(CudaExecuteInfo& info) const
	{
		info.executeType = 2;
		info.executeParamsHandle = mParamsHandle;
	}

private:
	ApexCudaConstMemGroup				mConstMemGroup;
	InplaceHandle<AttractorFSParams>	mParamsHandle;

};

#endif

}
}
} // end namespace apex

#endif
