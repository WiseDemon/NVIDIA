/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __NOISE_FS_ACTOR_H__
#define __NOISE_FS_ACTOR_H__

#include "BasicFSActor.h"
#include "NxNoiseFSActor.h"
#include "ApexRWLockable.h"
#include "NoiseFSCommon.h"

#include "ReadCheck.h"
#include "WriteCheck.h"

namespace physx
{
namespace apex
{

class NxRenderMeshActor;	
	
namespace basicfs
{

class NoiseFSAsset;
class BasicFSScene;
class NoiseFSActorParams;

class NoiseFSActor : public BasicFSActor, public NxNoiseFSActor, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	/* NxNoiseFSActor methods */
	NoiseFSActor(const NoiseFSActorParams&, NoiseFSAsset&, NxResourceList&, BasicFSScene&);
	~NoiseFSActor();

	NxBasicFSAsset* 		getNoiseFSAsset() const;

	physx::PxMat44			getCurrentPose() const
	{
		NX_READ_ZONE();
		return PxMat44(mPose);
	}

	void					setCurrentPose(const physx::PxMat44& pose)
	{
		NX_WRITE_ZONE();
		mPose = pose;
		mFieldSamplerChanged = true;
	}

	physx::PxVec3			getCurrentPosition() const
	{		
		NX_READ_ZONE();
		return mPose.t;
	}
	void					setCurrentPosition(const physx::PxVec3& pos)
	{
		NX_WRITE_ZONE();
		mPose.t = pos;
		mFieldSamplerChanged = true;
	}

	physx::PxF32			getCurrentScale() const
	{
		NX_READ_ZONE();
		return mScale;
	}

	void					setCurrentScale(const physx::PxF32& scale)
	{
		NX_WRITE_ZONE();
		mScale = scale;
		mFieldSamplerChanged = true;
	}

	void					setNoiseStrength(physx::PxF32 strength);

	void					setEnabled(bool isEnabled)
	{
		NX_WRITE_ZONE();
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

protected:
	NoiseFSAsset*				mAsset;

	NoiseFSParams				mExecuteParams; 

	friend class BasicFSScene;
};

class NoiseFSActorCPU : public NoiseFSActor
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	NoiseFSActorCPU(const NoiseFSActorParams&, NoiseFSAsset&, NxResourceList&, BasicFSScene&);
	~NoiseFSActorCPU();

	/* NiFieldSampler */
	virtual void executeFieldSampler(const ExecuteData& data);

private:
};

#if defined(APEX_CUDA_SUPPORT)

class NoiseFSActorGPU : public NoiseFSActorCPU
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	NoiseFSActorGPU(const NoiseFSActorParams&, NoiseFSAsset&, NxResourceList&, BasicFSScene&);
	~NoiseFSActorGPU();

	/* NiFieldSampler */
	virtual bool updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled);

	virtual void getFieldSamplerCudaExecuteInfo(CudaExecuteInfo& info) const
	{
		info.executeType = 3;
		info.executeParamsHandle = mParamsHandle;
	}

private:
	ApexCudaConstMemGroup			mConstMemGroup;
	InplaceHandle<NoiseFSParams>	mParamsHandle;

};

#endif

}
}
} // end namespace apex

#endif
