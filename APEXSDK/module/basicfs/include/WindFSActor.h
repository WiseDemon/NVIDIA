/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __WIND_FS_ACTOR_H__
#define __WIND_FS_ACTOR_H__

#include "BasicFSActor.h"
#include "NxWindFSActor.h"

#include "WindFSCommon.h"
#include "ApexRWLockable.h"
#include "variable_oscillator.h"

#include "ReadCheck.h"
#include "WriteCheck.h"

namespace physx
{
namespace apex
{

class NxRenderMeshActor;	
	
namespace basicfs
{

class WindFSAsset;
class BasicFSScene;
class WindFSActorParams;

class WindFSActor : public BasicFSActor, public NxWindFSActor, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	/* NxWindFSActor methods */
	WindFSActor(const WindFSActorParams&, WindFSAsset&, NxResourceList&, BasicFSScene&);
	~WindFSActor();

	NxBasicFSAsset* 		getWindFSAsset() const;

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

	void					setFieldStrength(physx::PxF32 strength);
	void					setFieldDirection(const physx::PxVec3& direction);

	physx::PxF32			getFieldStrength() const
	{
		NX_READ_ZONE();
		return mFieldStrength;
	}
	const physx::PxVec3&	getFieldDirection() const
	{
		NX_READ_ZONE();
		return mFieldDirBasis.column0;
	}

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

	virtual physx::PxVec3 queryFieldSamplerVelocity() const
	{
		return mExecuteParams.fieldValue;
	}

	///Sets the uniform overall object scale
	virtual void				setCurrentScale(PxF32 scale) 
	{
		NX_WRITE_ZONE();
		mScale = scale;
		mFieldSamplerChanged = true;
	}

	//Retrieves the uniform overall object scale
	virtual PxF32				getCurrentScale(void) const
	{
		NX_READ_ZONE();
		return mScale;
	}

protected:
	WindFSAsset*			mAsset;

	physx::PxMat33			mFieldDirBasis;
	physx::PxF32			mFieldStrength;

	variableOscillator*		mFieldDirectionVO1;
	variableOscillator*		mFieldDirectionVO2;
	variableOscillator*		mFieldStrengthVO;

	physx::PxF32			mStrengthVar;
	physx::PxVec3			mLocalDirVar;

	WindFSParams			mExecuteParams; 

	friend class BasicFSScene;
};

class WindFSActorCPU : public WindFSActor
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	WindFSActorCPU(const WindFSActorParams&, WindFSAsset&, NxResourceList&, BasicFSScene&);
	~WindFSActorCPU();

	/* NiFieldSampler */
	virtual void executeFieldSampler(const ExecuteData& data);

private:
};

#if defined(APEX_CUDA_SUPPORT)

class WindFSActorGPU : public WindFSActorCPU
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	WindFSActorGPU(const WindFSActorParams&, WindFSAsset&, NxResourceList&, BasicFSScene&);
	~WindFSActorGPU();

	/* NiFieldSampler */
	virtual bool updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled);

	virtual void getFieldSamplerCudaExecuteInfo(CudaExecuteInfo& info) const
	{
		info.executeType = 5;
		info.executeParamsHandle = mParamsHandle;
	}

private:
	ApexCudaConstMemGroup			mConstMemGroup;
	InplaceHandle<WindFSParams>		mParamsHandle;

};

#endif

}
}
} // end namespace apex

#endif
