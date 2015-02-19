/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __VORTEX_FS_ACTOR_H__
#define __VORTEX_FS_ACTOR_H__

#include "BasicFSActor.h"
#include "NxVortexFSActor.h"

#include "VortexFSCommon.h"

#include "ReadCheck.h"
#include "WriteCheck.h"

namespace physx
{
namespace apex
{

class NxRenderMeshActor;	
	
namespace basicfs
{

class VortexFSAsset;
class BasicFSScene;
class VortexFSActorParams;

class VortexFSActor : public BasicFSActor, public NxVortexFSActor, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	/* NxVortexFSActor methods */
	VortexFSActor(const VortexFSActorParams&, VortexFSAsset&, NxResourceList&, BasicFSScene&);
	~VortexFSActor();

	NxBasicFSAsset* 		getVortexFSAsset() const;

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
	void					setAxis(const physx::PxVec3& axis)
	{
		NX_WRITE_ZONE();
		mAxis = axis;
		mFieldSamplerChanged = true;
	}
	void					setHeight(physx::PxF32 height)
	{
		NX_WRITE_ZONE();
		mHeight = height;
		mFieldSamplerChanged = true;
		mDebugShapeChanged = true;
	}
	void					setBottomRadius(physx::PxF32 radius)
	{
		mBottomRadius = radius;
		mFieldSamplerChanged = true;
		mDebugShapeChanged = true;
	}
	void					setTopRadius(physx::PxF32 radius)
	{
		NX_WRITE_ZONE();
		mTopRadius = radius;
		mFieldSamplerChanged = true;
		mDebugShapeChanged = true;
	}

	void					setBottomSphericalForce(bool isEnabled)
	{
		NX_WRITE_ZONE();
		mBottomSphericalForce = isEnabled;
		mFieldSamplerChanged = true;
	}
	void					setTopSphericalForce(bool isEnabled)
	{
		NX_WRITE_ZONE();
		mTopSphericalForce = isEnabled;
		mFieldSamplerChanged = true;
	}

	void					setRotationalStrength(physx::PxF32 strength);
	void					setRadialStrength(physx::PxF32 strength);
	void					setLiftStrength(physx::PxF32 strength);

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
	VortexFSAsset*			mAsset;
	
	bool					mBottomSphericalForce;
	bool					mTopSphericalForce;

	physx::PxVec3			mAxis;
	physx::PxF32			mHeight;
	physx::PxF32			mBottomRadius;
	physx::PxF32			mTopRadius;

	physx::PxF32			mRotationalStrength;
	physx::PxF32			mRadialStrength;
	physx::PxF32			mLiftStrength;

	VortexFSParams			mExecuteParams; 

	physx::PxMat34Legacy	mDirToWorld;

	bool						mDebugShapeChanged;
	physx::Array<physx::PxVec3> mDebugPoints;

	friend class BasicFSScene;
};

class VortexFSActorCPU : public VortexFSActor
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	VortexFSActorCPU(const VortexFSActorParams&, VortexFSAsset&, NxResourceList&, BasicFSScene&);
	~VortexFSActorCPU();

	/* NiFieldSampler */
	virtual void executeFieldSampler(const ExecuteData& data);

private:
};

#if defined(APEX_CUDA_SUPPORT)

class VortexFSActorGPU : public VortexFSActorCPU
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	VortexFSActorGPU(const VortexFSActorParams&, VortexFSAsset&, NxResourceList&, BasicFSScene&);
	~VortexFSActorGPU();

	/* NiFieldSampler */
	virtual bool updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled);

	virtual void getFieldSamplerCudaExecuteInfo(CudaExecuteInfo& info) const
	{
		info.executeType = 4;
		info.executeParamsHandle = mParamsHandle;
	}

private:
	ApexCudaConstMemGroup			mConstMemGroup;
	InplaceHandle<VortexFSParams>	mParamsHandle;

};

#endif

}
}
} // end namespace apex

#endif
