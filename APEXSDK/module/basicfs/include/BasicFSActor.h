/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __BASIC_FS_ACTOR_H__
#define __BASIC_FS_ACTOR_H__

#include "NxApex.h"

#include "ApexActor.h"
#include "ApexInterface.h"
#include "NiFieldSampler.h"
#include "BasicFSAsset.h"

#include "PxTask.h"

#if defined(APEX_CUDA_SUPPORT)
#include "ApexCudaWrapper.h"
#endif


namespace physx
{
namespace apex
{

class NxRenderMeshActor;

namespace basicfs
{

class BasicFSScene;

class BasicFSActor : public ApexActor, public NxApexResource, public ApexResource, public NiFieldSampler
{
public:
	BasicFSActor(BasicFSScene&);
	virtual ~BasicFSActor();

	/* NxApexResource, ApexResource */
	PxU32					getListIndex() const
	{
		return m_listIndex;
	}
	void					setListIndex(class NxResourceList& list, PxU32 index)
	{
		m_list = &list;
		m_listIndex = index;
	}

	virtual void			visualize()
	{
	}

	virtual void			simulate(physx::PxF32 dt)
	{
		PX_UNUSED(dt);
	}


#if NX_SDK_VERSION_MAJOR == 2
	void					setPhysXScene(NxScene*);
	NxScene*				getPhysXScene() const;
#elif NX_SDK_VERSION_MAJOR == 3
	void					setPhysXScene(PxScene*);
	PxScene*				getPhysXScene() const;
#endif

	/* NiFieldSampler */
	virtual bool			updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled) = 0;

protected:
	BasicFSScene*			mScene;

	physx::PxMat34Legacy	mPose;
	physx::PxF32			mScale;

	bool					mFieldSamplerChanged;
	bool					mFieldSamplerEnabled;

	physx::PxF32			mFieldWeight;

	friend class BasicFSScene;
};

}
}
} // end namespace apex

#endif
