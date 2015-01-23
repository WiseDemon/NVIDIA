/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FIELD_SAMPLER_WRAPPER_H__
#define __FIELD_SAMPLER_WRAPPER_H__

#include "NxApex.h"
#include "ApexSDKHelpers.h"
#include "ApexActor.h"
#include "NiFieldSampler.h"

#if defined(APEX_CUDA_SUPPORT)
#include "ApexCudaWrapper.h"
#endif

#include "FieldSamplerSceneWrapper.h"
#include "FieldSamplerCommon.h"

namespace physx
{
namespace apex
{
namespace fieldsampler
{

class FieldSamplerManager;

class FieldBoundaryWrapper;

class FieldSamplerWrapper : public NxApexResource, public ApexResource
{
public:
	// NxApexResource methods
	void			release();
	void			setListIndex(NxResourceList& list, physx::PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}
	physx::PxU32	getListIndex() const
	{
		return m_listIndex;
	}

	FieldSamplerWrapper(NxResourceList& list, FieldSamplerManager* manager, NiFieldSampler* fieldSampler, const NiFieldSamplerDesc& fieldSamplerDesc, FieldSamplerSceneWrapper* fieldSamplerSceneWrapper);

	virtual void update();

	PX_INLINE NiFieldSampler* getNiFieldSampler() const
	{
		return mFieldSampler;
	}
	PX_INLINE const NiFieldSamplerDesc& getNiFieldSamplerDesc() const
	{
		return mFieldSamplerDesc;
	}
	PX_INLINE FieldSamplerSceneWrapper* getFieldSamplerSceneWrapper() const
	{
		return mSceneWrapper;
	}

	bool addFieldBoundary(FieldBoundaryWrapper* wrapper);
	bool removeFieldBoundary(FieldBoundaryWrapper* wrapper);

	PxU32 getFieldBoundaryCount() const
	{
		return mFieldBoundaryInfoArray.size();
	}
	FieldBoundaryWrapper* getFieldBoundaryWrapper(physx::PxU32 index) const
	{
		return mFieldBoundaryInfoArray[index]->getFieldBoundaryWrapper();
	}

	PX_INLINE const NiFieldShapeDesc&   getNiFieldSamplerShape() const
	{
		return mFieldSamplerShape;
	}
	PX_INLINE bool                      isFieldSamplerChanged() const
	{
		return mFieldSamplerShapeChanged;
	}
	PX_INLINE bool                      isEnabled() const
	{
		return mIsEnabled;
	}
	PX_INLINE bool                      isEnabledChanged() const
	{
		return (mIsEnabled != mIsEnabledLast);
	}

protected:
	FieldSamplerManager*		mManager;
	NiFieldSampler*				mFieldSampler;
	NiFieldSamplerDesc			mFieldSamplerDesc;

	NiFieldShapeDesc			mFieldSamplerShape;
	bool						mFieldSamplerShapeChanged;

	FieldSamplerSceneWrapper*	mSceneWrapper;
	physx::PxU32				mQueryRefCount;

	physx::Array<FieldSamplerSceneWrapper::FieldBoundaryInfo*>	mFieldBoundaryInfoArray;
	bool														mFieldBoundaryInfoArrayChanged;

	bool						mIsEnabled;
	bool						mIsEnabledLast;

	friend class FieldSamplerManager;
};


class FieldSamplerWrapperCPU : public FieldSamplerWrapper
{
public:
	FieldSamplerWrapperCPU(NxResourceList& list, FieldSamplerManager* manager, NiFieldSampler* fieldSampler, const NiFieldSamplerDesc& fieldSamplerDesc, FieldSamplerSceneWrapper* fieldSamplerSceneWrapper);

private:
};

#if defined(APEX_CUDA_SUPPORT)
class FieldSamplerWrapperGPU : public FieldSamplerWrapperCPU
{
public:
	FieldSamplerWrapperGPU(NxResourceList& list, FieldSamplerManager* manager, NiFieldSampler* fieldSampler, const NiFieldSamplerDesc& fieldSamplerDesc, FieldSamplerSceneWrapper* fieldSamplerSceneWrapper);

	virtual void update();

	PX_INLINE InplaceHandle<FieldSamplerParams>   getParamsHandle() const
	{
		PX_ASSERT(mFieldSamplerParamsHandle.isNull() == false);
		return mFieldSamplerParamsHandle;
	}

private:
	ApexCudaConstMemGroup               mConstMemGroup;
	InplaceHandle<FieldSamplerParams>   mFieldSamplerParamsHandle;
};
#endif

}
}
} // end namespace physx::apex

#endif
