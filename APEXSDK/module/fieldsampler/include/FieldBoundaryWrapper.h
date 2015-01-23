/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FIELD_BOUNDARY_WRAPPER_H__
#define __FIELD_BOUNDARY_WRAPPER_H__

#include "NxApex.h"
#include "ApexSDKHelpers.h"
#include "ApexActor.h"
#include "NiFieldBoundary.h"

#if defined(APEX_CUDA_SUPPORT)
#include "ApexCudaWrapper.h"
#endif

#include "FieldSamplerCommon.h"


namespace physx
{
namespace apex
{
namespace fieldsampler
{

class FieldSamplerManager;


class FieldBoundaryWrapper : public NxApexResource, public ApexResource
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

	FieldBoundaryWrapper(NxResourceList& list, FieldSamplerManager* manager, NiFieldBoundary* fieldBoundary, const NiFieldBoundaryDesc& fieldBoundaryDesc);

	NiFieldBoundary* getNiFieldBoundary() const
	{
		return mFieldBoundary;
	}
	PX_INLINE const NiFieldBoundaryDesc& getNiFieldBoundaryDesc() const
	{
		return mFieldBoundaryDesc;
	}

	void update();

	const physx::Array<NiFieldShapeDesc>&	getFieldShapes() const
	{
		return mFieldShapes;
	}
	bool									getFieldShapesChanged() const
	{
		return mFieldShapesChanged;
	}

protected:
	FieldSamplerManager*			mManager;

	NiFieldBoundary*				mFieldBoundary;
	NiFieldBoundaryDesc				mFieldBoundaryDesc;

	physx::Array<NiFieldShapeDesc>	mFieldShapes;
	bool							mFieldShapesChanged;
};

}
}
} // end namespace physx::apex

#endif
