/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "NxApexDefs.h"
#include "MinPhysxSdkVersion.h"
#if NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED

#include "NxApex.h"
#include "FieldBoundaryWrapper.h"
#include "FieldSamplerManager.h"


namespace physx
{
namespace apex
{
namespace fieldsampler
{

FieldBoundaryWrapper::FieldBoundaryWrapper(NxResourceList& list, FieldSamplerManager* manager, NiFieldBoundary* fieldBoundary, const NiFieldBoundaryDesc& fieldBoundaryDesc)
	: mManager(manager)
	, mFieldBoundary(fieldBoundary)
	, mFieldBoundaryDesc(fieldBoundaryDesc)
	, mFieldShapesChanged(false)
{
	list.add(*this);

}

void FieldBoundaryWrapper::release()
{
	delete this;
}

void FieldBoundaryWrapper::update()
{
	mFieldShapesChanged = mFieldBoundary->updateFieldBoundary(mFieldShapes);
}

}
}
} // end namespace physx::apex

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
