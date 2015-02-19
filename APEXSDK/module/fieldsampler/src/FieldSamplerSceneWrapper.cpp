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
#include "FieldSamplerSceneWrapper.h"
#include "FieldSamplerManager.h"
#include "FieldBoundaryWrapper.h"
#include "FieldSamplerWrapper.h"


namespace physx
{
namespace apex
{
namespace fieldsampler
{

FieldSamplerSceneWrapper::FieldSamplerSceneWrapper(NxResourceList& list, FieldSamplerManager* manager, NiFieldSamplerScene* fieldSamplerScene)
	: mManager(manager)
	, mFieldSamplerScene(fieldSamplerScene)
	, mFieldBoundaryListChanged(false)
{
	mFieldSamplerScene->getFieldSamplerSceneDesc(mFieldSamplerSceneDesc);

	list.add(*this);
}

void FieldSamplerSceneWrapper::release()
{
	delete this;
}

FieldSamplerSceneWrapper::FieldBoundaryInfo* FieldSamplerSceneWrapper::addFieldBoundary(FieldBoundaryWrapper* fieldBoundaryWrapper)
{
	FieldBoundaryInfo* fieldBoundaryInfo = NULL;
	for (physx::PxU32 i = 0; i < mFieldBoundaryList.getSize(); ++i)
	{
		FieldBoundaryInfo* info = DYNAMIC_CAST(FieldBoundaryInfo*)(mFieldBoundaryList.getResource(i));
		if (info->getFieldBoundaryWrapper() == fieldBoundaryWrapper)
		{
			fieldBoundaryInfo = info;
			break;
		}
	}
	if (fieldBoundaryInfo == NULL)
	{
		fieldBoundaryInfo = createFieldBoundaryInfo(fieldBoundaryWrapper);
		mFieldBoundaryListChanged = true;
	}
	fieldBoundaryInfo->addRef();
	return fieldBoundaryInfo;
}

void FieldSamplerSceneWrapper::removeFieldBoundary(FieldBoundaryInfo* fieldBoundaryInfo)
{
	if (fieldBoundaryInfo->releaseRef())
	{
		mFieldBoundaryListChanged = true;
	}
}

void FieldSamplerSceneWrapper::update()
{
	for (physx::PxU32 i = 0; i < mFieldBoundaryList.getSize(); ++i)
	{
		FieldBoundaryInfo* info = DYNAMIC_CAST(FieldBoundaryInfo*)(mFieldBoundaryList.getResource(i));
		info->update();
	}
}

/******************************** CPU Version ********************************/
FieldSamplerSceneWrapperCPU::FieldSamplerSceneWrapperCPU(NxResourceList& list, FieldSamplerManager* manager, NiFieldSamplerScene* fieldSamplerScene)
	: FieldSamplerSceneWrapper(list, manager, fieldSamplerScene)
{
}

/******************************** GPU Version ********************************/
#if defined(APEX_CUDA_SUPPORT)

FieldSamplerSceneWrapperGPU::FieldSamplerSceneWrapperGPU(NxResourceList& list, FieldSamplerManager* manager, NiFieldSamplerScene* fieldSamplerScene)
	: FieldSamplerSceneWrapper(list, manager, fieldSamplerScene)
	, mConstStorage(*fieldSamplerScene->getFieldSamplerCudaConstStorage())
{
}

void FieldSamplerSceneWrapperGPU::postUpdate()
{
	PxCudaContextManager* ctx = DYNAMIC_CAST(FieldSamplerSceneGPU*)(mManager->getScene())->getCudaContext();
	physx::PxScopedCudaLock _lock_(*ctx);

	getConstStorage().copyToDevice(ctx, 0);
}


FieldSamplerSceneWrapperGPU::FieldBoundaryInfoGPU::FieldBoundaryInfoGPU(NxResourceList& list, FieldBoundaryWrapper* fieldBoundaryWrapper, ApexCudaConstStorage& constStorage)
	: FieldBoundaryInfo(list, fieldBoundaryWrapper)
	, mConstMemGroup(constStorage)
{
	APEX_CUDA_CONST_MEM_GROUP_SCOPE(mConstMemGroup);
	mFieldShapeGroupParamsHandle.alloc(_storage_);
}

void FieldSamplerSceneWrapperGPU::FieldBoundaryInfoGPU::update()
{
	if (mFieldBoundaryWrapper->getFieldShapesChanged())
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(mConstMemGroup);

		FieldShapeGroupParams shapeGroupParams;
		mFieldShapeGroupParamsHandle.fetch(_storage_, shapeGroupParams);

		const physx::Array<NiFieldShapeDesc>& shapes = mFieldBoundaryWrapper->getFieldShapes();
		physx::PxU32 shapeCount = shapes.size();
		shapeGroupParams.shapeArray.resize(_storage_, shapeCount);
		for (physx::PxU32 i = 0; i < shapeCount; ++i)
		{
			FieldShapeParams elem;
			elem.type          = shapes[i].type;
			elem.dimensions    = shapes[i].dimensions;
			elem.worldToShape  = shapes[i].worldToShape;
			PX_ASSERT(shapes[i].weight >= 0.0f && shapes[i].weight <= 1.0f);
			elem.weight        = PxClamp(shapes[i].weight, 0.0f, 1.0f);
			elem.fade          = 0;

			shapeGroupParams.shapeArray.updateElem(_storage_, elem, i);
		}

		mFieldShapeGroupParamsHandle.update(_storage_, shapeGroupParams);
	}
}

#endif // defined(APEX_CUDA_SUPPORT)

}
}
} // end namespace physx::apex

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
