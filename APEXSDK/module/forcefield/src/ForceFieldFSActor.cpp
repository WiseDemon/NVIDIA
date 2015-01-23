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

#include "ForceFieldActor.h"
#include "ForceFieldAsset.h"
#include "ForceFieldScene.h"
#include "NiFieldSamplerManager.h"
#include "ApexResourceHelper.h"
#include "PsShare.h"

#include "NiApexScene.h"

namespace physx
{
namespace apex
{
namespace forcefield
{

void ForceFieldActor::initFieldSampler(const NxForceFieldActorDesc& desc)
{
	NiFieldSamplerManager* fieldSamplerManager = mForceFieldScene->getNiFieldSamplerManager();
	if (fieldSamplerManager != 0)
	{
		NiFieldSamplerDesc fieldSamplerDesc;
		fieldSamplerDesc.type = NiFieldSamplerType::FORCE;
		fieldSamplerDesc.gridSupportType = NiFieldSamplerGridSupportType::SINGLE_VELOCITY;

		fieldSamplerDesc.samplerFilterData = desc.samplerFilterData;
		fieldSamplerDesc.boundaryFilterData = desc.boundaryFilterData;
		
		fieldSamplerManager->registerFieldSampler(this, fieldSamplerDesc, mForceFieldScene);
		mFieldSamplerChanged = true;
	}
}

void ForceFieldActor::releaseFieldSampler()
{
	NiFieldSamplerManager* fieldSamplerManager = mForceFieldScene->getNiFieldSamplerManager();
	if (fieldSamplerManager != 0)
	{
		fieldSamplerManager->unregisterFieldSampler(this);
	}
}

bool ForceFieldActor::updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled)
{
	isEnabled = mEnable;
	if (mFieldSamplerChanged)
	{
		shapeDesc.type = NiFieldShapeType::NONE;	//not using field sampler include shape (force field has its own implementation for shapes)

		//copy to buffered kernel data for execution
		memcpy(&mKernelExecutionParams, &mKernelParams, sizeof(ForceFieldFSKernelParamsUnion));
		mFieldSamplerChanged = false;
		return true;
	}
	return false;
}

/******************************** CPU Version ********************************/

ForceFieldActorCPU::ForceFieldActorCPU(const NxForceFieldActorDesc& desc, ForceFieldAsset& asset, NxResourceList& list, ForceFieldScene& scene)
	: ForceFieldActor(desc, asset, list, scene)
{
}

ForceFieldActorCPU::~ForceFieldActorCPU()
{
}

void ForceFieldActorCPU::executeFieldSampler(const ExecuteData& data)
{
	// totalElapsedMS is always 0 in PhysX 3
	physx::PxU32 totalElapsedMS = mForceFieldScene->getApexScene().getTotalElapsedMS();

	if (mKernelParams.kernelType == ForceFieldKernelType::RADIAL)
	{
		for (physx::PxU32 iter = 0; iter < data.count; ++iter)
		{
			PxU32 i = data.indices[iter & data.indicesMask] + (iter & ~data.indicesMask);
			physx::PxVec3* pos = (physx::PxVec3*)((physx::PxU8*)data.position + i * data.positionStride);
			data.resultField[iter] = executeForceFieldFS(mKernelExecutionParams.getRadialForceFieldFSKernelParams(), *pos, totalElapsedMS);
		}
	}
	else if (mKernelParams.kernelType == ForceFieldKernelType::GENERIC)
	{
		for (physx::PxU32 iter = 0; iter < data.count; ++iter)
		{
			PxU32 i = data.indices[iter & data.indicesMask] + (iter & ~data.indicesMask);
			physx::PxVec3* pos = (physx::PxVec3*)((physx::PxU8*)data.position + i * data.positionStride);
			physx::PxVec3* vel = (physx::PxVec3*)((physx::PxU8*)data.velocity + i * data.velocityStride);

			data.resultField[iter] = executeForceFieldFS(mKernelExecutionParams.getGenericForceFieldFSKernelParams(), *pos, *vel, totalElapsedMS);
		}
	}
}

/******************************** GPU Version ********************************/

#if defined(APEX_CUDA_SUPPORT)

ForceFieldActorGPU::ForceFieldActorGPU(const NxForceFieldActorDesc& desc, ForceFieldAsset& asset, NxResourceList& list, ForceFieldScene& scene)
	: ForceFieldActorCPU(desc, asset, list, scene)
	, mConstMemGroup(CUDA_OBJ(fieldSamplerStorage))
{
}

ForceFieldActorGPU::~ForceFieldActorGPU()
{
}

bool ForceFieldActorGPU::updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled)
{
	if (ForceFieldActor::updateFieldSampler(shapeDesc, isEnabled))
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(mConstMemGroup);

		if (mParamsHandle.isNull())
		{
			mParamsHandle.alloc(_storage_);
		}
		
		if (mKernelParams.kernelType == ForceFieldKernelType::GENERIC)
		{
			mParamsHandle.update(_storage_, mKernelExecutionParams.getGenericForceFieldFSKernelParams());
		}
		else if (mKernelParams.kernelType == ForceFieldKernelType::RADIAL)
		{
			mParamsHandle.update(_storage_, mKernelExecutionParams.getRadialForceFieldFSKernelParams());
		}
		else
		{
			PX_ASSERT("Wrong kernel type");
		}
		
		return true;
	}
	return false;
}

void ForceFieldActorGPU::getFieldSamplerCudaExecuteInfo(CudaExecuteInfo& info) const
{
	if (mKernelParams.kernelType == ForceFieldKernelType::GENERIC)
	{
		info.executeType = 1;
	}
	else if (mKernelParams.kernelType == ForceFieldKernelType::RADIAL)
	{
		info.executeType = 2;
	}
	else
	{
		PX_ASSERT("Wrong kernel type");
		info.executeType = 0;
	}
	info.executeParamsHandle = mParamsHandle;
}

#endif

}
}
} // namespace physx::apex

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
