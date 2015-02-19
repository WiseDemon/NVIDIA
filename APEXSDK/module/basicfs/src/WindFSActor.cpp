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
#include "NxRenderMeshActorDesc.h"
#include "NxRenderMeshActor.h"
#include "NxRenderMeshAsset.h"

#if NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED

#include "NxApex.h"

#include "WindFSActor.h"
#include "WindFSAsset.h"
#include "BasicFSScene.h"
#include "NiApexSDK.h"
#include "NiApexScene.h"
#include "NiApexRenderDebug.h"

#if NX_SDK_VERSION_MAJOR == 2
#include <NxScene.h>
#include "NxFromPx.h"
#elif NX_SDK_VERSION_MAJOR == 3
#include <PxScene.h>
#endif

#include <NiFieldSamplerManager.h>
#include "ApexResourceHelper.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

WindFSActor::WindFSActor(const WindFSActorParams& params, WindFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: BasicFSActor(scene)
	, mAsset(&asset)
	, mFieldDirectionVO1(NULL)
	, mFieldDirectionVO2(NULL)
	, mFieldStrengthVO(NULL)
{
	mFieldWeight = asset.mParams->fieldWeight;

	mPose = params.initialPose;
	mScale = params.initialScale;
	setFieldDirection(mAsset->mParams->fieldDirection);
	setFieldStrength(mAsset->mParams->fieldStrength);

	mExecuteParams.fieldValue = getFieldDirection() * (getFieldStrength() * mScale);

	mStrengthVar = 0.0f;
	mLocalDirVar = PxVec3(1, 0, 0);

	if (mAsset->mParams->fieldStrengthDeviationPercentage > 0 && mAsset->mParams->fieldStrengthOscillationPeriod > 0)
	{
		mFieldStrengthVO = PX_NEW(variableOscillator)(-mAsset->mParams->fieldStrengthDeviationPercentage,
													   +mAsset->mParams->fieldStrengthDeviationPercentage,
													   0.0f,
													   mAsset->mParams->fieldStrengthOscillationPeriod);
	}

	PxF32 diviationAngle = physx::degToRad(mAsset->mParams->fieldDirectionDeviationAngle);
	if (diviationAngle > 0 && mAsset->mParams->fieldDirectionOscillationPeriod > 0)
	{
		mFieldDirectionVO1 = PX_NEW(variableOscillator)(-diviationAngle,
														 +diviationAngle,
														 0,
														 mAsset->mParams->fieldDirectionOscillationPeriod);

		mFieldDirectionVO2 = PX_NEW(variableOscillator)(-PxTwoPi,
														 +PxTwoPi,
														 0,
														 mAsset->mParams->fieldDirectionOscillationPeriod);
	}

	list.add(*this);			// Add self to asset's list of actors
	addSelfToContext(*scene.getApexScene().getApexContext());    // Add self to ApexScene
	addSelfToContext(scene);	// Add self to BasicFSScene's list of actors

	NiFieldSamplerManager* fieldSamplerManager = mScene->getNiFieldSamplerManager();
	if (fieldSamplerManager != 0)
	{
		NiFieldSamplerDesc fieldSamplerDesc;
		if (asset.mParams->fieldDragCoeff > 0)
		{
			fieldSamplerDesc.type = NiFieldSamplerType::VELOCITY_DRAG;
			fieldSamplerDesc.dragCoeff = asset.mParams->fieldDragCoeff;
		}
		else
		{
			fieldSamplerDesc.type = NiFieldSamplerType::VELOCITY_DIRECT;
		}
		fieldSamplerDesc.gridSupportType = NiFieldSamplerGridSupportType::SINGLE_VELOCITY;
#if NX_SDK_VERSION_MAJOR == 2
		fieldSamplerDesc.samplerFilterData = ApexResourceHelper::resolveCollisionGroup64(params.fieldSamplerFilterDataName ? params.fieldSamplerFilterDataName : mAsset->mParams->fieldSamplerFilterDataName);
		fieldSamplerDesc.boundaryFilterData = ApexResourceHelper::resolveCollisionGroup64(params.fieldBoundaryFilterDataName ? params.fieldBoundaryFilterDataName : mAsset->mParams->fieldBoundaryFilterDataName);
#else
		fieldSamplerDesc.samplerFilterData = ApexResourceHelper::resolveCollisionGroup128(params.fieldSamplerFilterDataName ? params.fieldSamplerFilterDataName : mAsset->mParams->fieldSamplerFilterDataName);
		fieldSamplerDesc.boundaryFilterData = ApexResourceHelper::resolveCollisionGroup128(params.fieldBoundaryFilterDataName ? params.fieldBoundaryFilterDataName : mAsset->mParams->fieldBoundaryFilterDataName);
#endif
		fieldSamplerDesc.boundaryFadePercentage = 0;

		fieldSamplerManager->registerFieldSampler(this, fieldSamplerDesc, mScene);
		mFieldSamplerChanged = true;
	}
}

WindFSActor::~WindFSActor()
{
}

/* Must be defined inside CPP file, since they require knowledge of asset class */
NxApexAsset* 		WindFSActor::getOwner() const
{
	return static_cast<NxApexAsset*>(mAsset);
}

NxBasicFSAsset* 	WindFSActor::getWindFSAsset() const
{
	NX_READ_ZONE();
	return mAsset;
}

void				WindFSActor::release()
{
	if (mInRelease)
	{
		return;
	}
	destroy();
} 

void WindFSActor::destroy()
{
	{
		NX_WRITE_ZONE();
		ApexActor::destroy();

		setPhysXScene(NULL);

		NiFieldSamplerManager* fieldSamplerManager = mScene->getNiFieldSamplerManager();
		if (fieldSamplerManager != 0)
		{
			fieldSamplerManager->unregisterFieldSampler(this);
		}

		if (mFieldStrengthVO)
		{
			PX_DELETE_AND_RESET(mFieldStrengthVO);
		}
		if (mFieldDirectionVO1)
		{
			PX_DELETE_AND_RESET(mFieldDirectionVO1);
		}
		if (mFieldDirectionVO2)
		{
			PX_DELETE_AND_RESET(mFieldDirectionVO2);
		}
	}
	delete this;
}

void WindFSActor::getPhysicalLodRange(PxReal& min, PxReal& max, bool& intOnly) const
{
	PX_UNUSED(min);
	PX_UNUSED(max);
	PX_UNUSED(intOnly);
	APEX_INVALID_OPERATION("not implemented");
}

physx::PxF32 WindFSActor::getActivePhysicalLod() const
{
	APEX_INVALID_OPERATION("NxExampleActor does not support this operation");
	return -1.0f;
}

void WindFSActor::forcePhysicalLod(PxReal lod)
{
	PX_UNUSED(lod);
	APEX_INVALID_OPERATION("not implemented");
}

// Called by game render thread
void WindFSActor::updateRenderResources(bool rewriteBuffers, void* userRenderData)
{
	PX_UNUSED(rewriteBuffers);
	PX_UNUSED(userRenderData);
}

// Called by game render thread
void WindFSActor::dispatchRenderResources(NxUserRenderer& renderer)
{
	PX_UNUSED(renderer);
}

bool WindFSActor::updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled)
{
	PX_UNUSED(shapeDesc);

	isEnabled = mFieldSamplerEnabled;
	if (mFieldSamplerChanged)
	{
		physx::PxVec3 instDirection = mFieldDirBasis.transform(mLocalDirVar);
		physx::PxF32 instStrength = mScale * mFieldStrength * (1.0f + mStrengthVar);

		mExecuteParams.fieldValue = instDirection * instStrength;

		shapeDesc.type = NiFieldShapeType::NONE;
		shapeDesc.worldToShape.setIdentity();
		shapeDesc.dimensions = PxVec3(0.0f);
		shapeDesc.weight = mFieldWeight;

		mFieldSamplerChanged = false;
		return true;
	}
	return false;
}

void WindFSActor::simulate(physx::PxF32 dt)
{
	if (mFieldStrengthVO != NULL)
	{
		mStrengthVar = mFieldStrengthVO->updateVariableOscillator(dt);

		mFieldSamplerChanged = true;
	}
	if (mFieldDirectionVO1 != NULL && mFieldDirectionVO2 != NULL)
	{
		PxF32 theta = mFieldDirectionVO1->updateVariableOscillator(dt);
		PxF32 phi = mFieldDirectionVO2->updateVariableOscillator(dt);

		mLocalDirVar.x = PxCos(theta);
		mLocalDirVar.y = PxSin(theta) * PxCos(phi);
		mLocalDirVar.z = PxSin(theta) * PxSin(phi);

		mFieldSamplerChanged = true;
	}
}

void WindFSActor::setFieldStrength(physx::PxF32 strength)
{
	NX_WRITE_ZONE();
	mFieldStrength = strength;
	mFieldSamplerChanged = true;
}

void WindFSActor::setFieldDirection(const physx::PxVec3& direction)
{
	NX_WRITE_ZONE();
	mFieldDirBasis.column0 = direction.getNormalized();
	BuildPlaneBasis(mFieldDirBasis.column0, mFieldDirBasis.column1, mFieldDirBasis.column2);

	mFieldSamplerChanged = true;
}

void WindFSActor::visualize()
{
#ifndef WITHOUT_DEBUG_VISUALIZE
	if ( !mEnableDebugVisualization ) return;
	NiApexRenderDebug* debugRender = mScene->mDebugRender;
	BasicFSDebugRenderParams* debugRenderParams = mScene->mBasicFSDebugRenderParams;

	if (!debugRenderParams->VISUALIZE_WIND_FS_ACTOR)
	{
		return;
	}

	if (debugRenderParams->VISUALIZE_WIND_FS_ACTOR_NAME)
	{
		char buf[128];
		buf[sizeof(buf) - 1] = 0;
		APEX_SPRINTF_S(buf, sizeof(buf) - 1, " %s %s", mAsset->getObjTypeName(), mAsset->getName());

		PxMat44 cameraFacingPose(mScene->mApexScene->getViewMatrix(0).inverseRT());
		PxVec3 textLocation = mPose.t;
		cameraFacingPose.setPosition(textLocation);

		debugRender->setCurrentTextScale(4.0f);
		debugRender->setCurrentColor(debugRender->getDebugColor(physx::DebugColors::Blue));
		debugRender->debugOrientedText(cameraFacingPose, buf);
	}
#endif
}

/******************************** CPU Version ********************************/

WindFSActorCPU::WindFSActorCPU(const WindFSActorParams& params, WindFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: WindFSActor(params, asset, list, scene)
{
}

WindFSActorCPU::~WindFSActorCPU()
{
}

void WindFSActorCPU::executeFieldSampler(const ExecuteData& data)
{
	for (PxU32 iter = 0; iter < data.count; ++iter)
	{
		PxU32 i = data.indices[iter & data.indicesMask] + (iter & ~data.indicesMask);
		physx::PxVec3* pos = (physx::PxVec3*)((physx::PxU8*)data.position + i * data.positionStride);
		data.resultField[iter] = executeWindFS(mExecuteParams, *pos);
	}
}

/******************************** GPU Version ********************************/

#if defined(APEX_CUDA_SUPPORT)


WindFSActorGPU::WindFSActorGPU(const WindFSActorParams& params, WindFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: WindFSActorCPU(params, asset, list, scene)
	, mConstMemGroup(CUDA_OBJ(fieldSamplerStorage))
{
}

WindFSActorGPU::~WindFSActorGPU()
{
}

bool WindFSActorGPU::updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled)
{
	if (WindFSActor::updateFieldSampler(shapeDesc, isEnabled))
	{
		APEX_CUDA_CONST_MEM_GROUP_SCOPE(mConstMemGroup);

		if (mParamsHandle.isNull())
		{
			mParamsHandle.alloc(_storage_);
		}
		mParamsHandle.update(_storage_, mExecuteParams);
		return true;
	}
	return false;
}


#endif

}
}
} // end namespace physx::apex

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
