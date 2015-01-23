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

#include "NoiseFSActor.h"
#include "NoiseFSAsset.h"
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

NoiseFSActor::NoiseFSActor(const NoiseFSActorParams& params, NoiseFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: BasicFSActor(scene)
	, mAsset(&asset)
{
	mFieldWeight = asset.mParams->fieldWeight;

	mPose = params.initialPose;
	mScale = params.initialScale * asset.mParams->defaultScale;

	mExecuteParams.useLocalSpace = mAsset->mParams->useLocalSpace;

	mExecuteParams.noiseTimeFreq = 1.0f / mAsset->mParams->noiseTimePeriod;
	mExecuteParams.noiseOctaves = mAsset->mParams->noiseOctaves;
	mExecuteParams.noiseStrengthOctaveMultiplier = mAsset->mParams->noiseStrengthOctaveMultiplier;
	mExecuteParams.noiseSpaceFreqOctaveMultiplier = PxVec3(1.0f / mAsset->mParams->noiseSpacePeriodOctaveMultiplier.x, 1.0f / mAsset->mParams->noiseSpacePeriodOctaveMultiplier.y, 1.0f / mAsset->mParams->noiseSpacePeriodOctaveMultiplier.z);
	mExecuteParams.noiseTimeFreqOctaveMultiplier = 1.0f / mAsset->mParams->noiseTimePeriodOctaveMultiplier;

	if (strcmp(mAsset->mParams->noiseType, "CURL") == 0)
	{
		mExecuteParams.noiseType = NoiseType::CURL;
	}
	else
	{
		mExecuteParams.noiseType = NoiseType::SIMPLEX;
	}
	mExecuteParams.noiseSeed = mAsset->mParams->noiseSeed;

	list.add(*this);			// Add self to asset's list of actors
	addSelfToContext(*scene.getApexScene().getApexContext());    // Add self to ApexScene
	addSelfToContext(scene);	// Add self to BasicFSScene's list of actors

	NiFieldSamplerManager* fieldSamplerManager = mScene->getNiFieldSamplerManager();
	if (fieldSamplerManager != 0)
	{
		NiFieldSamplerDesc fieldSamplerDesc;

		fieldSamplerDesc.gridSupportType = NiFieldSamplerGridSupportType::VELOCITY_PER_CELL;
		if (strcmp(mAsset->mParams->fieldType, "FORCE") == 0)
		{
			fieldSamplerDesc.type = NiFieldSamplerType::FORCE;
			fieldSamplerDesc.gridSupportType = NiFieldSamplerGridSupportType::NONE;
		}
		else if (strcmp(mAsset->mParams->fieldType, "VELOCITY_DRAG") == 0)
		{
			fieldSamplerDesc.type = NiFieldSamplerType::VELOCITY_DRAG;
			fieldSamplerDesc.dragCoeff = mAsset->mParams->fieldDragCoeff;
		}
		else
		{
			fieldSamplerDesc.type = NiFieldSamplerType::VELOCITY_DIRECT;
		}
#if NX_SDK_VERSION_MAJOR == 2
		fieldSamplerDesc.samplerFilterData = ApexResourceHelper::resolveCollisionGroup64(params.fieldSamplerFilterDataName ? params.fieldSamplerFilterDataName : mAsset->mParams->fieldSamplerFilterDataName);
		fieldSamplerDesc.boundaryFilterData = ApexResourceHelper::resolveCollisionGroup64(params.fieldBoundaryFilterDataName ? params.fieldBoundaryFilterDataName : mAsset->mParams->fieldBoundaryFilterDataName);
#else
		fieldSamplerDesc.samplerFilterData = ApexResourceHelper::resolveCollisionGroup128(params.fieldSamplerFilterDataName ? params.fieldSamplerFilterDataName : mAsset->mParams->fieldSamplerFilterDataName);
		fieldSamplerDesc.boundaryFilterData = ApexResourceHelper::resolveCollisionGroup128(params.fieldBoundaryFilterDataName ? params.fieldBoundaryFilterDataName : mAsset->mParams->fieldBoundaryFilterDataName);
#endif
		fieldSamplerDesc.boundaryFadePercentage = mAsset->mParams->boundaryFadePercentage;

		fieldSamplerManager->registerFieldSampler(this, fieldSamplerDesc, mScene);
		mFieldSamplerChanged = true;
	}
}

NoiseFSActor::~NoiseFSActor()
{
}

/* Must be defined inside CPP file, since they require knowledge of asset class */
NxApexAsset* 		NoiseFSActor::getOwner() const
{
	return static_cast<NxApexAsset*>(mAsset);
}

NxBasicFSAsset* 	NoiseFSActor::getNoiseFSAsset() const
{
	NX_READ_ZONE();
	return mAsset;
}

void				NoiseFSActor::release()
{
	if (mInRelease)
	{
		return;
	}
	destroy();
} 

void NoiseFSActor::destroy()
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
	}

	delete this;
}

void NoiseFSActor::getPhysicalLodRange(PxReal& min, PxReal& max, bool& intOnly) const
{
	PX_UNUSED(min);
	PX_UNUSED(max);
	PX_UNUSED(intOnly);
	APEX_INVALID_OPERATION("not implemented");
}

physx::PxF32 NoiseFSActor::getActivePhysicalLod() const
{
	APEX_INVALID_OPERATION("NxExampleActor does not support this operation");
	return -1.0f;
}

void NoiseFSActor::forcePhysicalLod(PxReal lod)
{
	PX_UNUSED(lod);
	APEX_INVALID_OPERATION("not implemented");
}

// Called by game render thread
void NoiseFSActor::updateRenderResources(bool rewriteBuffers, void* userRenderData)
{
	PX_UNUSED(rewriteBuffers);
	PX_UNUSED(userRenderData);
}

// Called by game render thread
void NoiseFSActor::dispatchRenderResources(NxUserRenderer& renderer)
{
	PX_UNUSED(renderer);
}

bool NoiseFSActor::updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled)
{
	PX_UNUSED(shapeDesc);

	isEnabled = mFieldSamplerEnabled;
	if (mFieldSamplerChanged)
	{
		mExecuteParams.worldToShape = mPose.getInverseRT();

		PxVec3 noiseSpacePeriod = mAsset->mParams->noiseSpacePeriod * mScale;
		mExecuteParams.noiseSpaceFreq = PxVec3(1.0f / noiseSpacePeriod.x, 1.0f / noiseSpacePeriod.y, 1.0f / noiseSpacePeriod.z);
		mExecuteParams.noiseStrength = mAsset->mParams->noiseStrength * mScale;

		shapeDesc.type = NiFieldShapeType::BOX;
		shapeDesc.worldToShape = mExecuteParams.worldToShape;
		shapeDesc.dimensions = mAsset->mParams->boundarySize * (mScale * 0.5f);
		shapeDesc.weight = mFieldWeight;

		mFieldSamplerChanged = false;
		return true;
	}
	return false;
}

void NoiseFSActor::simulate(physx::PxF32 )
{
}

void NoiseFSActor::setNoiseStrength(physx::PxF32 strength)
{
	NX_WRITE_ZONE();
	mExecuteParams.noiseStrength = strength;
	mFieldSamplerChanged = true;
}

void NoiseFSActor::visualize()
{
#ifndef WITHOUT_DEBUG_VISUALIZE
	if ( !mEnableDebugVisualization ) return;
	NiApexRenderDebug* debugRender = mScene->mDebugRender;
	BasicFSDebugRenderParams* debugRenderParams = mScene->mBasicFSDebugRenderParams;

	if (!debugRenderParams->VISUALIZE_NOISE_FS_ACTOR)
	{
		return;
	}

	if (debugRenderParams->VISUALIZE_NOISE_FS_ACTOR_NAME)
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

	if (debugRenderParams->VISUALIZE_NOISE_FS_SHAPE)
	{
		debugRender->setCurrentColor(debugRender->getDebugColor(physx::DebugColors::Blue));

		PxVec3 shapeSides = mScale * mAsset->mParams->boundarySize;
		debugRender->debugOrientedBound( shapeSides, mPose );
	}
	if (debugRenderParams->VISUALIZE_NOISE_FS_POSE)
	{
		debugRender->debugAxes(PxMat44(mPose), 1);
	}
#endif
}

/******************************** CPU Version ********************************/

NoiseFSActorCPU::NoiseFSActorCPU(const NoiseFSActorParams& params, NoiseFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: NoiseFSActor(params, asset, list, scene)
{
}

NoiseFSActorCPU::~NoiseFSActorCPU()
{
}

void NoiseFSActorCPU::executeFieldSampler(const ExecuteData& data)
{
	PxU32 totalElapsedMS = mScene->getApexScene().getTotalElapsedMS();
	for (PxU32 iter = 0; iter < data.count; ++iter)
	{
		PxU32 i = data.indices[iter & data.indicesMask] + (iter & ~data.indicesMask);
		physx::PxVec3* pos = (physx::PxVec3*)((physx::PxU8*)data.position + i * data.positionStride);
		data.resultField[iter] = executeNoiseFS(mExecuteParams, *pos, totalElapsedMS);
	}
}

/******************************** GPU Version ********************************/

#if defined(APEX_CUDA_SUPPORT)


NoiseFSActorGPU::NoiseFSActorGPU(const NoiseFSActorParams& params, NoiseFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: NoiseFSActorCPU(params, asset, list, scene)
	, mConstMemGroup(CUDA_OBJ(fieldSamplerStorage))
{
}

NoiseFSActorGPU::~NoiseFSActorGPU()
{
}

bool NoiseFSActorGPU::updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled)
{
	if (NoiseFSActor::updateFieldSampler(shapeDesc, isEnabled))
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
