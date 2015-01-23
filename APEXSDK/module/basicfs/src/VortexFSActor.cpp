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

#include "VortexFSActor.h"
#include "VortexFSAsset.h"
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

#define NUM_DEBUG_POINTS 2048

VortexFSActor::VortexFSActor(const VortexFSActorParams& params, VortexFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: BasicFSActor(scene)
	, mAsset(&asset)
{
	mFieldWeight = asset.mParams->fieldWeight;

	mPose					= params.initialPose;
	mScale					= params.initialScale;
	mAxis					= mAsset->mParams->axis;
	mBottomSphericalForce   = mAsset->mParams->bottomSphericalForce;
	mTopSphericalForce      = mAsset->mParams->topSphericalForce;
	mHeight					= mAsset->mParams->height;
	mBottomRadius			= mAsset->mParams->bottomRadius;
	mTopRadius				= mAsset->mParams->topRadius;
	mRotationalStrength		= mAsset->mParams->rotationalStrength;
	mRadialStrength			= mAsset->mParams->radialStrength;
	mLiftStrength			= mAsset->mParams->liftStrength;

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
		fieldSamplerDesc.gridSupportType = NiFieldSamplerGridSupportType::VELOCITY_PER_CELL;
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
	mDebugShapeChanged = true;
}

VortexFSActor::~VortexFSActor()
{
}

/* Must be defined inside CPP file, since they require knowledge of asset class */
NxApexAsset* 		VortexFSActor::getOwner() const
{
	return static_cast<NxApexAsset*>(mAsset);
}

NxBasicFSAsset* 	VortexFSActor::getVortexFSAsset() const
{
	NX_READ_ZONE();
	return mAsset;
}

void				VortexFSActor::release()
{
	if (mInRelease)
	{
		return;
	}
	destroy();
} 

void VortexFSActor::destroy()
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

void VortexFSActor::getPhysicalLodRange(PxReal& min, PxReal& max, bool& intOnly) const
{
	PX_UNUSED(min);
	PX_UNUSED(max);
	PX_UNUSED(intOnly);
	APEX_INVALID_OPERATION("not implemented");
}

physx::PxF32 VortexFSActor::getActivePhysicalLod() const
{
	APEX_INVALID_OPERATION("NxExampleActor does not support this operation");
	return -1.0f;
}

void VortexFSActor::forcePhysicalLod(PxReal lod)
{
	PX_UNUSED(lod);
	APEX_INVALID_OPERATION("not implemented");
}

// Called by game render thread
void VortexFSActor::updateRenderResources(bool rewriteBuffers, void* userRenderData)
{
	PX_UNUSED(rewriteBuffers);
	PX_UNUSED(userRenderData);
}

// Called by game render thread
void VortexFSActor::dispatchRenderResources(NxUserRenderer& renderer)
{
	PX_UNUSED(renderer);
}

bool VortexFSActor::updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled)
{
	isEnabled = mFieldSamplerEnabled;
	if (mFieldSamplerChanged)
	{
		mExecuteParams.bottomSphericalForce		= mBottomSphericalForce;
		mExecuteParams.topSphericalForce		= mTopSphericalForce;

		mExecuteParams.bottomRadius				= mScale * mBottomRadius;
		mExecuteParams.topRadius				= mScale * mTopRadius;
		mExecuteParams.height					= mScale * mHeight;
		mExecuteParams.rotationalStrength		= mScale * mRotationalStrength;
		mExecuteParams.radialStrength			= mScale * mRadialStrength;
		mExecuteParams.liftStrength				= mScale * mLiftStrength;

		physx::PxVec3 vecN = mPose.M * mAxis;
		vecN.normalize();
		physx::PxVec3 vecP, vecQ;
		BuildPlaneBasis(vecN, vecP, vecQ);

		mDirToWorld.M.setColumn(0, vecP);
		mDirToWorld.M.setColumn(1, vecN);
		mDirToWorld.M.setColumn(2, vecQ);
		mDirToWorld.t = mPose.t;

		mDirToWorld.getInverseRT(mExecuteParams.worldToDir);

		shapeDesc.type = NiFieldShapeType::CAPSULE;
		shapeDesc.dimensions = PxVec3(PxMax(mExecuteParams.bottomRadius, mExecuteParams.topRadius), mExecuteParams.height, 0);
		shapeDesc.worldToShape = mExecuteParams.worldToDir;
		shapeDesc.weight = mFieldWeight;

		mFieldSamplerChanged = false;
		return true;
	}
	return false;
}

void VortexFSActor::simulate(physx::PxF32 dt)
{
	PX_UNUSED(dt);
}

void VortexFSActor::setRotationalStrength(physx::PxF32 strength)
{
	NX_WRITE_ZONE();
	mRotationalStrength = strength;
	mFieldSamplerChanged = true;
}

void VortexFSActor::setRadialStrength(physx::PxF32 strength)
{
	NX_WRITE_ZONE();
	mRadialStrength = strength;
	mFieldSamplerChanged = true;
}

void VortexFSActor::setLiftStrength(physx::PxF32 strength)
{
	NX_WRITE_ZONE();
	mLiftStrength = strength;
	mFieldSamplerChanged = true;
}

void VortexFSActor::visualize()
{
#ifndef WITHOUT_DEBUG_VISUALIZE
	if ( !mEnableDebugVisualization ) return;
	NiApexRenderDebug* debugRender = mScene->mDebugRender;
	BasicFSDebugRenderParams* debugRenderParams = mScene->mBasicFSDebugRenderParams;

	if (!debugRenderParams->VISUALIZE_VORTEX_FS_ACTOR)
	{
		return;
	}

	if (debugRenderParams->VISUALIZE_VORTEX_FS_ACTOR_NAME)
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

	if (debugRenderParams->VISUALIZE_VORTEX_FS_SHAPE)
	{
		debugRender->setCurrentColor(debugRender->getDebugColor(physx::DebugColors::Blue));
		debugRender->debugOrientedCapsuleTapered(mExecuteParams.topRadius, mExecuteParams.bottomRadius, mExecuteParams.height, 2, mDirToWorld);
	}

	if (debugRenderParams->VISUALIZE_VORTEX_FS_FIELD)
	{
	}

	if (debugRenderParams->VISUALIZE_VORTEX_FS_POSE)
	{
		debugRender->debugAxes(PxMat44(mPose), 1);
	}

	if (debugRenderParams->VISUALIZE_VORTEX_FS_FIELD)
	{
		if (mDebugShapeChanged || mDebugPoints.empty())
		{
			mDebugShapeChanged = false;
			mDebugPoints.resize(NUM_DEBUG_POINTS);
			for (PxU32 i = 0; i < NUM_DEBUG_POINTS; ++i)
			{
				PxF32 r1 = mBottomRadius;
				PxF32 r2 = mTopRadius;
				PxF32 h = mHeight;
				PxF32 maxR = physx::PxMax(r1, r2);
				PxF32 rx, ry, rz;
				bool isInside = false;
				do
				{
					rx = physx::rand(-maxR, maxR);
					ry = physx::rand(-h/2 - r1, h/2 + r2);
					rz = physx::rand(-maxR, maxR);

					isInside = 2*ry <= h && -h <= 2*ry &&
						rx*rx + rz*rz <= physx::sqr(r1 + (ry / h + 0.5) * (r2-r1));
					isInside |= 2*ry < -h && rx*rx + rz*rz <= r1*r1 - (2*ry+h)*(2*ry+h)*0.25;
					isInside |= 2*ry > h && rx*rx + rz*rz <= r2*r2 - (2*ry-h)*(2*ry-h)*0.25;
				}
				while (!isInside);

				PxVec3& vec = mDebugPoints[i];

				// we need transform from local to world
				vec.x = rx;
				vec.y = ry;
				vec.z = rz;
			}
		}

		PxU32 c1 = mScene->mDebugRender->getDebugColor(physx::DebugColors::Blue);
		PxU32 c2 = mScene->mDebugRender->getDebugColor(physx::DebugColors::Red);

		for (PxU32 i = 0; i < NUM_DEBUG_POINTS; ++i)
		{
			PxVec3 localPos = mScale * mDebugPoints[i];
			PxVec3 pos = mDirToWorld * localPos;
			PxVec3 fieldVec = executeVortexFS(mExecuteParams, pos/*, totalElapsedMS*/);
			debugRender->debugGradientLine(pos, pos + fieldVec, c1, c2);
		}
	}

#endif
}

/******************************** CPU Version ********************************/

VortexFSActorCPU::VortexFSActorCPU(const VortexFSActorParams& params, VortexFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: VortexFSActor(params, asset, list, scene)
{
}

VortexFSActorCPU::~VortexFSActorCPU()
{
}

void VortexFSActorCPU::executeFieldSampler(const ExecuteData& data)
{
	for (PxU32 iter = 0; iter < data.count; ++iter)
	{
		PxU32 i = data.indices[iter & data.indicesMask] + (iter & ~data.indicesMask);
		physx::PxVec3* pos = (physx::PxVec3*)((physx::PxU8*)data.position + i * data.positionStride);
		data.resultField[iter] = executeVortexFS(mExecuteParams, *pos/*, totalElapsedMS*/);
	}
}

/******************************** GPU Version ********************************/

#if defined(APEX_CUDA_SUPPORT)


VortexFSActorGPU::VortexFSActorGPU(const VortexFSActorParams& params, VortexFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: VortexFSActorCPU(params, asset, list, scene)
	, mConstMemGroup(CUDA_OBJ(fieldSamplerStorage))
{
}

VortexFSActorGPU::~VortexFSActorGPU()
{
}

bool VortexFSActorGPU::updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled)
{
	if (VortexFSActor::updateFieldSampler(shapeDesc, isEnabled))
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
