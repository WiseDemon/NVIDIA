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

#include "JetFSActor.h"
#include "JetFSAsset.h"
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

#include "ReadCheck.h"
#include "WriteCheck.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

#define NUM_DEBUG_POINTS 2048


JetFSActor::JetFSActor(const JetFSActorParams& params, JetFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: BasicFSActor(scene)
	, mAsset(&asset)
	, mFieldDirectionVO1(NULL)
	, mFieldDirectionVO2(NULL)
	, mFieldStrengthVO(NULL)
{
	NX_WRITE_ZONE();
	mFieldWeight = asset.mParams->fieldWeight;

	mPose = params.initialPose;
	mScale = params.initialScale * asset.mParams->defaultScale;

	mFieldDirection = mAsset->mParams->fieldDirection.getNormalized();
	mFieldStrength = mAsset->mParams->fieldStrength;

	mStrengthVar = 0.0f;
	mLocalDirVar = PxVec3(0, 1, 0);

	mExecuteParams.noiseTimeScale = mAsset->mParams->noiseTimeScale;
	mExecuteParams.noiseOctaves = mAsset->mParams->noiseOctaves;

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
}

JetFSActor::~JetFSActor()
{
}

/* Must be defined inside CPP file, since they require knowledge of asset class */
NxApexAsset* 		JetFSActor::getOwner() const
{
	NX_READ_ZONE();
	return static_cast<NxApexAsset*>(mAsset);
}

NxBasicFSAsset* 	JetFSActor::getJetFSAsset() const
{
	NX_READ_ZONE();
	return mAsset;
}

void				JetFSActor::release()
{
	if (mInRelease)
	{
		return;
	}
	destroy();
} 

void JetFSActor::destroy()
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

void JetFSActor::getPhysicalLodRange(PxReal& min, PxReal& max, bool& intOnly) const
{
	NX_READ_ZONE();
	PX_UNUSED(min);
	PX_UNUSED(max);
	PX_UNUSED(intOnly);
	APEX_INVALID_OPERATION("not implemented");
}

physx::PxF32 JetFSActor::getActivePhysicalLod() const
{
	NX_READ_ZONE();
	APEX_INVALID_OPERATION("NxExampleActor does not support this operation");
	return -1.0f;
}

void JetFSActor::forcePhysicalLod(PxReal lod)
{
	NX_WRITE_ZONE();
	PX_UNUSED(lod);
	APEX_INVALID_OPERATION("not implemented");
}

// Called by game render thread
void JetFSActor::updateRenderResources(bool rewriteBuffers, void* userRenderData)
{
	NX_WRITE_ZONE();
	PX_UNUSED(rewriteBuffers);
	PX_UNUSED(userRenderData);
}

// Called by game render thread
void JetFSActor::dispatchRenderResources(NxUserRenderer& renderer)
{
	NX_WRITE_ZONE();
	PX_UNUSED(renderer);
}

bool JetFSActor::updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled)
{
	NX_WRITE_ZONE();
	isEnabled = mFieldSamplerEnabled;
	if (mFieldSamplerChanged)
	{
		mExecuteParams.nearRadius = mAsset->mParams->nearRadius * mScale;
		mExecuteParams.pivotRadius = mAsset->mParams->pivotRadius * mScale;
		mExecuteParams.farRadius = mAsset->mParams->farRadius * mScale;
		mExecuteParams.directionalStretch = mAsset->mParams->directionalStretch;
		mExecuteParams.averageStartDistance = mAsset->mParams->averageStartDistance * mScale;
		mExecuteParams.averageEndDistance = mAsset->mParams->averageEndDistance * mScale;

		mExecuteParams.pivotRatio = (mExecuteParams.farRadius - mExecuteParams.pivotRadius) / (mExecuteParams.pivotRadius - mExecuteParams.nearRadius);


		physx::PxVec3 vecN = mPose.M * mFieldDirection;
		vecN.normalize();
		physx::PxVec3 vecP, vecQ;
		BuildPlaneBasis(vecN, vecP, vecQ);

		mDirToWorld.M.setColumn(0, vecP);
		mDirToWorld.M.setColumn(1, vecN);
		mDirToWorld.M.setColumn(2, vecQ);
		mDirToWorld.t = mPose.t;

		mDirToWorld.getInverseRT(mExecuteParams.worldToDir);


		vecN = mDirToWorld.M * mLocalDirVar;
		BuildPlaneBasis(vecN, vecP, vecQ);

		PxMat34Legacy instDirToWorld;
		instDirToWorld.M.setColumn(0, vecP);
		instDirToWorld.M.setColumn(1, vecN);
		instDirToWorld.M.setColumn(2, vecQ);
		instDirToWorld.t = mPose.t;
		instDirToWorld.getInverseRT(mExecuteParams.worldToInstDir);

		mExecuteParams.strength = mFieldStrength * mScale;
		mExecuteParams.instStrength = mExecuteParams.strength * (1.0f + mStrengthVar);

		shapeDesc.type = NiFieldShapeType::CAPSULE;
		shapeDesc.dimensions = PxVec3(mExecuteParams.farRadius, mExecuteParams.farRadius * mExecuteParams.directionalStretch, 0);
		shapeDesc.worldToShape = mExecuteParams.worldToDir;
		shapeDesc.weight = mFieldWeight;

		PxF32 gridShapeRadius = mAsset->mParams->gridShapeRadius * mScale;
		PxF32 gridShapeHeight = mAsset->mParams->gridShapeHeight * mScale;

		mExecuteParams.gridIncludeShape.type = NiFieldShapeType::CAPSULE;
		mExecuteParams.gridIncludeShape.dimensions = PxVec3(gridShapeRadius, gridShapeHeight, 0);
		mExecuteParams.gridIncludeShape.worldToShape = mExecuteParams.worldToDir;
		mExecuteParams.gridIncludeShape.weight = mFieldWeight;
		mExecuteParams.gridIncludeShape.fade = mAsset->mParams->gridBoundaryFadePercentage;

		mExecuteParams.noiseStrength = mAsset->mParams->noisePercentage * mExecuteParams.strength;
		mExecuteParams.noiseSpaceScale = mAsset->mParams->noiseSpaceScale / mScale;

		mFieldSamplerChanged = false;
		return true;
	}
	return false;
}

void JetFSActor::simulate(physx::PxF32 dt)
{
	NX_WRITE_ZONE();
	if (mFieldStrengthVO != NULL)
	{
		mStrengthVar = mFieldStrengthVO->updateVariableOscillator(dt);

		mFieldSamplerChanged = true;
	}
	if (mFieldDirectionVO1 != NULL && mFieldDirectionVO2 != NULL)
	{
		PxF32 theta = mFieldDirectionVO1->updateVariableOscillator(dt);
		PxF32 phi = mFieldDirectionVO2->updateVariableOscillator(dt);

		mLocalDirVar.x = PxCos(phi) * PxSin(theta);
		mLocalDirVar.y = PxCos(theta);
		mLocalDirVar.z = PxSin(phi) * PxSin(theta);

		mFieldSamplerChanged = true;
	}
}

void JetFSActor::setFieldStrength(physx::PxF32 strength)
{
	NX_WRITE_ZONE();
	mFieldStrength = strength;
	mFieldSamplerChanged = true;
}

void JetFSActor::setFieldDirection(const physx::PxVec3& direction)
{
	NX_WRITE_ZONE();
	mFieldDirection = direction.getNormalized();
	mFieldSamplerChanged = true;
}

void JetFSActor::visualize()
{
	NX_WRITE_ZONE();
#ifndef WITHOUT_DEBUG_VISUALIZE
	if ( !mEnableDebugVisualization ) return;
	NiApexRenderDebug* debugRender = mScene->mDebugRender;
	BasicFSDebugRenderParams* debugRenderParams = mScene->mBasicFSDebugRenderParams;

	if (!debugRenderParams->VISUALIZE_JET_FS_ACTOR)
	{
		return;
	}

	if (debugRenderParams->VISUALIZE_JET_FS_ACTOR_NAME)
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

	if (debugRenderParams->VISUALIZE_JET_FS_SHAPE)
	{
		debugRender->setCurrentColor(debugRender->getDebugColor(physx::DebugColors::Blue));
		debugRender->debugOrientedCapsule(mExecuteParams.farRadius, mExecuteParams.farRadius * mExecuteParams.directionalStretch, 2, mDirToWorld);

		debugRender->setCurrentColor(debugRender->getDebugColor(physx::DebugColors::DarkBlue));
		debugRender->debugOrientedCapsule(mExecuteParams.gridIncludeShape.dimensions.x, mExecuteParams.gridIncludeShape.dimensions.y, 2, mDirToWorld);


		//draw torus
		const PxU32 NUM_PHI_SLICES = 16;
		const PxU32 NUM_THETA_SLICES = 16;

		const PxF32 torusRadius = mExecuteParams.farRadius / 2;

		PxF32 cosPhiLast = 1;
		PxF32 sinPhiLast = 0;
		for (PxU32 i = 1; i <= NUM_PHI_SLICES; ++i)
		{
			PxF32 phi = (i * PxTwoPi / NUM_PHI_SLICES);
			PxF32 cosPhi = PxCos(phi);
			PxF32 sinPhi = PxSin(phi);

			debugRender->debugLine(
			    mDirToWorld * PxVec3(cosPhiLast * mExecuteParams.pivotRadius, 0, sinPhiLast * mExecuteParams.pivotRadius),
			    mDirToWorld * PxVec3(cosPhi * mExecuteParams.pivotRadius, 0, sinPhi * mExecuteParams.pivotRadius));

			debugRender->debugLine(
			    mDirToWorld * PxVec3(cosPhiLast * mExecuteParams.nearRadius, 0, sinPhiLast * mExecuteParams.nearRadius),
			    mDirToWorld * PxVec3(cosPhi * mExecuteParams.nearRadius, 0, sinPhi * mExecuteParams.nearRadius));

			PxF32 cosThetaLast = 1;
			PxF32 sinThetaLast = 0;
			for (PxU32 j = 1; j <= NUM_THETA_SLICES; ++j)
			{
				PxF32 theta = (j * PxTwoPi / NUM_THETA_SLICES);
				PxF32 cosTheta = PxCos(theta);
				PxF32 sinTheta = PxSin(theta);

				PxF32 d = torusRadius * (1 + cosTheta);
				PxF32 h = torusRadius * sinTheta * mExecuteParams.directionalStretch;

				PxF32 dLast = torusRadius * (1 + cosThetaLast);
				PxF32 hLast = torusRadius * sinThetaLast * mExecuteParams.directionalStretch;

				debugRender->debugLine(
				    mDirToWorld * PxVec3(cosPhi * dLast, hLast, sinPhi * dLast),
				    mDirToWorld * PxVec3(cosPhi * d, h, sinPhi * d));

				debugRender->debugLine(
				    mDirToWorld * PxVec3(cosPhiLast * d, h, sinPhiLast * d),
				    mDirToWorld * PxVec3(cosPhi * d, h, sinPhi * d));

				debugRender->debugLine(
				    mDirToWorld * PxVec3(cosPhiLast * dLast, hLast, sinPhiLast * dLast),
				    mDirToWorld * PxVec3(cosPhi * dLast, hLast, sinPhi * dLast));

				cosThetaLast = cosTheta;
				sinThetaLast = sinTheta;
			}

			cosPhiLast = cosPhi;
			sinPhiLast = sinPhi;
		}
	}

	const PxF32 spreadDistance = 0.5f * mExecuteParams.farRadius * mExecuteParams.directionalStretch;

	/*
		const PxU32 CircleDivCount = 8;
		for (PxF32 angle = 0; angle < PxTwoPi; angle += PxTwoPi / CircleDivCount)
		{
			PxF32 c1 = cos(angle);
			PxF32 s1 = sin(angle);

			//debugRender->setCurrentColor( debugRender->getDebugColor(physx::DebugColors::LightBlue) );
			debugRender->debugOrientedLine(
				PxVec3(innerRadius * c1, 0, innerRadius * s1),
				PxVec3(outerRadius * c1, -spreadDistance, outerRadius * s1),
				mLocalToWorld);
			debugRender->debugOrientedLine(
				PxVec3(innerRadius * c1, 0, innerRadius * s1),
				PxVec3(outerRadius * c1, +spreadDistance, outerRadius * s1),
				mLocalToWorld);
		}
	*/
	if (debugRenderParams->VISUALIZE_JET_FS_FIELD)
	{
		PxF32 fieldScale = debugRenderParams->JET_FS_FIELD_SCALE;

		PxVec3 rayBeg0 = mPose.t;
		PxVec3 rayEnd0 = rayBeg0 + (mFieldStrength * fieldScale) * mFieldDirection;
		debugRender->setCurrentColor(debugRender->getDebugColor(physx::DebugColors::DarkBlue));
		debugRender->debugRay(rayBeg0, rayEnd0);


		//PxVec3 rayBeg = mPose.t;
		//PxVec3 rayEnd = rayBeg + (mExecuteParams.strength * fieldScale) * mExecuteParams.direction;
		//debugRender->setCurrentColor( debugRender->getDebugColor(physx::DebugColors::Blue) );
		//debugRender->debugRay(rayBeg, rayEnd);
	}
	if (debugRenderParams->VISUALIZE_JET_FS_POSE)
	{
		debugRender->debugAxes(PxMat44(mPose), 1);
	}

	if (debugRenderParams->VISUALIZE_JET_FS_FIELD)
	{
		if (mDebugPoints.empty())
		{
			mDebugPoints.resize(NUM_DEBUG_POINTS);

			for (PxU32 i = 0; i < NUM_DEBUG_POINTS; ++i)
			{
				PxF32 rx, ry;
				do
				{
					rx = physx::rand(-1.0f, +1.0f);
					ry = physx::rand(-1.0f, +1.0f);
				}
				while (rx * rx + ry * ry > 1.0f);

				PxVec3& vec = mDebugPoints[i];

				vec.x = rx;
				vec.y = physx::rand(-1.0f, +1.0f);
				vec.z = ry;
			}
		}

		PxU32 c1 = mScene->mDebugRender->getDebugColor(physx::DebugColors::Blue);
		PxU32 c2 = mScene->mDebugRender->getDebugColor(physx::DebugColors::Red);

		PxU32 totalElapsedMS = mScene->getApexScene().getTotalElapsedMS();

		for (PxU32 i = 0; i < NUM_DEBUG_POINTS; ++i)
		{
			PxVec3 localPos = mDebugPoints[i];
			localPos.x *= mExecuteParams.farRadius;
			localPos.y *= spreadDistance;
			localPos.z *= mExecuteParams.farRadius;

			PxVec3 pos = mDirToWorld * localPos;
			PxVec3 fieldVec = executeJetFS(mExecuteParams, pos, totalElapsedMS);
			fieldVec *= debugRenderParams->JET_FS_FIELD_SCALE;
			debugRender->debugGradientLine(pos, pos + fieldVec, c1, c2);
		}
	}
#endif
}

/******************************** CPU Version ********************************/

JetFSActorCPU::JetFSActorCPU(const JetFSActorParams& params, JetFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: JetFSActor(params, asset, list, scene)
{
}

JetFSActorCPU::~JetFSActorCPU()
{
}

void JetFSActorCPU::executeFieldSampler(const ExecuteData& data)
{
	NX_WRITE_ZONE();
	PxU32 totalElapsedMS = mScene->getApexScene().getTotalElapsedMS();
	for (PxU32 iter = 0; iter < data.count; ++iter)
	{
		PxU32 i = data.indices[iter & data.indicesMask] + (iter & ~data.indicesMask);
		physx::PxVec3* pos = (physx::PxVec3*)((physx::PxU8*)data.position + i * data.positionStride);
		data.resultField[iter] = executeJetFS(mExecuteParams, *pos, totalElapsedMS);
	}
}

/******************************** GPU Version ********************************/

#if defined(APEX_CUDA_SUPPORT)


JetFSActorGPU::JetFSActorGPU(const JetFSActorParams& params, JetFSAsset& asset, NxResourceList& list, BasicFSScene& scene)
	: JetFSActorCPU(params, asset, list, scene)
	, mConstMemGroup(CUDA_OBJ(fieldSamplerStorage))
{
}

JetFSActorGPU::~JetFSActorGPU()
{
}

bool JetFSActorGPU::updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled)
{
	NX_WRITE_ZONE();
	if (JetFSActor::updateFieldSampler(shapeDesc, isEnabled))
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
