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
#include "ForceFieldActor.h"
#include "ForceFieldAsset.h"
#include "ForceFieldScene.h"
#include "NiApexSDK.h"
#include "NiApexScene.h"
#include "NxModuleForceField.h"
#include "NxFromPx.h"

namespace physx
{
namespace apex
{
namespace forcefield
{

ForceFieldActor::ForceFieldActor(const NxForceFieldActorDesc& desc, ForceFieldAsset& asset, NxResourceList& list, ForceFieldScene& scene):
	mForceFieldScene(&scene),
	mName(desc.actorName),
	mAsset(&asset),
	mEnable(true),
	mElapsedTime(0)
{
	//not actually used
	//mScale = desc.scale * mAsset->getDefaultScale();

	list.add(*this);			// Add self to asset's list of actors
	addSelfToContext(*scene.mApexScene->getApexContext());    // Add self to ApexScene
	addSelfToContext(scene);	// Add self to FieldBoundaryScene's list of actors

	initActorParams(desc.initialPose);
	initFieldSampler(desc);
}

ForceFieldShapeType::Enum getIncludeShapeType(const ForceFieldAssetParams& assetParams)
{
	NxParameterized::Handle hEnum(assetParams);
	assetParams.getParameterHandle("includeShapeParameters.shape", hEnum);
	PX_ASSERT(hEnum.isValid());

	// assuming that enums in ForceFieldAssetParamSchema line up with ForceFieldShapeType::Enum
	physx::PxI32 shapeInt = hEnum.parameterDefinition()->enumValIndex(assetParams.includeShapeParameters.shape);
	if (-1 != shapeInt)
	{
		return (ForceFieldShapeType::Enum)shapeInt;
	}
	return ForceFieldShapeType::NONE;
}

ForceFieldCoordinateSystemType::Enum getCoordinateSystemType(const GenericForceFieldKernelParams& assetParams)
{
	NxParameterized::Handle hEnum(assetParams);
	assetParams.getParameterHandle("coordinateSystemParameters.type", hEnum);
	PX_ASSERT(hEnum.isValid());

	// assuming that enums in ForceFieldAssetParamSchema line up with ForceFieldShapeType::Enum
	physx::PxI32 shapeInt = hEnum.parameterDefinition()->enumValIndex(assetParams.coordinateSystemParameters.type);
	if (-1 != shapeInt)
	{
		return (ForceFieldCoordinateSystemType::Enum)shapeInt;
	}
	return ForceFieldCoordinateSystemType::CARTESIAN;
}

void ForceFieldActor::initActorParams(const PxMat44& initialPose)
{
	mLifetime = mAsset->mParams->lifetime;

	//ForceFieldFSKernelParams initialization
	{
		ForceFieldFSKernelParams& params = mKernelParams.getForceFieldFSKernelParams();
		params.pose = initialPose;
		params.strength = mAsset->mParams->strength;
		params.includeShape.dimensions = mAsset->mParams->includeShapeParameters.dimensions;
		params.includeShape.forceFieldToShape = mAsset->mParams->includeShapeParameters.forceFieldToShape;
		params.includeShape.type = getIncludeShapeType(*mAsset->mParams);
	}

	if (mAsset->mGenericParams)
	{
		mKernelParams.kernelType = ForceFieldKernelType::GENERIC;
		GenericForceFieldFSKernelParams& params = mKernelParams.getGenericForceFieldFSKernelParams();
		
		params.cs.torusRadius = mAsset->mGenericParams->coordinateSystemParameters.torusRadius;
		params.cs.type = getCoordinateSystemType(*mAsset->mGenericParams);
		params.constant = mAsset->mGenericParams->constant;
		params.positionMultiplier = mAsset->mGenericParams->positionMultiplier;
		params.positionTarget = mAsset->mGenericParams->positionTarget;
		params.velocityMultiplier = mAsset->mGenericParams->velocityMultiplier;
		params.velocityTarget = mAsset->mGenericParams->velocityTarget;
		params.noise = mAsset->mGenericParams->noise;
		params.falloffLinear = mAsset->mGenericParams->falloffLinear;
		params.falloffQuadratic = mAsset->mGenericParams->falloffQuadratic;
	}
	else if (mAsset->mRadialParams)
	{
		mKernelParams.kernelType = ForceFieldKernelType::RADIAL;
		RadialForceFieldFSKernelParams& params = mKernelParams.getRadialForceFieldFSKernelParams();
		
		// falloff parameters TODO: these should move to  mAsset->mRadialParams->parameters()
		params.falloffTable.multiplier = mAsset->mFalloffParams->multiplier;
		params.falloffTable.x1 = mAsset->mFalloffParams->start;
		params.falloffTable.x2 = mAsset->mFalloffParams->end;
		setRadialFalloffType(mAsset->mFalloffParams->type);

		// noise parameters TODO: these should move to  mAsset->mRadialParams->parameters()
		params.noiseParams.strength = mAsset->mNoiseParams->strength;
		params.noiseParams.spaceScale = mAsset->mNoiseParams->spaceScale;
		params.noiseParams.timeScale = mAsset->mNoiseParams->timeScale;
		params.noiseParams.octaves = mAsset->mNoiseParams->octaves;

		params.radius = mAsset->mRadialParams->radius;
		PX_ASSERT(params.radius > 0.9e-3f);
	}
	else
	{
		PX_ASSERT(0 && "Invalid geometry type for APEX turbulence source.");
		return;
	}

	//do first copy of double buffered params
	memcpy(&mKernelExecutionParams, &mKernelParams, sizeof(ForceFieldFSKernelParamsUnion));
}


void ForceFieldActor::setPhysXScene(PxScene* /*scene*/)
{
}

PxScene* ForceFieldActor::getPhysXScene() const
{
	return NULL;
}

void ForceFieldActor::getPhysicalLodRange(physx::PxF32& min, physx::PxF32& max, bool& intOnly) const
{
	NX_READ_ZONE();
	PX_UNUSED(min);
	PX_UNUSED(max);
	PX_UNUSED(intOnly);
	APEX_INVALID_OPERATION("not implemented");
}

physx::PxF32 ForceFieldActor::getActivePhysicalLod() const
{
	NX_READ_ZONE();
	APEX_INVALID_OPERATION("NxForceFieldActor does not support this operation");
	return -1.0f;
}

void ForceFieldActor::forcePhysicalLod(physx::PxF32 lod)
{
	NX_WRITE_ZONE();
	PX_UNUSED(lod);
	APEX_INVALID_OPERATION("not implemented");
}

/* Must be defined inside CPP file, since they require knowledge of asset class */
NxApexAsset* ForceFieldActor::getOwner() const
{
	NX_READ_ZONE();
	return (NxApexAsset*)mAsset;
}

NxForceFieldAsset* ForceFieldActor::getForceFieldAsset() const
{
	NX_READ_ZONE();
	return mAsset;
}

void ForceFieldActor::release()
{
	if (mInRelease)
	{
		return;
	}
	destroy();
}

void ForceFieldActor::destroy()
{
	ApexActor::destroy();
	setPhysXScene(NULL);
	releaseFieldSampler();
	delete this;
}

bool ForceFieldActor::enable()
{
	NX_WRITE_ZONE();
	if (mEnable)
	{
		return true;
	}
	mEnable = true;
	mFieldSamplerChanged = true;
	return true;
}

bool ForceFieldActor::disable()
{
	NX_WRITE_ZONE();
	if (!mEnable)
	{
		return true;
	}
	mEnable = false;
	mFieldSamplerChanged = true;
	return true;
}

physx::PxMat44 ForceFieldActor::getPose() const
{
	NX_READ_ZONE();
	const ForceFieldFSKernelParams& kernelParams = mKernelParams.getForceFieldFSKernelParams();
	return kernelParams.pose;
}

void ForceFieldActor::setPose(const physx::PxMat44& pose)
{
	NX_WRITE_ZONE();
	ForceFieldFSKernelParams& kernelParams = mKernelParams.getForceFieldFSKernelParams();
	kernelParams.pose = pose;
	mFieldSamplerChanged = true;
}

//deprecated, has no effect
void ForceFieldActor::setScale(physx::PxF32)
{
	NX_WRITE_ZONE();
}

void ForceFieldActor::setStrength(const physx::PxF32 strength)
{
	NX_WRITE_ZONE();
	ForceFieldFSKernelParams& kernelParams = mKernelParams.getForceFieldFSKernelParams();
	kernelParams.strength = strength;
	mFieldSamplerChanged = true;
}

void ForceFieldActor::setLifetime(const physx::PxF32 lifetime)
{
	NX_WRITE_ZONE();
	mLifetime = lifetime;
	mFieldSamplerChanged = true;
}

//deprecated
void ForceFieldActor::setFalloffType(const char* type)
{
	NX_WRITE_ZONE();
	if (mKernelParams.kernelType == ForceFieldKernelType::RADIAL)
	{
		setRadialFalloffType(type);
	}
}

//deprecated
void ForceFieldActor::setFalloffMultiplier(const physx::PxF32 multiplier)
{
	NX_WRITE_ZONE();
	if (mKernelParams.kernelType == ForceFieldKernelType::RADIAL)
	{
		setRadialFalloffMultiplier(multiplier);
	}
}

void ForceFieldActor::setRadialFalloffType(const char* type)
{
	NX_WRITE_ZONE();
	PX_ASSERT(mKernelParams.kernelType == ForceFieldKernelType::RADIAL);
	RadialForceFieldFSKernelParams& kernelParams = mKernelParams.getRadialForceFieldFSKernelParams();

	ForceFieldFalloffType::Enum falloffType;

	NxParameterized::Handle hEnum(mAsset->mFalloffParams);
	mAsset->mFalloffParams->getParameterHandle("type", hEnum);
	PX_ASSERT(hEnum.isValid());

	// assuming that enums in ForceFieldAssetParamSchema line up with ForceFieldFalloffType::Enum
	physx::PxI32 typeInt = hEnum.parameterDefinition()->enumValIndex(type);
	if (-1 != typeInt)
	{
		falloffType = (ForceFieldFalloffType::Enum)typeInt;
	}
	else
	{
		falloffType = ForceFieldFalloffType::NONE;
	}

	switch (falloffType)
	{
	case ForceFieldFalloffType::LINEAR:
		{
			kernelParams.falloffTable.applyStoredTable(TableName::LINEAR);
			break;
		}
	case ForceFieldFalloffType::STEEP:
		{
			kernelParams.falloffTable.applyStoredTable(TableName::STEEP);
			break;
		}
	case ForceFieldFalloffType::SCURVE:
		{
			kernelParams.falloffTable.applyStoredTable(TableName::SCURVE);
			break;
		}
	case ForceFieldFalloffType::CUSTOM:
		{
			kernelParams.falloffTable.applyStoredTable(TableName::CUSTOM);
			break;
		}
	case ForceFieldFalloffType::NONE:
		{
			kernelParams.falloffTable.applyStoredTable(TableName::CUSTOM);	// all-1 stored table
		}
	}

	kernelParams.falloffTable.buildTable();
	mFieldSamplerChanged = true;
}

void ForceFieldActor::setRadialFalloffMultiplier(const physx::PxF32 multiplier)
{
	PX_ASSERT(mKernelParams.kernelType == ForceFieldKernelType::RADIAL);
	RadialForceFieldFSKernelParams& kernelParams = mKernelParams.getRadialForceFieldFSKernelParams();

	kernelParams.falloffTable.multiplier = multiplier;
	kernelParams.falloffTable.buildTable();
	mFieldSamplerChanged = true;
}

void ForceFieldActor::updateForceField(physx::PxF32 dt)
{
	mElapsedTime += dt;

	if (mLifetime > 0.0f && mElapsedTime > mLifetime)
	{
		disable();
	}
}

// Called by ForceFieldScene::fetchResults()
void ForceFieldActor::updatePoseAndBounds()
{
}

void ForceFieldActor::visualize()
{
#ifndef WITHOUT_DEBUG_VISUALIZE
	if ( !mEnableDebugVisualization ) return;
	if (mEnable)
	{
		visualizeIncludeShape();
	}
#endif
}

void ForceFieldActor::visualizeIncludeShape()
{
#ifndef WITHOUT_DEBUG_VISUALIZE
	if (mEnable)
	{
		mForceFieldScene->mRenderDebug->pushRenderState();
		mForceFieldScene->mRenderDebug->setCurrentColor(0xFF0000);

		ForceFieldFSKernelParams& kernelParams = mKernelParams.getForceFieldFSKernelParams();
		physx::PxMat44 debugPose = kernelParams.includeShape.forceFieldToShape * kernelParams.pose;

		switch (kernelParams.includeShape.type)
		{
		case ForceFieldShapeType::SPHERE:
			{
				mForceFieldScene->mRenderDebug->debugOrientedSphere(kernelParams.includeShape.dimensions.x, 2, debugPose);
				break;
			}
		case ForceFieldShapeType::CAPSULE:
			{
				mForceFieldScene->mRenderDebug->debugOrientedCapsule(kernelParams.includeShape.dimensions.x, kernelParams.includeShape.dimensions.y * 2, 2, debugPose);
				break;
			}
		case ForceFieldShapeType::CYLINDER:
			{
				mForceFieldScene->mRenderDebug->debugOrientedCylinder(kernelParams.includeShape.dimensions.x, kernelParams.includeShape.dimensions.y * 2, 2, true, debugPose);
				break;
			}
		case ForceFieldShapeType::CONE:
			{
				// using a cylinder to approximate a cone for debug rendering
				// TODO: draw a cone using lines
				mForceFieldScene->mRenderDebug->debugOrientedCylinder(kernelParams.includeShape.dimensions.x, kernelParams.includeShape.dimensions.y * 2, 2, true, debugPose);
				break;
			}
		case ForceFieldShapeType::BOX:
			{
				mForceFieldScene->mRenderDebug->debugOrientedBound(kernelParams.includeShape.dimensions * 2, debugPose);
				break;
			}
		default:
			{
			}
		}

		mForceFieldScene->mRenderDebug->popRenderState();
	}
#endif
}

void ForceFieldActor::visualizeForces()
{
#ifndef WITHOUT_DEBUG_VISUALIZE
	if (mEnable)
	{
	}
#endif
}

}
}
} // end namespace physx::apex

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
