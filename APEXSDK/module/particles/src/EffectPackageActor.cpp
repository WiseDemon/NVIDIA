/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "EffectPackageActor.h"
#include "FloatMath.h"
#include "ParticlesScene.h"
#include "NiApexScene.h"
#include "NxApexEmitterActor.h"
#include "NxApexEmitterAsset.h"
#include "NxTurbulenceFSActor.h"
#include "NxAttractorFSActor.h"
#include "NxJetFSActor.h"
#include "NxWindFSActor.h"
#include "NxNoiseFSActor.h"
#include "NxVortexFSActor.h"
#include "NxParamUtils.h"
#include "NxImpactEmitterActor.h"
#include "NxGroundEmitterActor.h"
#include "NxTurbulenceFSAsset.h"
#include "NxBasicIosAsset.h"
#include "NxBasicFSAsset.h"
#include "PxRenderDebug.h"
#include "NiApexScene.h"
#include "NxForceFieldAsset.h"
#include "NxForceFieldActor.h"
#include "NxHeatSourceAsset.h"
#include "NxHeatSourceActor.h"
#include "HeatSourceAssetParams.h"

#include "ApexEmitterActorParameters.h"
#include "ApexEmitterAssetParameters.h"

#include "NxSubstanceSourceAsset.h"
#include "NxSubstanceSourceActor.h"
#include "NxSubstanceSourceAsset.h"
#include "SubstanceSourceAssetParams.h"

#include "NxVelocitySourceAsset.h"
#include "NxVelocitySourceActor.h"
#include "VelocitySourceAssetParams.h"

#include "NxFlameEmitterAsset.h"
#include "NxFlameEmitterActor.h"
#include "FlameEmitterAssetParams.h"

#include "TurbulenceFSAssetParams.h"

#if NX_SDK_VERSION_MAJOR == 3
#include "PxPhysics.h"
#include "PxScene.h"
#include "ApexResourceHelper.h"
#endif

#pragma warning(disable:4100)

namespace physx
{
namespace apex
{
namespace particles
{

static PxTransform _getPose(physx::PxF32 x, physx::PxF32 y, physx::PxF32 z, physx::PxF32 rotX, physx::PxF32 rotY, physx::PxF32 rotZ)
{
	PxTransform ret;
	ret.p = PxVec3(x, y, z);
	fm_eulerToQuat(rotX * FM_DEG_TO_RAD, rotY * FM_DEG_TO_RAD, rotZ * FM_DEG_TO_RAD, &ret.q.x);
	return ret;
}

void _getRot(const PxQuat& q, PxF32& rotX, PxF32& rotY, PxF32& rotZ)
{
	fm_quatToEuler((PxF32*)&q, rotX, rotY, rotZ);
	rotX *= FM_RAD_TO_DEG;
	rotY *= FM_RAD_TO_DEG;
	rotZ *= FM_RAD_TO_DEG;
}

static PxF32 ranf(void)
{
	PxU32 r = (PxU32)::rand();
	r &= 0x7FFF;
	return (PxF32)r * (1.0f / 32768.0f);
}

static PxF32 ranf(PxF32 min, PxF32 max)
{
	return ranf() * (max - min) + min;
}

EffectPackageActor::EffectPackageActor(NxEffectPackageAsset* apexAsset,
                                       const EffectPackageAssetParams* assetParams,
                                       const EffectPackageActorParams* actorParams,
                                       physx::apex::NxApexSDK& sdk,
                                       physx::apex::NxApexScene& scene,
                                       ParticlesScene& dynamicSystemScene,
									   NxModuleTurbulenceFS* moduleTurbulenceFS)
{
	mRigidBodyChange = false;
	mRenderVolume = NULL;
	mEmitterValidateCallback = NULL;
	mAlive = true;
	mSimTime = 0;
	mCurrentLifeTime = 0;
	mFadeIn = false;
	mFadeInTime = 0;
	mFadeInDuration = 0;
	mFadeOut = false;
	mFadeOutTime = 0;
	mFadeOutDuration = 0;
	mFirstFrame = true;
	mData = assetParams;
	mAsset = apexAsset;
	mScene = &scene;
	mModuleTurbulenceFS = moduleTurbulenceFS;
	mEnabled = actorParams->Enabled;
	mVisible = false;
	mEverVisible = false;
	mOffScreenTime = 0;
	mVisState = VS_ON_SCREEN;
	mFadeTime = 0;
	mNotVisibleTime = 0;
	mPose = actorParams->InitialPose;
	mObjectScale = actorParams->objectScale;
	mEffectPath = PX_NEW(EffectPath);
	{
		RigidBodyEffectNS::EffectPath_Type *path = (RigidBodyEffectNS::EffectPath_Type *)&mData->Path;
		if ( !mEffectPath->init(*path))
		{
			delete mEffectPath;
			mEffectPath = NULL;
		}
	}

	for (physx::PxI32 i = 0; i < mData->Effects.arraySizes[0]; i++)
	{
		NxParameterized::Interface* iface = mData->Effects.buf[i];
		PX_ASSERT(iface);
		if (iface)
		{
			EffectType type = getEffectType(iface);
			switch (type)
			{
			case ET_EMITTER:
			{
				EmitterEffect* ed = static_cast< EmitterEffect*>(iface);
				if (ed->Emitter)
				{
					EffectEmitter* ee = PX_NEW(EffectEmitter)(ed->Emitter->name(), ed, sdk, scene, dynamicSystemScene, mPose, mEnabled);
					ee->setCurrentScale(mObjectScale,mEffectPath);
					mEffects.pushBack(static_cast< EffectData*>(ee));
				}
			}
			break;
			case ET_HEAT_SOURCE:
			{
				HeatSourceEffect* ed = static_cast< HeatSourceEffect*>(iface);
				EffectHeatSource* ee = PX_NEW(EffectHeatSource)(ed->HeatSource->name(), ed, sdk, scene, dynamicSystemScene, moduleTurbulenceFS, mPose, mEnabled);
				ee->setCurrentScale(mObjectScale,mEffectPath);
				mEffects.pushBack(static_cast< EffectData*>(ee));
			}
			break;
			case ET_SUBSTANCE_SOURCE:
				{
					SubstanceSourceEffect* ed = static_cast< SubstanceSourceEffect*>(iface);
					EffectSubstanceSource* ee = PX_NEW(EffectSubstanceSource)(ed->SubstanceSource->name(), ed, sdk, scene, dynamicSystemScene, moduleTurbulenceFS, mPose, mEnabled);
					ee->setCurrentScale(mObjectScale,mEffectPath);
					mEffects.pushBack(static_cast< EffectData*>(ee));
				}
				break;
			case ET_VELOCITY_SOURCE:
				{
					VelocitySourceEffect* ed = static_cast< VelocitySourceEffect*>(iface);
					EffectVelocitySource* ee = PX_NEW(EffectVelocitySource)(ed->VelocitySource->name(), ed, sdk, scene, dynamicSystemScene, moduleTurbulenceFS, mPose, mEnabled);
					ee->setCurrentScale(mObjectScale,mEffectPath);
					mEffects.pushBack(static_cast< EffectData*>(ee));
				}
				break;
			case ET_FLAME_EMITTER:
				{
					FlameEmitterEffect* ed = static_cast< FlameEmitterEffect*>(iface);
					EffectFlameEmitter* ee = PX_NEW(EffectFlameEmitter)(ed->FlameEmitter->name(), ed, sdk, scene, dynamicSystemScene, moduleTurbulenceFS, mPose, mEnabled);
					ee->setCurrentScale(mObjectScale,mEffectPath);
					mEffects.pushBack(static_cast< EffectData*>(ee));
				}
				break;
			case ET_TURBULENCE_FS:
			{
				TurbulenceFieldSamplerEffect* ed = static_cast< TurbulenceFieldSamplerEffect*>(iface);
				EffectTurbulenceFS* ee = PX_NEW(EffectTurbulenceFS)(ed->TurbulenceFieldSampler->name(), ed, sdk, scene, dynamicSystemScene, moduleTurbulenceFS, mPose, mEnabled);
				ee->setCurrentScale(mObjectScale,mEffectPath);
				mEffects.pushBack(static_cast< EffectData*>(ee));
			}
			break;
			case ET_JET_FS:
			{
				JetFieldSamplerEffect* ed = static_cast< JetFieldSamplerEffect*>(iface);
				EffectJetFS* ee = PX_NEW(EffectJetFS)(ed->JetFieldSampler->name(), ed, sdk, scene, dynamicSystemScene, mPose, mEnabled);
				ee->setCurrentScale(mObjectScale,mEffectPath);
				mEffects.pushBack(static_cast< EffectData*>(ee));
			}
			break;
			case ET_WIND_FS:
				{
					WindFieldSamplerEffect* ed = static_cast< WindFieldSamplerEffect*>(iface);
					EffectWindFS* ee = PX_NEW(EffectWindFS)(ed->WindFieldSampler->name(), ed, sdk, scene, dynamicSystemScene, mPose, mEnabled);
					ee->setCurrentScale(mObjectScale,mEffectPath);
					mEffects.pushBack(static_cast< EffectData*>(ee));
				}
				break;
			case ET_RIGID_BODY:
				{
					RigidBodyEffect* ed = static_cast< RigidBodyEffect*>(iface);
					EffectRigidBody* ee = PX_NEW(EffectRigidBody)("RigidBody", ed, sdk, scene, dynamicSystemScene, mPose, mEnabled);
					ee->setCurrentScale(mObjectScale,mEffectPath);
					mEffects.pushBack(static_cast< EffectData*>(ee));
				}
				break;
			case ET_NOISE_FS:
			{
				NoiseFieldSamplerEffect* ed = static_cast< NoiseFieldSamplerEffect*>(iface);
				EffectNoiseFS* ee = PX_NEW(EffectNoiseFS)(ed->NoiseFieldSampler->name(), ed, sdk, scene, dynamicSystemScene, mPose, mEnabled);
				ee->setCurrentScale(mObjectScale,mEffectPath);
				mEffects.pushBack(static_cast< EffectData*>(ee));
			}
			break;
			case ET_VORTEX_FS:
			{
				VortexFieldSamplerEffect* ed = static_cast< VortexFieldSamplerEffect*>(iface);
				EffectVortexFS* ee = PX_NEW(EffectVortexFS)(ed->VortexFieldSampler->name(), ed, sdk, scene, dynamicSystemScene, mPose, mEnabled);
				ee->setCurrentScale(mObjectScale,mEffectPath);
				mEffects.pushBack(static_cast< EffectData*>(ee));
			}
			break;
			case ET_ATTRACTOR_FS:
			{
				AttractorFieldSamplerEffect* ed = static_cast< AttractorFieldSamplerEffect*>(iface);
				EffectAttractorFS* ee = PX_NEW(EffectAttractorFS)(ed->AttractorFieldSampler->name(), ed, sdk, scene, dynamicSystemScene, mPose, mEnabled);
				ee->setCurrentScale(mObjectScale,mEffectPath);
				mEffects.pushBack(static_cast< EffectData*>(ee));
			}
			break;
			case ET_FORCE_FIELD:
			{
				ForceFieldEffect* ed = static_cast< ForceFieldEffect*>(iface);
				EffectForceField* ee = PX_NEW(EffectForceField)(ed->ForceField->name(), ed, sdk, scene, dynamicSystemScene, mPose, mEnabled);
				ee->setCurrentScale(mObjectScale,mEffectPath);
				mEffects.pushBack(static_cast< EffectData*>(ee));
			}
			break;
			default:
				PX_ALWAYS_ASSERT();
				break;
			}
		}
	}
	addSelfToContext(*dynamicSystemScene.mApexScene->getApexContext());    // Add self to ApexScene
	addSelfToContext(dynamicSystemScene);	// Add self to ParticlesScene's list of actors
}

EffectPackageActor::~EffectPackageActor(void)
{
	for (PxU32 i = 0; i < mEffects.size(); i++)
	{
		EffectData* ed = mEffects[i];
		delete ed;
	}
	delete mEffectPath;
}

EffectType EffectPackageActor::getEffectType(const NxParameterized::Interface* iface)
{
	EffectType ret = ET_LAST;

	if (strcmp(iface->className(), EmitterEffect::staticClassName()) == 0)
	{
		ret = ET_EMITTER;
	}
	else if (strcmp(iface->className(), HeatSourceEffect::staticClassName()) == 0)
	{
		ret = ET_HEAT_SOURCE;
	}
	else if (strcmp(iface->className(), SubstanceSourceEffect::staticClassName()) == 0)
	{
		ret = ET_SUBSTANCE_SOURCE;
	}
	else if (strcmp(iface->className(), VelocitySourceEffect::staticClassName()) == 0)
	{
		ret = ET_VELOCITY_SOURCE;
	}
	else if (strcmp(iface->className(), FlameEmitterEffect::staticClassName()) == 0)
	{
		ret = ET_FLAME_EMITTER;
	}
	else if (strcmp(iface->className(), ForceFieldEffect::staticClassName()) == 0)
	{
		ret = ET_FORCE_FIELD;
	}
	else if (strcmp(iface->className(), JetFieldSamplerEffect::staticClassName()) == 0)
	{
		ret = ET_JET_FS;
	}
	else if (strcmp(iface->className(), WindFieldSamplerEffect::staticClassName()) == 0)
	{
		ret = ET_WIND_FS;
	}
	else if (strcmp(iface->className(), RigidBodyEffect::staticClassName()) == 0)
	{
		ret = ET_RIGID_BODY;
	}
	else if (strcmp(iface->className(), NoiseFieldSamplerEffect::staticClassName()) == 0)
	{
		ret = ET_NOISE_FS;
	}
	else if (strcmp(iface->className(), VortexFieldSamplerEffect::staticClassName()) == 0)
	{
		ret = ET_VORTEX_FS;
	}
	else if (strcmp(iface->className(), AttractorFieldSamplerEffect::staticClassName()) == 0)
	{
		ret = ET_ATTRACTOR_FS;
	}
	else if (strcmp(iface->className(), TurbulenceFieldSamplerEffect::staticClassName()) == 0)
	{
		ret = ET_TURBULENCE_FS;
	}
	else
	{
		PX_ALWAYS_ASSERT();
	}

	return ret;
}

const PxTransform& EffectPackageActor::getPose(void) const
{
	NX_READ_ZONE();
	return mPose;
}

void EffectPackageActor::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{
	if ( !mEnableDebugVisualization ) return;
	callback->pushRenderState();

	callback->addToCurrentState(DebugRenderState::CameraFacing);
	callback->addToCurrentState(DebugRenderState::CenterText);

	callback->debugText(mPose.p - PxVec3(0, 0.35f, 0), mAsset->getName());

	callback->debugAxes(PxMat44(mPose));

	for (physx::PxU32 i = 0; i < mEffects.size(); i++)
	{
		mEffects[i]->visualize(callback, solid);
	}

	callback->popRenderState();
}

void EffectPackageActor::setPose(const PxTransform& pose)
{
	NX_WRITE_ZONE();
	mPose = pose;

	for (physx::PxU32 i = 0; i < mEffects.size(); i++)
	{
		mEffects[i]->refresh(mPose, mEnabled, true, mRenderVolume,mEmitterValidateCallback);
	}
}

void EffectPackageActor::refresh(void)
{
	NX_WRITE_ZONE();
	for (physx::PxU32 i = 0; i < mEffects.size(); i++)
	{
		EffectData* ed = mEffects[i];

		if (ed->getEffectActor() )
		{
			ed->refresh(mPose, mEnabled, true, mRenderVolume,mEmitterValidateCallback);
		}
		else if ( ed->getType() == ET_RIGID_BODY )
		{
			EffectRigidBody *erb = static_cast< EffectRigidBody *>(ed);
			if ( erb->mRigidDynamic )
			{
				ed->refresh(mPose, mEnabled, true, mRenderVolume,mEmitterValidateCallback);
			}
		}
	}
}

void EffectPackageActor::release(void)
{
	delete this;
}

const char* EffectPackageActor::getName(void) const
{
	NX_READ_ZONE();
	return mAsset ? mAsset->getName() : NULL;
}


PxU32 EffectPackageActor::getEffectCount(void) const // returns the number of effects in the effect package
{
	NX_READ_ZONE();
	return mEffects.size();
}

EffectType EffectPackageActor::getEffectType(PxU32 effectIndex) const // return the type of effect.
{
	NX_READ_ZONE();
	EffectType ret = ET_LAST;

	if (effectIndex < mEffects.size())
	{
		EffectData* ed = mEffects[effectIndex];
		ret = ed->getType();
	}

	return ret;
}

NxApexActor* EffectPackageActor::getEffectActor(PxU32 effectIndex) const // return the base NxApexActor pointer
{
	NX_READ_ZONE();
	NxApexActor* ret = NULL;

	if (effectIndex < mEffects.size())
	{
		EffectData* ed = mEffects[effectIndex];
		ret = ed->getEffectActor();
	}


	return ret;
}
void EffectPackageActor::setEmitterState(bool state) // set the state for all emitters in this effect package.
{
	NX_WRITE_ZONE();
	for (PxU32 i = 0; i < mEffects.size(); i++)
	{
		if (mEffects[i]->getType() == ET_EMITTER)
		{
			NxApexActor* a = mEffects[i]->getEffectActor();
			if (a)
			{
				NxApexEmitterActor* ae = static_cast< NxApexEmitterActor*>(a);
				if (state)
				{
					ae->startEmit(false);
				}
				else
				{
					ae->stopEmit();
				}
			}
		}
	}
}

PxU32 EffectPackageActor::getActiveParticleCount(void) const // return the total number of particles still active in this effect package.
{
	NX_READ_ZONE();
	PxU32 ret = 0;

	for (PxU32 i = 0; i < mEffects.size(); i++)
	{
		if (mEffects[i]->getType() == ET_EMITTER)
		{
			NxApexActor* a = mEffects[i]->getEffectActor();
			if (a)
			{
				NxApexEmitterActor* ae = static_cast< NxApexEmitterActor*>(a);
				ret += ae->getActiveParticleCount();
			}
		}
	}
	return ret;
}

bool EffectPackageActor::isStillEmitting(void) const // return true if any emitters are still actively emitting particles.
{
	NX_READ_ZONE();
	bool ret = false;

	for (PxU32 i = 0; i < mEffects.size(); i++)
	{
		if (mEffects[i]->getType() == ET_EMITTER)
		{
			NxApexActor* a = mEffects[i]->getEffectActor();
			if (a)
			{
				NxApexEmitterActor* ae = static_cast< NxApexEmitterActor*>(a);
				if (ae->isEmitting())
				{
					ret = true;
					break;
				}
			}
		}
	}
	return ret;
}

// Effect class implementations

EffectEmitter::EffectEmitter(const char* parentName,
                             const EmitterEffect* data,
                             NxApexSDK& sdk,
                             NxApexScene& scene,
                             ParticlesScene& dscene,
                             const PxTransform& rootPose,
                             bool parentEnabled) : mData(data), EffectData(ET_EMITTER, &sdk,
	                                     &scene,
	                                     &dscene,
	                                     parentName,
	                                     NX_APEX_EMITTER_AUTHORING_TYPE_NAME,
	                                     *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
	mEmitterVelocity = physx::PxVec3(0, 0, 0);
	mFirstVelocityFrame = true;
	mHaveSetPosition = false;
	mVelocityTime = 0;
	mLastEmitterPosition = physx::PxVec3(0, 0, 0);
}

EffectEmitter::~EffectEmitter(void)
{
}

EffectTurbulenceFS::EffectTurbulenceFS(const char* parentName,
                                       TurbulenceFieldSamplerEffect* data,
                                       NxApexSDK& sdk,
                                       NxApexScene& scene,
                                       ParticlesScene& dscene,
									   NxModuleTurbulenceFS* moduleTurbulenceFS,
                                       const PxTransform& rootPose,
                                       bool parentEnabled) :
	mData(data),
	mModuleTurbulenceFS(moduleTurbulenceFS),
	EffectData(ET_TURBULENCE_FS, &sdk, &scene, &dscene, parentName, NX_TURBULENCE_FS_AUTHORING_TYPE_NAME, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
}

EffectTurbulenceFS::~EffectTurbulenceFS(void)
{
}

EffectJetFS::EffectJetFS(const char* parentName,
                         JetFieldSamplerEffect* data,
                         NxApexSDK& sdk,
                         NxApexScene& scene,
                         ParticlesScene& dscene,
                         const PxTransform& rootPose,
                         bool parentEnabled) : mData(data), EffectData(ET_JET_FS, &sdk, &scene, &dscene, parentName, NX_JET_FS_AUTHORING_TYPE_NAME, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
}

EffectJetFS::~EffectJetFS(void)
{
}

EffectWindFS::EffectWindFS(const char* parentName,
	WindFieldSamplerEffect* data,
	NxApexSDK& sdk,
	NxApexScene& scene,
	ParticlesScene& dscene,
	const PxTransform& rootPose,
	bool parentEnabled) : mData(data), EffectData(ET_WIND_FS, &sdk, &scene, &dscene, parentName, NX_WIND_FS_AUTHORING_TYPE_NAME, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
}

EffectWindFS::~EffectWindFS(void)
{
}



EffectAttractorFS::EffectAttractorFS(const char* parentName,
                                     AttractorFieldSamplerEffect* data,
                                     NxApexSDK& sdk,
                                     NxApexScene& scene,
                                     ParticlesScene& dscene,
                                     const PxTransform& rootPose,
                                     bool parentEnabled) : mData(data), EffectData(ET_ATTRACTOR_FS, &sdk, &scene, &dscene, parentName, NX_ATTRACTOR_FS_AUTHORING_TYPE_NAME, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
}

EffectAttractorFS::~EffectAttractorFS(void)
{
}

void EffectEmitter::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{
	callback->debugText(mPose.p, mData->Emitter->name());
	callback->debugAxes(PxMat44(mPose));

}



EffectHeatSource::EffectHeatSource(const char* parentName,
                                   HeatSourceEffect* data,
                                   NxApexSDK& sdk,
                                   NxApexScene& scene,
                                   ParticlesScene& dscene,
                                   NxModuleTurbulenceFS* moduleTurbulenceFS,
                                   const PxTransform& rootPose,
                                   bool parentEnabled) : mData(data),
	mModuleTurbulenceFS(moduleTurbulenceFS),
	EffectData(ET_HEAT_SOURCE, &sdk, &scene, &dscene, parentName, NX_HEAT_SOURCE_AUTHORING_TYPE_NAME, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
}

EffectHeatSource::~EffectHeatSource(void)
{
}

void EffectHeatSource::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{
}

bool EffectHeatSource::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();
		physx::PxTransform myPose(parent * localPose);
		getSamplePoseSpline(myPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				PxTransform pose(myPose);
				bool ok = NxParameterized::setParamMat34(*descParams, "initialPose", pose);
				PX_ASSERT(ok);
				ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				PX_UNUSED(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				ret = true;
			}
			const NxParameterized::Interface* iface = mAsset->getAssetNxParameterized();
			const turbulencefs::HeatSourceAssetParams* hap = static_cast< const turbulencefs::HeatSourceAssetParams*>(iface);
			mAverageTemperature = hap->averageTemperature;
			mStandardDeviationTemperature = hap->stdTemperature;
		}
		else if (mActor)
		{
			NxHeatSourceActor* a = static_cast< NxHeatSourceActor*>(mActor);
			physx::apex::NxApexShape *s = a->getShape();
			s->setPose(myPose);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
		}
	}
	else
	{
		if ( mActor )
		{
			releaseActor();
			ret = true;
		}
	}
	return ret;
}

EffectSubstanceSource::EffectSubstanceSource(const char* parentName,
								   SubstanceSourceEffect* data,
								   NxApexSDK& sdk,
								   NxApexScene& scene,
								   ParticlesScene& dscene,
								   NxModuleTurbulenceFS* moduleTurbulenceFS,
								   const PxTransform& rootPose,
								   bool parentEnabled) : mData(data),
								   mModuleTurbulenceFS(moduleTurbulenceFS),
								   EffectData(ET_SUBSTANCE_SOURCE, &sdk, &scene, &dscene, parentName, NX_SUBSTANCE_SOURCE_AUTHORING_TYPE_NAME, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
}

EffectSubstanceSource::~EffectSubstanceSource(void)
{
}

void EffectSubstanceSource::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{
}

bool EffectSubstanceSource::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();
		physx::PxTransform myPose(parent * localPose);
		getSamplePoseSpline(myPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				PxTransform pose(myPose);
				bool ok = NxParameterized::setParamMat34(*descParams, "initialPose", pose);
				PX_ASSERT(ok);
				ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				PX_UNUSED(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				ret = true;
			}
			const NxParameterized::Interface* iface = mAsset->getAssetNxParameterized();
			const turbulencefs::SubstanceSourceAssetParams* hap = static_cast< const turbulencefs::SubstanceSourceAssetParams*>(iface);
			mAverageDensity = hap->averageDensity;
			mStandardDeviationDensity = hap->stdDensity;
		}
		else if (mActor)
		{
			NxSubstanceSourceActor* a = static_cast< NxSubstanceSourceActor*>(mActor);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
			physx::apex::NxApexShape *s = a->getShape();
			s->setPose(myPose);
		}
	}
	else
	{
		if ( mActor )
		{
			ret = true;
			releaseActor();
		}
	}
	return ret;
}

EffectVelocitySource::EffectVelocitySource(const char* parentName,
	VelocitySourceEffect* data,
	NxApexSDK& sdk,
	NxApexScene& scene,
	ParticlesScene& dscene,
	NxModuleTurbulenceFS* moduleTurbulenceFS,
	const PxTransform& rootPose,
	bool parentEnabled) : mData(data),
	mModuleTurbulenceFS(moduleTurbulenceFS),
	EffectData(ET_VELOCITY_SOURCE, &sdk, &scene, &dscene, parentName, NX_VELOCITY_SOURCE_AUTHORING_TYPE_NAME, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
}

EffectVelocitySource::~EffectVelocitySource(void)
{
}

void EffectVelocitySource::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{
}

bool EffectVelocitySource::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();
		physx::PxTransform myPose(parent * localPose);
		getSamplePoseSpline(myPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				PxTransform pose(myPose);
				bool ok = NxParameterized::setParamMat34(*descParams, "initialPose", pose);
				PX_ASSERT(ok);
				ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				PX_UNUSED(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				ret = true;
			}
			const NxParameterized::Interface* iface = mAsset->getAssetNxParameterized();
			const turbulencefs::VelocitySourceAssetParams* hap = static_cast< const turbulencefs::VelocitySourceAssetParams*>(iface);
			mAverageVelocity = hap->averageVelocity;
			mStandardDeviationVelocity = hap->stdVelocity;
		}
		else if (mActor)
		{
			NxVelocitySourceActor* a = static_cast< NxVelocitySourceActor*>(mActor);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
			physx::apex::NxApexShape *s = a->getShape();
			s->setPose(myPose);
		}
	}
	else
	{
		if ( mActor )
		{
			ret = true;
			releaseActor();
		}
	}
	return ret;
}


EffectFlameEmitter::EffectFlameEmitter(const char* parentName,
	FlameEmitterEffect* data,
	NxApexSDK& sdk,
	NxApexScene& scene,
	ParticlesScene& dscene,
	NxModuleTurbulenceFS* moduleTurbulenceFS,
	const PxTransform& rootPose,
	bool parentEnabled) : mData(data),
	mModuleTurbulenceFS(moduleTurbulenceFS),
	EffectData(ET_FLAME_EMITTER, &sdk, &scene, &dscene, parentName, NX_FLAME_EMITTER_AUTHORING_TYPE_NAME, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
}

EffectFlameEmitter::~EffectFlameEmitter(void)
{
}

void EffectFlameEmitter::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{
}

bool EffectFlameEmitter::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();
		physx::PxTransform myPose(parent * localPose);
		getSamplePoseSpline(myPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				PxTransform pose(myPose);
				bool ok = NxParameterized::setParamMat34(*descParams, "initialPose", pose);
				PX_ASSERT(ok);
				ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				PX_UNUSED(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				ret = true;
			}
		}
		else if (mActor)
		{
			NxFlameEmitterActor* a = static_cast< NxFlameEmitterActor*>(mActor);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
			a->setPose(myPose);
		}
	}
	else
	{
		if ( mActor )
		{
			ret = true;
			releaseActor();
		}
	}
	return ret;
}


void EffectTurbulenceFS::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{

}

void EffectJetFS::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{

}

void EffectWindFS::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{

}

void EffectAttractorFS::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{

}

void EffectEmitter::computeVelocity(physx::PxF32 dtime)
{
	mVelocityTime += dtime;
	if (mFirstVelocityFrame)
	{
		if (mActor)
		{
			mFirstVelocityFrame = false;
			mVelocityTime = 0;
			NxApexEmitterActor* ea = static_cast< NxApexEmitterActor*>(mActor);
			mLastEmitterPosition = ea->getGlobalPose().getPosition();
		}
	}
	else if (mHaveSetPosition && mActor)
	{
		mHaveSetPosition = false;
		NxApexEmitterActor* ea = static_cast< NxApexEmitterActor*>(mActor);
		physx::PxVec3 newPos = ea->getGlobalPose().getPosition();
		mEmitterVelocity = (newPos - mLastEmitterPosition) * (1.0f / mVelocityTime);
		mLastEmitterPosition = newPos;
		mVelocityTime = 0;
	}
}

bool EffectEmitter::refresh(const PxTransform& parent,
							bool parentEnabled,
							bool fromSetPose,
							NxApexRenderVolume* renderVolume,
							NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();

		mPose = parent * localPose;
		getSamplePoseSpline(mPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			mFirstVelocityFrame = true;
			mHaveSetPosition = false;
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				PxMat44 myPose(mPose);
				bool ok = NxParameterized::setParamMat44(*descParams, "initialPose", myPose);
				PX_UNUSED(ok);
				PX_ASSERT(ok);
				ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				ok = NxParameterized::setParamBool(*descParams, "emitAssetParticles", true);
				PX_ASSERT(ok);
				NxApexEmitterAsset* easset = static_cast< NxApexEmitterAsset*>(mAsset);
				NxApexEmitterActor* ea = mParticlesScene->getEmitterFromPool(easset);
				if (ea)
				{
					ea->setCurrentPose(myPose);
					ea->setCurrentScale(mObjectScale*getSampleScaleSpline());
					mActor = static_cast< NxApexActor*>(ea);
				}
				else
				{
					mActor = mAsset->createApexActor(*descParams, *mApexScene);
				}
				if (mActor)
				{
					const NxParameterized::Interface* iface = mAsset->getAssetNxParameterized();
					const char* className = iface->className();
					if (strcmp(className, "ApexEmitterAssetParameters") == 0)
					{						
						const emitter::ApexEmitterAssetParameters* ap = static_cast<const  emitter::ApexEmitterAssetParameters*>(iface);
						mRateRange.minimum = ap->rateRange.min;
						mRateRange.maximum = ap->rateRange.max;
						mLifetimeRange.minimum = ap->lifetimeRange.min;
						mLifetimeRange.maximum = ap->lifetimeRange.max;
						NxApexEmitterActor* ea = static_cast< NxApexEmitterActor*>(mActor);
						ea->setRateRange(mRateRange);
						ea->setLifetimeRange(mLifetimeRange);
						ea->startEmit(false);
						ea->setPreferredRenderVolume(renderVolume);
						ea->setApexEmitterValidateCallback(callback);
					}
					ret = true;
				}
			}
		}
		else if (mActor)
		{
			NxApexEmitterActor* ea = static_cast< NxApexEmitterActor*>(mActor);
			mHaveSetPosition = true; // set semaphore for computing the velocity.
			ea->setCurrentPose(mPose);
			ea->setObjectScale(mObjectScale*getSampleScaleSpline());
			ea->setPreferredRenderVolume(renderVolume);
		}
	}
	else
	{
		if ( mActor )
		{
			releaseActor();
			ret = true;
		}
	}
	return ret;
}


bool EffectTurbulenceFS::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();
		physx::PxTransform myPose(parent * localPose);
		getSamplePoseSpline(myPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				bool ok = NxParameterized::setParamMat44(*descParams, "initialPose", myPose);
				PX_UNUSED(ok);
				PX_ASSERT(ok);
				ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				ret = true;
			}
		}
		else if (mActor)
		{
			NxTurbulenceFSActor* a = static_cast< NxTurbulenceFSActor*>(mActor);
			a->setPose(myPose);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
		}
	}
	else
	{
		if ( mActor )
		{
			releaseActor();
			ret = true;
		}
	}
	return ret;
}

bool EffectJetFS::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();

		physx::PxTransform myPose(parent * localPose);
		getSamplePoseSpline(myPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				bool ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				PX_UNUSED(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				if (mActor)
				{
					NxJetFSActor* fs = static_cast< NxJetFSActor*>(mActor);
					if (fs)
					{
						fs->setCurrentPose(myPose);
						fs->setCurrentScale(mObjectScale*getSampleScaleSpline());
					}
					ret = true;
				}
			}
		}
		else if (mActor)
		{
			NxJetFSActor* a = static_cast< NxJetFSActor*>(mActor);
			a->setCurrentPose(myPose);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
		}
	}
	else
	{
		if ( mActor )
		{
			releaseActor();
			ret = true;
		}
	}
	return ret;
}

bool EffectWindFS::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();
		physx::PxTransform myPose(parent * localPose);
		getSamplePoseSpline(myPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				bool ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				PX_UNUSED(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				if (mActor)
				{
					NxWindFSActor* fs = static_cast< NxWindFSActor*>(mActor);
					if (fs)
					{
						fs->setCurrentPose(myPose);
						fs->setCurrentScale(mObjectScale*getSampleScaleSpline());
					}
					ret = true;
				}
			}
		}
		else if (mActor)
		{
			NxWindFSActor* a = static_cast< NxWindFSActor*>(mActor);
			a->setCurrentPose(myPose);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
		}
	}
	else
	{
		if ( mActor )
		{
			releaseActor();
			ret = true;
		}
	}
	return ret;
}

bool EffectAttractorFS::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform initialPose = _getPose(mData->EffectProperties.Position.TranslateX*mObjectScale*getSampleScaleSpline(),
		                                   mData->EffectProperties.Position.TranslateY*mObjectScale*getSampleScaleSpline(),
		                                   mData->EffectProperties.Position.TranslateZ*mObjectScale*getSampleScaleSpline(),
		                                   mData->EffectProperties.Orientation.RotateX,
		                                   mData->EffectProperties.Orientation.RotateY,
		                                   mData->EffectProperties.Orientation.RotateZ);

		physx::PxTransform myPose(parent * initialPose);
		getSamplePoseSpline(myPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				
				bool ok = NxParameterized::setParamMat34(*descParams, "initialPose", myPose);
				PX_UNUSED(ok);
				PX_ASSERT(ok);
				ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				ret = true;
			}
		}
		else if (mActor)
		{
			NxAttractorFSActor* a = static_cast< NxAttractorFSActor*>(mActor);
			a->setCurrentPosition(myPose.p);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
		}
	}
	else
	{
		if ( mActor )
		{
			releaseActor();
			ret = true;
		}
	}
	return ret;
}


EffectForceField::EffectForceField(const char* parentName, ForceFieldEffect* data, NxApexSDK& sdk, NxApexScene& scene, ParticlesScene& dscene, const PxTransform& rootPose, bool parentEnabled) : mData(data), EffectData(ET_FORCE_FIELD, &sdk, &scene, &dscene, parentName, NX_FORCEFIELD_AUTHORING_TYPE_NAME, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
}

EffectForceField::~EffectForceField(void)
{
}

void EffectForceField::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{

}

bool EffectForceField::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform initialPose = _getPose(mData->EffectProperties.Position.TranslateX*mObjectScale*getSampleScaleSpline(),
		                                   mData->EffectProperties.Position.TranslateY*mObjectScale*getSampleScaleSpline(),
		                                   mData->EffectProperties.Position.TranslateZ*mObjectScale*getSampleScaleSpline(),
		                                   mData->EffectProperties.Orientation.RotateX,
		                                   mData->EffectProperties.Orientation.RotateY,
		                                   mData->EffectProperties.Orientation.RotateZ);
		physx::PxTransform myPose(parent * initialPose);
		getSamplePoseSpline(myPose);
		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				bool ok = NxParameterized::setParamF32(*descParams, "scale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				PX_UNUSED(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				if (mActor)
				{
					NxForceFieldActor* fs = static_cast< NxForceFieldActor*>(mActor);
					if (fs)
					{
						fs->setPose(myPose);
						fs->setCurrentScale(mObjectScale*getSampleScaleSpline());
					}
					ret = true;
				}
			}
		}
		else if (mActor)
		{
			NxForceFieldActor* a = static_cast< NxForceFieldActor*>(mActor);
			a->setPose(myPose);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
		}
	}
	else
	{
		if ( mActor )
		{
			releaseActor();
			ret = true;
		}
	}
	return ret;
}

// Returns true if this is a type which has a named resource asset which needs to be resolved.  RigidBody effects do not; all of their properties are embedded
static bool isNamedResourceType(EffectType type)
{
	bool ret = true;

	if ( type == ET_RIGID_BODY )
	{
		ret = false;
	}

	return ret;
}

EffectData::EffectData(EffectType type,
                       NxApexSDK* sdk,
                       NxApexScene* scene,
                       ParticlesScene* dscene,
                       const char* assetName,
                       const char* nameSpace,
					   RigidBodyEffectNS::EffectProperties_Type &effectProperties)
{
	mEffectPath = NULL;
	mParentPath = NULL;
	mObjectScale = 1;

	mLocalPose = _getPose(effectProperties.Position.TranslateX,
		effectProperties.Position.TranslateY,
		effectProperties.Position.TranslateZ,
		effectProperties.Orientation.RotateX,
		effectProperties.Orientation.RotateY,
		effectProperties.Orientation.RotateZ);

	mEffectPath = PX_NEW(EffectPath);
	if ( !mEffectPath->init(effectProperties.Path) )
	{
		delete mEffectPath;
		mEffectPath = NULL;
	}

	mEnabled = effectProperties.Enable;
	mRandomDeviation = effectProperties.RandomizeRepeatTime;
	mForceRenableEmitter = false;
	mUseEmitterPool = dscene->getModuleParticles()->getUseEmitterPool();
	mFirstRate = true;
	mType = type;
	mState = ES_INITIAL_DELAY;
	mStateTime = getRandomTime(effectProperties.InitialDelayTime);
	mSimulationTime = 0;
	mStateCount = 0;
	mParticlesScene = dscene;
	mInitialDelayTime = effectProperties.InitialDelayTime;
	mDuration = effectProperties.Duration;
	mRepeatCount = effectProperties.RepeatCount;
	mRepeatDelay = effectProperties.RepeatDelay;
	mApexSDK = sdk;
	mApexScene = scene;
	mActor = NULL;
	mNameSpace = nameSpace;
	mAsset = NULL;
	if ( isNamedResourceType(mType) ) 
	{
		mAsset = (physx::apex::NxApexAsset*)mApexSDK->getNamedResourceProvider()->getResource(mNameSpace, assetName);
		if (mAsset)
		{
			if (mType == ET_EMITTER)
			{
				if (!mUseEmitterPool)
				{
					mApexSDK->getNamedResourceProvider()->setResource(mNameSpace, assetName, mAsset, true);
				}
			}
			else
			{
				mApexSDK->getNamedResourceProvider()->setResource(mNameSpace, assetName, mAsset, true);
			}
		}
	}
}

EffectData::~EffectData(void)
{
	releaseActor();

	delete mEffectPath;

	if (mAsset)
	{
		if (mType == ET_EMITTER)
		{
			if (!mUseEmitterPool)
			{
				mApexSDK->getNamedResourceProvider()->releaseResource(mNameSpace, mAsset->getName());
			}
		}
		else
		{
			mApexSDK->getNamedResourceProvider()->releaseResource(mNameSpace, mAsset->getName());
		}
	}
}

void EffectData::releaseActor(void)
{
	if (mActor)
	{
		if (mType == ET_EMITTER && mUseEmitterPool)
		{
			NxApexEmitterActor* ae = static_cast< NxApexEmitterActor*>(mActor);
			mParticlesScene->addToEmitterPool(ae);
		}
		else
		{
			mActor->release();
		}
		mActor = NULL;
	}
}

PxF32 EffectData::getRandomTime(PxF32 baseTime)
{
	PxF32 deviation = baseTime * mRandomDeviation;
	return ranf(baseTime - deviation, baseTime + deviation);
}

bool EffectData::simulate(PxF32 dtime, bool& reset)
{
	bool ret = false;

	if (!mEnabled)
	{
		return false;
	}

	switch (mState)
	{
		case ES_INITIAL_DELAY:
			mStateTime -= dtime;
			if (mStateTime <= 0)
			{
				mState = ES_ACTIVE;		// once past the initial delay, it is now active
				mStateTime = getRandomTime(mDuration);	// Time to the next state change...
				ret = true;	// set ret to true because the effect is alive
			}
			break;
		case ES_ACTIVE:
			ret = true;		// it's still active..
			mStateTime -= dtime;	// decrement delta time to next state change.
			mSimulationTime+=dtime;
			if (mStateTime <= 0)   // if it's time for a state change.
			{
				if (mDuration == 0)   // if the
				{
					if (mRepeatDelay > 0)   // if there is a delay until the time we repeate
					{
						mStateTime = getRandomTime(mRepeatDelay);	// set time until repeat delay
						mState = ES_REPEAT_DELAY; // change state to repeat delay
					}
					else
					{
						mStateTime = getRandomTime(mDuration); // if there is no repeat delay; just continue
						if (mRepeatCount > 1)
						{
							reset = true; // looped..
						}
					}
				}
				else
				{
					mStateCount++;	// increment the state change counter.
					if (mStateCount >= mRepeatCount && mRepeatCount != 9999)   // have we hit the total number repeat counts.
					{
						mState = ES_DONE; // then we are completely done; the actor is no longer alive
						ret = false;
					}
					else
					{
						if (mRepeatDelay > 0)   // is there a repeat delay?
						{
							mStateTime = getRandomTime(mRepeatDelay);
							mState = ES_REPEAT_DELAY;
						}
						else
						{
							mStateTime = getRandomTime(mDuration);
							reset = true;
						}
					}
				}
			}
			else
			{
				if ( mEffectPath && mDuration != 0 )
				{
					mEffectPath->computeSampleTime(mStateTime,mDuration);
				}
			}
			if ( mDuration == 0 && mEffectPath )
			{
				mEffectPath->computeSampleTime(mSimulationTime,mEffectPath->getPathDuration());
			}
			break;
		case ES_REPEAT_DELAY:
			mStateTime -= dtime;
			if (mStateTime < 0)
			{
				mState = ES_ACTIVE;
				mStateTime = getRandomTime(mDuration);
				reset = true;
				ret = true;
			}
			break;
		case ES_DONE:
			break;
		default:
			//PX_ASSERT(0);
			break;
	}

	return ret;
}

void EffectPackageActor::updateParticles(PxF32 dtime)
{
	mSimTime = dtime;
	mCurrentLifeTime += mSimTime;
}

PxF32 EffectPackageActor::internalGetDuration(void)
{
	PxF32 duration = 1000;
	for (physx::PxU32 i = 0; i < mEffects.size(); i++)
	{
		EffectData* ed = mEffects[i];
		if (ed->getDuration() < duration)
		{
			duration = ed->getDuration();
		}
	}
	return duration;
}

// applies velocity adjustment to this range
static void processVelocityAdjust(const particles::EmitterEffectNS::EmitterVelocityAdjust_Type& vprops,
                                  const physx::PxVec3& velocity,
                                  NxRange<physx::PxF32> &range)
{
	
	physx::PxF32 r = 1;
	physx::PxF32 v = velocity.magnitude();	// compute the absolute magnitude of the current emitter velocity
	if (v <= vprops.VelocityLow)   // if the velocity is less than the minimum velocity adjustment range
	{
		r = vprops.LowValue;	// Use the 'low-value' ratio adjustment
	}
	else if (v >= vprops.VelocityHigh)   // If the velocity is greater than the high range
	{
		r = vprops.HighValue; // then clamp tot he high value adjustment
	}
	else
	{
		PxF32 ratio = 1;
		PxF32 diff = vprops.VelocityHigh - vprops.VelocityLow; // Compute the velocity differntial
		if (diff > 0)
		{
			ratio = 1.0f / diff;	// compute the inverse velocity differential
		}
		physx::PxF32 l = (v - vprops.VelocityLow) * ratio;	// find out the velocity lerp rate
		r = (vprops.HighValue - vprops.LowValue) * l + vprops.LowValue;
	}
	range.minimum *= r;
	range.maximum *= r;
}


void EffectPackageActor::updatePoseAndBounds(bool screenCulling, bool znegative)
{
	if (!mEnabled)
	{
		for (physx::PxU32 i = 0; i < mEffects.size(); i++)
		{
			EffectData* ed = mEffects[i];
			if (ed->getEffectActor())
			{
				ed->releaseActor();
			}
			else if ( ed->getType() == ET_RIGID_BODY )
			{
				EffectRigidBody *erb = static_cast< EffectRigidBody *>(ed);
				erb->releaseRigidBody();
			}
		}
		mAlive = false;
		return;
	}

	PxF32 ZCOMPARE = -1;
	if (znegative)
	{
		ZCOMPARE *= -1;
	}

	//
	bool prevVisible = mVisible;
	mVisible = true;  // default visibile state is based on whether or not this effect is enabled.
	physx::PxF32 emitterRate = 1;
	mVisState = VS_ON_SCREEN; // default value
	if (mVisible)   // if it's considered visible/enabled then let's do the LOD cacluation
	{
		const physx::PxMat44& viewMatrix = mScene->getViewMatrix();
		physx::PxVec3 pos = viewMatrix.transform(mPose.p);
		float magnitudeSquared = pos.magnitudeSquared();
		if (mData->LODSettings.CullByDistance)   // if distance culling is enabled
		{

			// If the effect is past the maximum distance then mark it is no longer visible.
			if (magnitudeSquared > mData->LODSettings.FadeDistanceEnd * mData->LODSettings.FadeDistanceEnd)
			{
				mVisible = false;
				mVisState = VS_OFF_SCREEN;
			} // if the effect is within the fade range; then compute the lerp value for it along that range as 'emitterRate'
			else if (magnitudeSquared > mData->LODSettings.FadeDistanceEnd * mData->LODSettings.FadeDistanceBegin)
			{
				physx::PxF32 distance = physx::PxSqrt(magnitudeSquared);
				physx::PxF32 delta = mData->LODSettings.FadeDistanceEnd - mData->LODSettings.FadeDistanceBegin;
				if (delta > 0)
				{
					emitterRate = 1.0f - ((distance - mData->LODSettings.FadeDistanceBegin) / delta);
				}
			}
		}
		// If it's still considered visible (i.e. in range) and off screen culling is enabled; let's test it's status on/off screen
		if (mVisible && mData->LODSettings.CullOffScreen && screenCulling)
		{
			if (magnitudeSquared < (mData->LODSettings.ScreenCullDistance * mData->LODSettings.ScreenCullDistance))
			{
				mVisState = VS_TOO_CLOSE;
			}
			else if (pos.z * ZCOMPARE > 0)
			{
				mVisState = VS_BEHIND_SCREEN;
			}
			else
			{
				const physx::PxMat44& projMatrix = mScene->getProjMatrix();
				PxVec4 p(pos.x, pos.y, pos.z, 1);
				p = projMatrix.transform(p);
				PxF32 recipW = 1.0f / p.w;

				p.x = p.x * recipW;
				p.y = p.y * recipW;
				p.z = p.z * recipW;

				PxF32 smin = -1 - mData->LODSettings.ScreenCullSize;
				PxF32 smax = 1 + mData->LODSettings.ScreenCullSize;

				if (p.x >= smin && p.x <= smax && p.y >= smin && p.y <= smax)
				{
					mVisState = VS_ON_SCREEN;
				}
				else
				{
					mVisState = VS_OFF_SCREEN;
				}
			}
		}
	}
	if (mVisState == VS_ON_SCREEN || mVisState == VS_TOO_CLOSE)
	{
		mOffScreenTime = 0;
	}
	else
	{
		mOffScreenTime += mSimTime;
		if (mOffScreenTime > mData->LODSettings.OffScreenCullTime)
		{
			mVisible = false; // mark it as non-visible due to it being off sceen too long.
			mAlive = false;
		}
		else
		{
			mVisible = mEverVisible; // was it ever visible?
		}
	}

	if ( mEffectPath )
	{
		mEffectPath->computeSampleTime(mCurrentLifeTime,mData->Path.PathDuration);
	}

	if (mFirstFrame && !mVisible && screenCulling)
	{
		if (getDuration() != 0)
		{
			mEnabled = false;
			return;
		}
	}


	if (mVisible)
	{
		mEverVisible = true;
	}


	bool aliveState = mVisible;

	// do the fade in/fade out over time logic...
	if (mData->LODSettings.FadeOutRate > 0)   // If there is a fade in/out time.
	{
		if (aliveState)    // if the effect is considered alive/visible then attenuate the emitterRate based on that fade in time value
		{
			mFadeTime += mSimTime;
			if (mFadeTime < mData->LODSettings.FadeOutRate)
			{
				emitterRate = emitterRate * mFadeTime / mData->LODSettings.FadeOutRate;
			}
			else
			{
				mFadeTime = mData->LODSettings.FadeOutRate;
			}
		}
		else // if the effect is not visible then attenuate it based on the fade out time
		{
			mFadeTime -= mSimTime;
			if (mFadeTime > 0)
			{
				emitterRate = emitterRate * mFadeTime / mData->LODSettings.FadeOutRate;
				aliveState = true; // still alive because it hasn't finsihed fading out...
			}
			else
			{
				mFadeTime = 0;
			}
		}
	}

	if (mFadeIn)
	{
		mFadeInDuration += mSimTime;
		if (mFadeInDuration > mFadeInTime)
		{
			mFadeIn = false;
		}
		else
		{
			PxF32 fadeScale = (mFadeInDuration / mFadeInTime);
			emitterRate *= fadeScale;
		}
	}

	if (mFadeOut)
	{
		mFadeOutDuration += mSimTime;
		if (mFadeOutDuration > mFadeOutTime)
		{
			aliveState = mVisible = false;
		}
		else
		{
			PxF32 fadeScale = 1.0f - (mFadeOutDuration / mFadeOutTime);
			emitterRate *= fadeScale;
		}
	}

	if (mVisible)
	{
		mNotVisibleTime = 0;
	}
	else
	{
		mNotVisibleTime += mSimTime;
	}

	bool anyAlive = false;
	bool rigidBodyChange = false;

	for (physx::PxU32 i = 0; i < mEffects.size(); i++)
	{
		bool alive = aliveState;
		EffectData* ed = mEffects[i];
		// only emitters can handle a 'repeat' count.
		// others have an initial delay
		bool reset = false;

		if (ed->getDuration() == 0)
		{
			reset = !prevVisible; // if it was not previously visible then force a reset to bring it back to life.
		}
		if (alive)
		{
			alive = ed->simulate(mSimTime, reset);
			if ( alive )
			{
				anyAlive = true;
			}
		}
		if (ed->isDead())   // if it's lifetime has completely expired kill it!
		{
			if (ed->getEffectActor())
			{
				ed->releaseActor();
			}
			else if ( ed->getType() == ET_RIGID_BODY )
			{
				EffectRigidBody *erb = static_cast< EffectRigidBody *>(ed);
				erb->releaseRigidBody();
				rigidBodyChange = true;
			}
		}
		else
		{
			switch (ed->getType())
			{
				case ET_EMITTER:
				{
					EffectEmitter* ee = static_cast< EffectEmitter*>(ed);
					ee->computeVelocity(mSimTime);
					NxApexEmitterActor* ea = static_cast< NxApexEmitterActor*>(ed->getEffectActor());
					// if there is already an emitter actor...
					if (ea)
					{
						if (alive)   // is it alive?
						{
							if ( ed->getForceRenableEmitterSemaphore() && !ea->isEmitting() )
							{
								reset = true;
							}
							if (reset)   // is it time to reset it's condition?
							{
								ea->startEmit(false);
							}
							if (mData->LODSettings.FadeEmitterRate || mData->LODSettings.RandomizeEmitterRate || mFadeOut || mFadeIn)
							{
								// attenuate the emitter rate range based on the previously computed LOD lerp value
								NxRange<physx::PxF32> rateRange;
								if (mData->LODSettings.RandomizeEmitterRate && ee->mFirstRate)
								{
									physx::PxF32 rate = ranf(ee->mRateRange.minimum, ee->mRateRange.maximum);
									ee->mRateRange.minimum = rate;
									ee->mRateRange.maximum = rate;
									if (!mData->LODSettings.FadeEmitterRate)
									{
										ea->setRateRange(rateRange);
									}
									ee->mFirstRate = false;
								}
								if (mData->LODSettings.FadeEmitterRate ||
										mFadeOut ||
										mFadeIn ||
										ee->mData->EmitterVelocityChanges.AdjustEmitterRate.AdjustEnabled ||
										ee->mData->EmitterVelocityChanges.AdjustLifetime.AdjustEnabled)
								{
									rateRange.minimum = ee->mRateRange.minimum * emitterRate;
									rateRange.maximum = ee->mRateRange.maximum * emitterRate;
									if (ee->mData->EmitterVelocityChanges.AdjustEmitterRate.AdjustEnabled)
									{
										processVelocityAdjust(ee->mData->EmitterVelocityChanges.AdjustEmitterRate, ee->mEmitterVelocity, rateRange);
									}
									ea->setRateRange(rateRange);
									if (ee->mData->EmitterVelocityChanges.AdjustLifetime.AdjustEnabled)
									{
										NxRange<physx::PxF32> rateRange = ee->mLifetimeRange;
										processVelocityAdjust(ee->mData->EmitterVelocityChanges.AdjustLifetime, ee->mEmitterVelocity, rateRange);
										ea->setLifetimeRange(rateRange);
									}
								}
							}

							if ( ee->activePath() )
							{
								ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());

								PxTransform pose = mPose;
								ee->getSamplePoseSpline(pose);

								PxMat44 p(pose);
								ea->setCurrentPose(p);
							}

						}
						else
						{
							if (mNotVisibleTime > mData->LODSettings.NonVisibleDeleteTime)   // if it's been non-visible for a long time; delete it, don't just disable it!
							{
								if (ed->getEffectActor())
								{
									ed->releaseActor();
								}
							}
							else
							{
								if (ea->isEmitting())
								{
									ea->stopEmit(); // just stop emitting but don't destroy the actor.
								}
							}
						}
					}
					else
					{
						if (alive)   // if it is now alive but was not previously; start the initial instance.
						{
							ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
						}
					}
				}
				break;
			case ET_ATTRACTOR_FS:
			{
				EffectAttractorFS *ee = static_cast< EffectAttractorFS *>(ed);
				NxAttractorFSActor* ea = static_cast< NxAttractorFSActor*>(ed->getEffectActor());
				// if there is already an emitter actor...
				if (ea)
				{
					if (alive)   // is it alive?
					{
						if (reset)   // is it time to reset it's condition?
						{
							ea->setEnabled(true);
						}
						if (mData->LODSettings.FadeAttractorFieldStrength)
						{
							// TODO
						}
						if ( ee->activePath() )
						{
							PxTransform pose = mPose;
							ee->getSamplePoseSpline(pose);
							ea->setCurrentPosition(pose.p);
							ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
						}
					}
					else
					{
						if (mNotVisibleTime > mData->LODSettings.NonVisibleDeleteTime)   // if it's been non-visible for a long time; delete it, don't just disable it!
						{

							if (ed->getEffectActor())
							{
								ed->releaseActor();
							}
						}
						else
						{
							ea->setEnabled(false);
						}
					}
				}
				else
				{
					if (alive)   // if it is now alive but was not previously; start the initial instance.
					{
						ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
					}
				}
			}
			break;
			case ET_JET_FS:
			{
				EffectJetFS *ee = static_cast< EffectJetFS *>(ed);
				NxJetFSActor* ea = static_cast< NxJetFSActor*>(ed->getEffectActor());
				// if there is already an emitter actor...
				if (ea)
				{
					if (alive)   // is it alive?
					{
						if (reset)   // is it time to reset it's condition?
						{
							ea->setEnabled(true);
						}
						if (mData->LODSettings.FadeJetFieldStrength)
						{
							// TODO
						}
						if ( ee->activePath() )
						{
							PxTransform pose = mPose;
							ee->getSamplePoseSpline(pose);
							ea->setCurrentPose(pose);
							ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
						}
					}
					else
					{
						if (mNotVisibleTime > mData->LODSettings.NonVisibleDeleteTime)   // if it's been non-visible for a long time; delete it, don't just disable it!
						{
							if (ed->getEffectActor())
							{
								ed->releaseActor();
							}
						}
						else
						{
							ea->setEnabled(false);
						}
					}
				}
				else
				{
					if (alive)   // if it is now alive but was not previously; start the initial instance.
					{
						ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
					}
				}
			}
			break;
			case ET_RIGID_BODY:
				{
					EffectRigidBody *erb = static_cast< EffectRigidBody *>(ed);
					if ( alive )
					{
						if ( ed->refresh(mPose, true, reset, mRenderVolume,mEmitterValidateCallback) )
						{
							rigidBodyChange = true;
						}
					}
					else
					{
						if ( erb->mRigidDynamic )
						{
							erb->releaseRigidBody();
							rigidBodyChange = true;
						}
					}
				}
				break;
			case ET_WIND_FS:
				{
					EffectWindFS *ee = static_cast< EffectWindFS *>(ed);
					NxWindFSActor* ea = static_cast< NxWindFSActor*>(ed->getEffectActor());
					// if there is already an emitter actor...
					if (ea)
					{
						if (alive)   // is it alive?
						{
							if (reset)   // is it time to reset it's condition?
							{
								ea->setEnabled(true);
							}
							if ( ee->activePath() )
							{
								PxTransform pose = mPose;
								ee->getSamplePoseSpline(pose);
								ea->setCurrentPose(pose);
								ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
							}
						}
						else
						{
							if (mNotVisibleTime > mData->LODSettings.NonVisibleDeleteTime)   // if it's been non-visible for a long time; delete it, don't just disable it!
							{
								if (ed->getEffectActor())
								{
									ed->releaseActor();
								}
							}
							else
							{
								ea->setEnabled(false);
							}
						}
					}
					else
					{
						if (alive)   // if it is now alive but was not previously; start the initial instance.
						{
							ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
						}
					}
				}
				break;
			case ET_NOISE_FS:
			{
				EffectNoiseFS *ee = static_cast< EffectNoiseFS *>(ed);
				NxNoiseFSActor* ea = static_cast< NxNoiseFSActor*>(ed->getEffectActor());
				// if there is already an emitter actor...
				if (ea)
				{
					if (alive)   // is it alive?
					{
						if (reset)   // is it time to reset it's condition?
						{
							ea->setEnabled(true);
						}
						if ( ee->activePath() )
						{
							PxTransform pose = mPose;
							ee->getSamplePoseSpline(pose);
							ea->setCurrentPose(pose);
							ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
						}
					}
					else
					{
						if (mNotVisibleTime > mData->LODSettings.NonVisibleDeleteTime)   // if it's been non-visible for a long time; delete it, don't just disable it!
						{
							if (ed->getEffectActor())
							{
								ed->releaseActor();
							}
						}
						else
						{
							ea->setEnabled(false);
						}
					}
				}
				else
				{
					if (alive)   // if it is now alive but was not previously; start the initial instance.
					{
						ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
					}
				}
			}
			break;
			case ET_VORTEX_FS:
			{
				EffectVortexFS *ee = static_cast< EffectVortexFS *>(ed);
				NxVortexFSActor* ea = static_cast< NxVortexFSActor*>(ed->getEffectActor());
				// if there is already an emitter actor...
				if (ea)
				{
					if (alive)   // is it alive?
					{
						if (reset)   // is it time to reset it's condition?
						{
							ea->setEnabled(true);
						}
						if ( ee->activePath() )
						{
							PxTransform pose = mPose;
							ee->getSamplePoseSpline(pose);
							ea->setCurrentPose(pose);
							ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
						}
					}
					else
					{
						if (mNotVisibleTime > mData->LODSettings.NonVisibleDeleteTime)   // if it's been non-visible for a long time; delete it, don't just disable it!
						{
							if (ed->getEffectActor())
							{
								ed->releaseActor();
							}
						}
						else
						{
							ea->setEnabled(false);
						}
					}
				}
				else
				{
					if (alive)   // if it is now alive but was not previously; start the initial instance.
					{
						ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
					}
				}
			}
			break;
			case ET_TURBULENCE_FS:
			{
				EffectTurbulenceFS *ee = static_cast< EffectTurbulenceFS *>(ed);
				NxTurbulenceFSActor* ea = static_cast< NxTurbulenceFSActor*>(ed->getEffectActor());
				// if there is already an emitter actor...
				if (ea)
				{
					if (alive)   // is it alive?
					{
						if (reset)   // is it time to reset it's condition?
						{
							ea->setEnabled(true);
						}
						if (mData->LODSettings.FadeTurbulenceNoise)
						{
							// TODO
						}
						if ( ee->activePath() )
						{
							PxTransform pose = mPose;
							ee->getSamplePoseSpline(pose);
							ea->setPose(pose);
							ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
						}
					}
					else
					{
						if (mNotVisibleTime > mData->LODSettings.NonVisibleDeleteTime)   // if it's been non-visible for a long time; delete it, don't just disable it!
						{
							if (ed->getEffectActor())
							{
								ed->releaseActor();
							}
						}
						else
						{
							ea->setEnabled(false);
						}
					}
				}
				else
				{
					if (alive)   // if it is now alive but was not previously; start the initial instance.
					{
						ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
					}
				}
			}
			break;
			case ET_FORCE_FIELD:
			{
				EffectForceField *ee = static_cast< EffectForceField *>(ed);
				NxForceFieldActor* ea = static_cast< NxForceFieldActor*>(ed->getEffectActor());
				// if there is already an emitter actor...
				if (ea)
				{
					if (alive)   // is it alive?
					{
						if (reset)   // is it time to reset it's condition?
						{
							ea->enable();
						}
						if (mData->LODSettings.FadeForceFieldStrength)
						{
							// TODO
						}
						if ( ee->activePath() )
						{
							PxTransform pose = mPose;
							ee->getSamplePoseSpline(pose);
							ea->setPose(pose);
							ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
						}
					}
					else
					{
						if (mNotVisibleTime > mData->LODSettings.NonVisibleDeleteTime)   // if it's been non-visible for a long time; delete it, don't just disable it!
						{
							if (ed->getEffectActor())
							{
								ed->releaseActor();
							}
						}
						else
						{
							if (ea->isEnable())
							{
								ea->disable();
							}
						}
					}
				}
				else
				{
					if (alive)   // if it is now alive but was not previously; start the initial instance.
					{
						ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
					}
				}
			}
			break;
			case ET_HEAT_SOURCE:
			{
				EffectHeatSource* ee = static_cast< EffectHeatSource*>(ed);
				NxHeatSourceActor* ea = static_cast< NxHeatSourceActor*>(ed->getEffectActor());
				// if there is already an emitter actor...
				if (ea)
				{
					if (alive)   // is it alive?
					{
						if (reset)   // is it time to reset it's condition?
						{
							ea->setTemperature(ee->mAverageTemperature, ee->mStandardDeviationTemperature);
							//ea->enable();
							//TODO!
						}
						if (mData->LODSettings.FadeHeatSourceTemperature)
						{
							// TODO
						}
						if ( ee->activePath() )
						{
							PxTransform pose = mPose;
							ee->getSamplePoseSpline(pose);
							ea->setPose(pose);
							ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
						}
					}
					else
					{
						ed->releaseActor();
					}
				}
				else
				{
					if (alive)   // if it is now alive but was not previously; start the initial instance.
					{
						ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
					}
				}
			}
			break;
			case ET_SUBSTANCE_SOURCE:
				{
					EffectSubstanceSource* ee = static_cast< EffectSubstanceSource*>(ed);
					NxSubstanceSourceActor* ea = static_cast< NxSubstanceSourceActor*>(ed->getEffectActor());
					// if there is already an emitter actor...
					if (ea)
					{
						if (alive)   // is it alive?
						{
							if (reset)   // is it time to reset it's condition?
							{
								ea->setDensity(ee->mAverageDensity, ee->mStandardDeviationDensity);
								//ea->enable();
								//TODO!
							}
							if (mData->LODSettings.FadeHeatSourceTemperature)
							{
								// TODO
							}
							if ( ee->activePath() )
							{
								PxTransform pose = mPose;
								ee->getSamplePoseSpline(pose);
								physx::apex::NxApexShape *shape = ea->getShape();
								shape->setPose(pose);
								ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
							}
						}
						else
						{
							ed->releaseActor();
						}
					}
					else
					{
						if (alive)   // if it is now alive but was not previously; start the initial instance.
						{
							ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
						}
					}
				}
				break;
			case ET_VELOCITY_SOURCE:
				{
					EffectVelocitySource* ee = static_cast< EffectVelocitySource*>(ed);
					NxVelocitySourceActor* ea = static_cast< NxVelocitySourceActor*>(ed->getEffectActor());
					// if there is already an emitter actor...
					if (ea)
					{
						if (alive)   // is it alive?
						{
							if (reset)   // is it time to reset it's condition?
							{
								ea->setVelocity(ee->mAverageVelocity, ee->mStandardDeviationVelocity);
								//ea->enable();
								//TODO!
							}
							if (mData->LODSettings.FadeHeatSourceTemperature)
							{
								// TODO
							}
							if ( ee->activePath() )
							{
								PxTransform pose = mPose;
								ee->getSamplePoseSpline(pose);
								ea->setPose(pose);
								ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
							}
						}
						else
						{
							ed->releaseActor();
						}
					}
					else
					{
						if (alive)   // if it is now alive but was not previously; start the initial instance.
						{
							ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
						}
					}
				}
				break;
			case ET_FLAME_EMITTER:
				{
					EffectFlameEmitter* ee = static_cast< EffectFlameEmitter*>(ed);
					NxFlameEmitterActor* ea = static_cast< NxFlameEmitterActor*>(ed->getEffectActor());
					// if there is already an emitter actor...
					if (ea)
					{
						if (alive)   // is it alive?
						{
							if (reset)   // is it time to reset it's condition?
							{
								ea->setEnabled(true);
							}
							if ( ee->activePath() )
							{
								PxTransform pose = mPose;
								ee->getSamplePoseSpline(pose);
								ea->setPose(pose);
								ea->setCurrentScale(ee->mObjectScale*ee->getSampleScaleSpline());
							}
						}
						else
						{
							ed->releaseActor();
						}
					}
					else
					{
						if (alive)   // if it is now alive but was not previously; start the initial instance.
						{
							ed->refresh(mPose, true, false, mRenderVolume,mEmitterValidateCallback);
						}
					}
				}
				break;
			default:
				PX_ALWAYS_ASSERT(); // effect type not handled!
				break;
			}
		}
	}
	if ( rigidBodyChange )
	{
		mRigidBodyChange = true;
	}
	mAlive = anyAlive;
	mFirstFrame = false;
}


/**
\brief Returns the name of the effect at this index.

\param [in] effectIndex : The effect number to refer to; must be less than the result of getEffectCount
*/
const char* EffectPackageActor::getEffectName(PxU32 effectIndex) const
{
	NX_READ_ZONE();
	const char* ret = NULL;
	if (effectIndex < mEffects.size())
	{
		EffectData* ed = mEffects[effectIndex];
		ret = ed->getEffectAsset()->getName();
	}
	return ret;
}

/**
\brief Returns true if this sub-effect is currently enabled.

\param [in] effectIndex : The effect number to refer to; must be less than the result of getEffectCount
*/
bool EffectPackageActor::isEffectEnabled(PxU32 effectIndex) const
{
	NX_READ_ZONE();
	bool ret = false;
	if (effectIndex < mEffects.size())
	{
		EffectData* ed = mEffects[effectIndex];
		ret = ed->isEnabled();
	}
	return ret;
}

/**
\brief Set's the enabled state of this sub-effect

\param [in] effectIndex : The effect number to refer to; must be less than the result of getEffectCount
\param [in] state : Whether the effect should be enabled or not.
*/
bool EffectPackageActor::setEffectEnabled(PxU32 effectIndex, bool state)
{
	NX_WRITE_ZONE();
	bool ret = false;

	if (effectIndex < mEffects.size())
	{
		EffectData* ed = mEffects[effectIndex];
		ed->setEnabled(state);
		if ( ed->getType() == ET_EMITTER )
		{
			ed->setForceRenableEmitter(state); // set the re-enable semaphore
		}
		ret = true;
	}

	return ret;
}

/**
\brief Returns the pose of this sub-effect; returns as a a bool the active state of this effect.

\param [in] effectIndex : The effect number to refer to; must be less than the result of getEffectCount
\param [pose] : Contains the pose requested
\param [worldSpace] : Whether to return the pose in world-space or in parent-relative space.
*/
bool EffectPackageActor::getEffectPose(PxU32 effectIndex, PxTransform& pose, bool worldSpace)
{
	NX_READ_ZONE();
	bool ret = false;

	if (effectIndex < mEffects.size())
	{
		EffectData* ed = mEffects[effectIndex];

		if (worldSpace)
		{
			pose = ed->getWorldPose();
		}
		else
		{
			pose = ed->getLocalPose();
		}
		ret = true;
	}

	return ret;
}

void EffectPackageActor::setCurrentScale(PxF32 scale)
{
	NX_WRITE_ZONE();
	mObjectScale = scale;
	for (physx::PxU32 i = 0; i < mEffects.size(); i++)
	{
		EffectData* ed = mEffects[i];
		ed->setCurrentScale(mObjectScale,mEffectPath);
		ed->refresh(mPose, mEnabled, true, mRenderVolume,mEmitterValidateCallback);
	}
}

/**
\brief Sets the pose of this sub-effect; returns as a a bool the active state of this effect.

\param [in] effectIndex : The effect number to refer to; must be less than the result of getEffectCount
\param [pose] : Contains the pose to be set
\param [worldSpace] : Whether to return the pose in world-space or in parent-relative space.
*/
bool EffectPackageActor::setEffectPose(PxU32 effectIndex, const PxTransform& pose, bool worldSpace)
{
	NX_WRITE_ZONE();
	bool ret = false;

	if (effectIndex < mEffects.size())
	{
		EffectData* ed = mEffects[effectIndex];
		if (worldSpace)
		{
			PxTransform p = getPose(); // get root pose
			PxTransform i = p.getInverse();
			PxTransform l = i * pose;
			ed->setLocalPose(pose);		// change the local pose
			setPose(p);
		}
		else
		{
			ed->setLocalPose(pose);		// change the local pose
			PxTransform p = getPose();
			setPose(p);
		}
		ret = true;
	}
	return ret;
}

/**
\brief Returns the current lifetime of the particle.
*/
PxF32 EffectPackageActor::getCurrentLife(void) const
{
	NX_READ_ZONE();
	return mCurrentLifeTime;
}

PxF32 EffectData::getRealDuration(void) const
{
	PxF32 ret = 0;

	if (mDuration != 0 && mRepeatCount != 9999)   // if it's not an infinite lifespan...
	{
		ret = mInitialDelayTime + mRepeatCount * mDuration + mRepeatCount * mRepeatDelay;
	}

	return ret;
}

PxF32 EffectPackageActor::getDuration(void) const
{
	NX_READ_ZONE();
	PxF32 ret = 0;

	for (physx::PxU32 i = 0; i < mEffects.size(); i++)
	{
		EffectData* ed = mEffects[i];
		if (ed->getType() == ET_EMITTER)   // if it's an emitter
		{
			PxF32 v = ed->getRealDuration();
			if (v > ret)
			{
				ret = v;
			}
		}
	}
	return ret;
}

void	EffectPackageActor::setPreferredRenderVolume(NxApexRenderVolume* volume)
{
	NX_WRITE_ZONE();
	mRenderVolume = volume;
	for (physx::PxU32 i = 0; i < mEffects.size(); i++)
	{
		EffectData* ed = mEffects[i];
		if (ed->getType() == ET_EMITTER)   // if it's an emitter
		{
			EffectEmitter* ee = static_cast< EffectEmitter*>(ed);
			if (ee->getEffectActor())
			{
				NxApexEmitterActor* ea = static_cast< NxApexEmitterActor*>(ee->getEffectActor());
				ea->setPreferredRenderVolume(volume);
			}
		}
	}

}


EffectNoiseFS::EffectNoiseFS(const char* parentName,
                             NoiseFieldSamplerEffect* data,
                             NxApexSDK& sdk,
                             NxApexScene& scene,
                             ParticlesScene& dscene,
                             const PxTransform& rootPose,
                             bool parentEnabled) : mData(data), EffectData(ET_NOISE_FS, &sdk, &scene, &dscene, parentName, NX_NOISE_FS_AUTHORING_TYPE_NAME,*(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties)  )
{
}

EffectNoiseFS::~EffectNoiseFS(void)
{
}

void EffectNoiseFS::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{

}

bool EffectNoiseFS::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();
		physx::PxTransform myPose(parent * localPose);
		getSamplePoseSpline(myPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				bool ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				PX_UNUSED(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				if (mActor)
				{
					NxNoiseFSActor* fs = static_cast< NxNoiseFSActor*>(mActor);
					if (fs)
					{
						fs->setCurrentPose(myPose);
						fs->setCurrentScale(mObjectScale*getSampleScaleSpline());
					}
					ret = true;
				}
			}
		}
		else if (mActor)
		{
			NxNoiseFSActor* a = static_cast< NxNoiseFSActor*>(mActor);
			a->setCurrentPose(myPose);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
		}
	}
	else
	{
		if ( mActor )
		{
			releaseActor();
			ret = true;
		}
	}
	return ret;
}

EffectVortexFS::EffectVortexFS(const char* parentName,
                               VortexFieldSamplerEffect* data,
                               NxApexSDK& sdk,
                               NxApexScene& scene,
                               ParticlesScene& dscene,
                               const PxTransform& rootPose,
                               bool parentEnabled) : mData(data), EffectData(ET_VORTEX_FS, &sdk, &scene, &dscene, parentName, NX_VORTEX_FS_AUTHORING_TYPE_NAME, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
}

EffectVortexFS::~EffectVortexFS(void)
{
}

void EffectVortexFS::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{

}

bool EffectVortexFS::refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;

	if (parentEnabled && mEnabled && mAsset)
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();
		physx::PxTransform myPose(parent * localPose);
		getSamplePoseSpline(myPose);

		if (mActor == NULL && mAsset && !fromSetPose)
		{
			NxParameterized::Interface* descParams = mAsset->getDefaultActorDesc();
			if (descParams)
			{
				bool ok = NxParameterized::setParamF32(*descParams, "initialScale", mObjectScale*getSampleScaleSpline() );
				PX_ASSERT(ok);
				PX_UNUSED(ok);
				mActor = mAsset->createApexActor(*descParams, *mApexScene);
				if (mActor)
				{
					NxVortexFSActor* fs = static_cast< NxVortexFSActor*>(mActor);
					if (fs)
					{
						fs->setCurrentPose(myPose);
						fs->setCurrentScale(mObjectScale*getSampleScaleSpline());
					}
					ret = true;
				}
			}
		}
		else if (mActor)
		{
			NxVortexFSActor* a = static_cast< NxVortexFSActor*>(mActor);
			a->setCurrentPose(myPose);
			a->setCurrentScale(mObjectScale*getSampleScaleSpline());
		}
	}
	else
	{
		if ( mActor )
		{
			releaseActor();
			ret = true;
		}
	}
	return ret;
}

const char * EffectPackageActor::hasVolumeRenderMaterial(physx::PxU32 &index) const
{
	NX_READ_ZONE();
	const char *ret = NULL;

	for (physx::PxU32 i=0; i<mEffects.size(); i++)
	{
		EffectData *d = mEffects[i];
		if ( d->getType() == ET_TURBULENCE_FS )
		{
			const NxParameterized::Interface *iface = d->getAsset()->getAssetNxParameterized();
			const turbulencefs::TurbulenceFSAssetParams *ap = static_cast< const turbulencefs::TurbulenceFSAssetParams *>(iface);
			if ( ap->volumeRenderMaterialName )
			{
				if ( strlen(ap->volumeRenderMaterialName->name()) > 0 )
				{
					index = i;
					ret = ap->volumeRenderMaterialName->name();
					break;
				}
			}
		}
	}

	return ret;
}

EffectRigidBody::EffectRigidBody(const char* parentName,
	RigidBodyEffect* data,
	NxApexSDK& sdk,
	NxApexScene& scene,
	ParticlesScene& dscene,
	const PxTransform& rootPose,
	bool parentEnabled) : mData(data), EffectData(ET_RIGID_BODY, &sdk, &scene, &dscene, parentName, NULL, *(RigidBodyEffectNS::EffectProperties_Type *)(&data->EffectProperties))
{
	mRigidDynamic = NULL;
}

EffectRigidBody::~EffectRigidBody(void)
{
	releaseRigidBody();
}


void EffectRigidBody::visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const
{

}

bool EffectRigidBody::refresh(const PxTransform& parent,
							  bool parentEnabled,
							  bool fromSetPose,
							  NxApexRenderVolume* renderVolume,
							  NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
{
	bool ret = false;
#if NX_SDK_VERSION_MAJOR == 3
	SCOPED_PHYSX_LOCK_WRITE(*mApexScene);
	if (parentEnabled && mEnabled )
	{
		PxTransform localPose = mLocalPose;
		localPose.p*=mObjectScale*getSampleScaleSpline();
		physx::PxTransform myPose = parent * localPose;
		getSamplePoseSpline(myPose);

		if (mRigidDynamic == NULL && !fromSetPose)
		{
			PxScene * scene = mApexScene->getPhysXScene();
			PxPhysics &sdk = scene->getPhysics();
			PxMaterial * material = mParticlesScene->getModuleParticles()->getDefaultMaterial();
			RigidBodyEffect *rbe = static_cast< RigidBodyEffect *>(mData);
			physx::PxFilterData data = ApexResourceHelper::resolveCollisionGroup128(rbe->CollisionFilterDataName.buf);
			mRigidDynamic = sdk.createRigidDynamic(myPose);
			ret = true;
			if ( mRigidDynamic )
			{
				mRigidDynamic->setLinearVelocity(rbe->InitialLinearVelocity);
				mRigidDynamic->setAngularVelocity(rbe->InitialAngularVelocity);
				mRigidDynamic->setMass(rbe->Mass);
				mRigidDynamic->setLinearDamping(rbe->LinearDamping);
				mRigidDynamic->setAngularDamping(rbe->AngularDamping);
				mRigidDynamic->setRigidDynamicFlag(PxRigidBodyFlag::eKINEMATIC,!rbe->Dynamic);

				physx::PxShape *shape = NULL;
				if ( strcmp(rbe->Type,"SPHERE") == 0 )
				{
					PxSphereGeometry sphere;
					sphere.radius = rbe->Extents.x*getSampleScaleSpline();
					shape = mRigidDynamic->createShape(sphere,*material,localPose);
				}
				else if ( strcmp(rbe->Type,"CAPSULE") == 0 )
				{
					PxCapsuleGeometry capsule;
					capsule.radius = rbe->Extents.x*getSampleScaleSpline();
					capsule.halfHeight = rbe->Extents.y*getSampleScaleSpline();
					shape = mRigidDynamic->createShape(capsule,*material,localPose);
				}
				else if ( strcmp(rbe->Type,"BOX") == 0 )
				{
					PxBoxGeometry box;
					box.halfExtents.x = rbe->Extents.x*0.5f*getSampleScaleSpline();
					box.halfExtents.y = rbe->Extents.y*0.5f*getSampleScaleSpline();
					box.halfExtents.z = rbe->Extents.z*0.5f*getSampleScaleSpline();
					shape = mRigidDynamic->createShape(box,*material,localPose);
				}
				else
				{
					PX_ALWAYS_ASSERT();
				}
				if ( shape )
				{
					// do stuff here...
					shape->setSimulationFilterData(data);
					shape->setQueryFilterData(data);
					shape->setFlag(PxShapeFlag::eSIMULATION_SHAPE,true);
					shape->setMaterials(&material,1);
				}
				mRigidDynamic->setActorFlag(PxActorFlag::eDISABLE_GRAVITY,!rbe->Gravity);
				mRigidDynamic->setActorFlag(PxActorFlag::eDISABLE_SIMULATION,false);
				mRigidDynamic->setActorFlag(PxActorFlag::eVISUALIZATION,true);
				scene->addActor(*mRigidDynamic);
			}
		}
		else if (mRigidDynamic && fromSetPose )
		{
			if ( mRigidDynamic->getRigidDynamicFlags() & PxRigidDynamicFlag::eKINEMATIC )
			{
				mRigidDynamic->setKinematicTarget(myPose);
			}
			else
			{
				mRigidDynamic->setGlobalPose(myPose);
			}
		}
		if ( activePath() && mRigidDynamic )
		{
			// if we are sampling a spline curve to control the scale of the object..
			RigidBodyEffect *rbe = static_cast< RigidBodyEffect *>(mData);
			physx::PxShape *shape = NULL;
			mRigidDynamic->getShapes(&shape,1,0);
			if ( shape )
			{
				if ( strcmp(rbe->Type,"SPHERE") == 0 )
				{
					PxSphereGeometry sphere;
					sphere.radius = rbe->Extents.x*getSampleScaleSpline();
					shape->setGeometry(sphere);
				}
				else if ( strcmp(rbe->Type,"CAPSULE") == 0 )
				{
					PxCapsuleGeometry capsule;
					capsule.radius = rbe->Extents.x*getSampleScaleSpline();
					capsule.halfHeight = rbe->Extents.y*getSampleScaleSpline();
					shape->setGeometry(capsule);
				}
				else if ( strcmp(rbe->Type,"BOX") == 0 )
				{
					PxBoxGeometry box;
					box.halfExtents.x = rbe->Extents.x*0.5f*getSampleScaleSpline();
					box.halfExtents.y = rbe->Extents.y*0.5f*getSampleScaleSpline();
					box.halfExtents.z = rbe->Extents.z*0.5f*getSampleScaleSpline();
					shape->setGeometry(box);
				}
			}
			if ( mRigidDynamic->getRigidDynamicFlags() & PxRigidDynamicFlag::eKINEMATIC )
			{
				mRigidDynamic->setKinematicTarget(myPose);
			}
			else
			{
				mRigidDynamic->setGlobalPose(myPose);
			}
		}
	}
	else
	{
		//releaseActor();
	}
#endif
	return ret;
}

PxRigidDynamic* EffectPackageActor::getEffectRigidDynamic(PxU32 effectIndex) const
{
	NX_READ_ZONE();
	PxRigidDynamic *ret = NULL;

	if (effectIndex < mEffects.size())
	{
		EffectData* ed = mEffects[effectIndex];
		if ( ed->getType() == ET_RIGID_BODY )
		{
			EffectRigidBody *erd = static_cast< EffectRigidBody *>(ed);
			ret = erd->mRigidDynamic;
		}
	}


	return ret;
}

void EffectRigidBody::releaseRigidBody(void)
{
#if NX_SDK_VERSION_MAJOR == 3
	if ( mRigidDynamic )
	{
		SCOPED_PHYSX_LOCK_WRITE(*mApexScene);
		mRigidDynamic->release();
		mRigidDynamic = NULL;
	}
#endif
}

EffectPath::EffectPath(void)
{
	mRotations = NULL;
	mPathSpline = NULL;
	mScaleSpline = NULL;
	mSampleScaleSpline = 1.0f;
	mSpeedSpline = NULL;
	mPathDuration = 1;
}

EffectPath::~EffectPath(void)
{
	delete mScaleSpline;
	delete mSpeedSpline;
	delete	mPathSpline;
	PX_FREE(mRotations);
}

bool EffectPath::init(RigidBodyEffectNS::EffectPath_Type &path)
{
	bool ret = path.Scale.arraySizes[0] > 2;

	mPathDuration = path.PathDuration;
	mMode = EM_LOOP;
	if ( strcmp(path.PlaybackMode,"LOOP") == 0 )
	{
		mMode = EM_LOOP;
	}
	else if ( strcmp(path.PlaybackMode,"PLAY_ONCE") == 0 )
	{
		mMode = EM_PLAY_ONCE;
	}
	else if ( strcmp(path.PlaybackMode,"PING_PONG") == 0 )
	{
		mMode = EM_PING_PONG;
	}

	delete mScaleSpline;
	mScaleSpline = NULL;

	if ( path.Scale.arraySizes[0] == 2 )
	{
		if ( path.Scale.buf[0].y != 1 ||
			path.Scale.buf[1].y != 1 )
		{
			ret = true;
		}
	}
	else if ( path.Scale.arraySizes[0] > 2 )
	{
		ret = true;
	}

	if ( ret )
	{
		mScaleSpline = PX_NEW(Spline);
		PxI32 scount = path.Scale.arraySizes[0];
		mScaleSpline->Reserve(scount);
		for (PxI32 i=0; i<scount; i++)
		{
			PxF32 x = path.Scale.buf[i].x;
			PxF32 y = path.Scale.buf[i].y;
			mScaleSpline->AddNode(x,y);
		}
		mScaleSpline->ComputeSpline();
		physx::PxU32 index;
		physx::PxF32 t;
		mSampleScaleSpline = mScaleSpline->Evaluate(0,index,t);
	}


	bool hasSpeed = false;

	if ( path.Speed.arraySizes[0] == 2 )
	{
		if ( path.Speed.buf[0].y != 1 ||
			path.Speed.buf[1].y != 1 )
		{
			hasSpeed = true;
		}
	}
	else if ( path.Speed.arraySizes[0] > 2 )
	{
		hasSpeed = true;
	}

	delete mSpeedSpline;
	mSpeedSpline = NULL;

	if ( hasSpeed )
	{
		ret = true;

		Spline speed;
		PxI32 scount = path.Speed.arraySizes[0];
		speed.Reserve(scount);
		for (PxI32 i=0; i<scount; i++)
		{
			PxF32 x = path.Speed.buf[i].x;
			PxF32 y = path.Speed.buf[i].y;
			speed.AddNode(x,y);
		}
		speed.ComputeSpline();

		PxF32 distance = 0;
		PxU32 index;
		for (PxI32 i=0; i<32; i++)
		{
			PxF32 t = (PxF32)i/32.0f;
			PxF32 fraction;
			PxF32 dt = speed.Evaluate(t,index,fraction);
			distance+=dt;
		}
		PxF32 recipDistance = 1.0f / distance;
		mSpeedSpline = PX_NEW(Spline);
		mSpeedSpline->Reserve(32);
		distance = 0;
		for (PxI32 i=0; i<32; i++)
		{
			PxF32 t = (PxF32)i/32.0f;
			PxF32 fraction;
			PxF32 dt = speed.Evaluate(t,index,fraction);
			distance+=dt;
			PxF32 d = distance*recipDistance;
			mSpeedSpline->AddNode(t,d);
		}
		mSpeedSpline->ComputeSpline();
		PxF32 fraction;
		mSampleSpeedSpline = mSpeedSpline->Evaluate(0,index,fraction);
	}


	PX_FREE(mRotations);
	mRotations = NULL;
	delete mPathSpline;
	mPathSpline = NULL;

	if ( path.ControlPoints.arraySizes[0] >= 3 )
	{
		mPathSpline = PX_NEW(SplineCurve);
		PxI32 count = path.ControlPoints.arraySizes[0];
		PxVec3Vector points;
		mRotationCount = (physx::PxU32)count-1;
		mRotations = (PxQuat *)PX_ALLOC(sizeof(PxQuat)*(count-1),"PathRotations");
		mPathRoot = path.ControlPoints.buf[0];
		for (PxI32 i=1; i<count; i++)
		{
			const PxTransform &t = path.ControlPoints.buf[i];
			mRotations[i-1] = t.q;
			points.pushBack(t.p);
		}
		mPathSpline->setControlPoints(points);

		physx::PxF32 fraction;
		physx::PxU32 index;
		mSamplePoseSpline.p = mPathSpline->Evaluate(0,index,fraction);
		mSamplePoseSpline.q = physx::PxQuat(physx::PxIdentity);
		mSamplePoseSpline = mPathRoot * mSamplePoseSpline;
		ret = true;
	}



	return ret;
}

PxF32 EffectPath::sampleSpline(PxF32 x)
{
	if ( mScaleSpline )
	{
		PxU32 index;
		PxF32 fraction;
		mSampleScaleSpline = mScaleSpline->Evaluate(x,index,fraction);
		if ( mSampleScaleSpline < 0.001f )
		{
			mSampleScaleSpline = 0.001f;
		}
	}
	if ( mPathSpline )
	{
		PxU32 index;
		PxF32 fraction;
		if ( mSpeedSpline )
		{
			x = mSpeedSpline->Evaluate(x,index,fraction);
		}
		PxF32 duration = mPathSpline->GetLength();
		duration*=x;
		mSamplePoseSpline.p = mPathSpline->Evaluate(duration,index,fraction);
		PX_ASSERT( index < mRotationCount );
		physx::PxQuat q0 = mRotations[index];
		PxU32 index2 = index+1;
		if ( index2 >= mRotationCount )
		{
			index2 = 0;
		}
		physx::PxQuat q1 = mRotations[index2];

		physx::PxQuat q = physx::shdfnd::slerp(fraction,q0,q1);

		mSamplePoseSpline.q = q;
		mSamplePoseSpline = mPathRoot * mSamplePoseSpline;
	}

	return mSampleScaleSpline;
}

void EffectPath::computeSampleTime(PxF32 ctime,PxF32 duration)
{
	PxF32 sampleTime=0;
	switch ( mMode )
	{
		case EM_PLAY_ONCE:
			if ( ctime >= duration )
			{
				sampleTime = 1;
			}
			else
			{
				sampleTime = ctime / duration;
			}
			break;
		case EM_LOOP:
			sampleTime = fmodf(ctime,duration) / duration;
			break;
		case EM_PING_PONG:
			sampleTime = fmodf(ctime,duration*2) / duration;
			if ( sampleTime > 1 )
			{
				sampleTime = 2.0f - sampleTime;
			}
			break;
		default:
			PX_ALWAYS_ASSERT();
			break;
	}
	sampleSpline(sampleTime);
}

} // end of particles namespace
} // end of apex namespace
} // end of physx namespace
