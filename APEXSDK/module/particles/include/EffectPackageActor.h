/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef PARTICLES_EFFECT_PACKAGE_ACTOR_H

#define PARTICLES_EFFECT_PACKAGE_ACTOR_H

#include "ApexActor.h"
#include "NxEffectPackageActor.h"
#include "EffectPackageAsset.h"
#include "ParticlesBase.h"
#include "Spline.h"
#include "foundation/PxTransform.h"
#include "ReadCheck.h"
#include "WriteCheck.h"

namespace physx
{
namespace apex
{

class NxApexScene;

namespace turbulencefs
{
class NxModuleTurbulenceFS;
}

namespace particles
{

enum VisState
{
	VS_TOO_CLOSE,
	VS_ON_SCREEN,
	VS_BEHIND_SCREEN,
	VS_OFF_SCREEN,
};

class EffectPath : public physx::shdfnd::UserAllocated
{
public:
	enum Mode
	{
		EM_LOOP,
		EM_PLAY_ONCE,
		EM_PING_PONG
	};
	EffectPath(void);
	~EffectPath(void);

	bool init(RigidBodyEffectNS::EffectPath_Type &path);
	PxF32 getSampleScaleSpline(void) const { return mSampleScaleSpline; };
	PxF32 getSampleSpeedSpline(void) const { return mSampleSpeedSpline; };

	void getSamplePoseSpline(PxTransform &pose)
	{
		if ( mPathSpline )
		{
			pose = pose * mSamplePoseSpline;
		}
	}

	PxF32 sampleSpline(PxF32 stime);

	Mode getMode(void) const
	{
		return mMode;
	}

	void computeSampleTime(PxF32 ctime,PxF32 duration);

	PxF32 getPathDuration(void)
	{
		return mPathDuration;
	}

private:
	Mode					mMode;
	PxF32					mPathDuration;
	Spline					*mScaleSpline;
	PxF32					mSampleScaleSpline; // the scale value sampled from the spline curve
	PxF32					mSampleSpeedSpline;
	Spline					*mSpeedSpline;
	physx::PxTransform		mPathRoot;
	physx::PxTransform		mSamplePoseSpline;
	physx::PxU32			mRotationCount;
	physx::PxQuat			*mRotations;
	SplineCurve				*mPathSpline;
};

class EffectData : public physx::shdfnd::UserAllocated
{
public:
	enum EffectState
	{
		ES_INITIAL_DELAY,
		ES_ACTIVE,
		ES_REPEAT_DELAY,
		ES_DONE
	};

	EffectData(EffectType type,
	           NxApexSDK* sdk,
	           NxApexScene* scene,
	           ParticlesScene* dscene,
	           const char* assetName,
	           const char* nameSpace,
			   RigidBodyEffectNS::EffectProperties_Type &effectProperties);

	virtual ~EffectData(void);
	void releaseActor(void);

	PxF32 getRandomTime(PxF32 baseTime);

	EffectType getType(void) const
	{
		return mType;
	}
	virtual void release(void) = 0;
	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const = 0;
	virtual bool refresh(const PxTransform& parent,
						bool parentEnabled,
						bool fromSetPose,
						NxApexRenderVolume* renderVolume,
						NxApexEmitterActor::NxApexEmitterValidateCallback *callback) = 0;

	bool isDead(void) const
	{
		return mState == ES_DONE;
	}

	NxApexActor* getEffectActor(void) const
	{
		return mActor;
	}
	NxApexAsset* getEffectAsset(void) const
	{
		return mAsset;
	}

	bool isEnabled(void) const
	{
		return mEnabled;
	}

	void setEnabled(bool state)
	{
		mEnabled = state;
	}

	bool simulate(PxF32 dtime, bool& reset);

	PxU32 getRepeatCount(void) const
	{
		return mRepeatCount;
	};
	PxF32 getDuration(void) const
	{
		return mDuration;
	};

	PxF32 getRealDuration(void) const;

	void setLocalPose(const PxTransform& p)
	{
		mLocalPose = p;
	}
	const PxTransform& getWorldPose(void) const
	{
		return mPose;
	};
	const PxTransform& getLocalPose(void) const
	{
		return mLocalPose;
	};

	NxApexAsset * getAsset(void) const { return mAsset; };

	void setForceRenableEmitter(bool state)
	{
		mForceRenableEmitter = state;
	}

	bool getForceRenableEmitterSemaphore(void)
	{
		bool ret = mForceRenableEmitter;
		mForceRenableEmitter = false;
		return ret;
	}

	void setCurrentScale(PxF32 objectScale,EffectPath *parentPath)
	{
		mParentPath = parentPath;
		mObjectScale = objectScale;
	}

	physx::apex::NxApexScene *getApexScene(void) const
	{
		return mApexScene;
	}

	void getSamplePoseSpline(physx::PxTransform &pose)
	{
		if ( mParentPath )
		{
			mParentPath->getSamplePoseSpline(pose);
		}
		if ( mEffectPath )
		{
			mEffectPath->getSamplePoseSpline(pose);
		}
	}

	PxF32 getSampleScaleSpline(void) const
	{
		PxF32 parentScale = mParentPath ? mParentPath->getSampleScaleSpline() : 1;
		PxF32 myScale = mEffectPath ? mEffectPath->getSampleScaleSpline() : 1;
		return myScale*parentScale;
	}

	bool activePath(void) const
	{
		bool ret = false;
		if ( mEffectPath || mParentPath )
		{
			ret = true;
		}
		return ret;
	}

	bool					mFirstRate: 1;
	PxF32					mObjectScale;
	EffectPath				*mParentPath;
	EffectPath				*mEffectPath;

protected:
	bool					mUseEmitterPool: 1;
	bool					mEnabled: 1;
	bool					mForceRenableEmitter:1;
	EffectState				mState;
	PxF32					mRandomDeviation;
	PxF32					mSimulationTime;
	PxF32					mStateTime;
	PxU32					mStateCount;
	const char*				mNameSpace;
	ParticlesScene*			mParticlesScene;
	NxApexScene*			mApexScene;
	NxApexSDK*				mApexSDK;
	NxApexAsset*			mAsset;
	NxApexActor*			mActor;
	PxF32					mInitialDelayTime;
	PxF32					mDuration;
	PxU32					mRepeatCount;
	PxF32					mRepeatDelay;
	EffectType				mType;
	PxTransform				mPose;				// world space pose
	PxTransform				mLocalPose;			// local space pose
};

class EffectForceField : public EffectData
{
public:
	EffectForceField(const char* parentName,
	                 ForceFieldEffect* data,
	                 NxApexSDK& sdk,
	                 NxApexScene& scene,
	                 ParticlesScene& dscene,
	                 const PxTransform& rootPose,
	                 bool parentEnabled);

	virtual ~EffectForceField(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPos, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);

	ForceFieldEffect*		mData;
};



class EffectEmitter : public EffectData
{
public:
	EffectEmitter(const char* parentName,
	              const EmitterEffect* data,
	              NxApexSDK& sdk,
	              NxApexScene& scene,
	              ParticlesScene& dscene,
	              const PxTransform& rootPose,
	              bool parentEnabled);

	virtual ~EffectEmitter(void);

	virtual void release(void)
	{
		delete this;
	}
	void computeVelocity(physx::PxF32 dtime);
	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;

	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);

	NxRange<physx::PxF32> mRateRange;
	NxRange<physx::PxF32> mLifetimeRange;
	const EmitterEffect*		mData;
	bool					mFirstVelocityFrame: 1;
	bool					mHaveSetPosition;
	physx::PxVec3			mLastEmitterPosition;
	physx::PxF32			mVelocityTime;
	physx::PxVec3			mEmitterVelocity;
};

class EffectHeatSource : public EffectData
{
public:
	EffectHeatSource(const char* parentName,
	                 HeatSourceEffect* data,
	                 NxApexSDK& sdk,
	                 NxApexScene& scene,
	                 ParticlesScene& dscene,
					 NxModuleTurbulenceFS* moduleTurbulenceFS,
	                 const PxTransform& rootPose,
	                 bool parentEnabled);

	virtual ~EffectHeatSource(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);


	physx::PxF32			mAverageTemperature;
	physx::PxF32			mStandardDeviationTemperature;

	NxModuleTurbulenceFS*	mModuleTurbulenceFS;
	HeatSourceEffect*		mData;
};

class EffectSubstanceSource : public EffectData
{
public:
	EffectSubstanceSource(const char* parentName,
		SubstanceSourceEffect* data,
		NxApexSDK& sdk,
		NxApexScene& scene,
		ParticlesScene& dscene,
		NxModuleTurbulenceFS* moduleTurbulenceFS,
		const PxTransform& rootPose,
		bool parentEnabled);

	virtual ~EffectSubstanceSource(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);


	physx::PxF32			mAverageDensity;
	physx::PxF32			mStandardDeviationDensity;

	NxModuleTurbulenceFS*	mModuleTurbulenceFS;
	SubstanceSourceEffect*		mData;
};

class EffectVelocitySource : public EffectData
{
public:
	EffectVelocitySource(const char* parentName,
		VelocitySourceEffect* data,
		NxApexSDK& sdk,
		NxApexScene& scene,
		ParticlesScene& dscene,
		NxModuleTurbulenceFS* moduleTurbulenceFS,
		const PxTransform& rootPose,
		bool parentEnabled);

	virtual ~EffectVelocitySource(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);


	physx::PxF32			mAverageVelocity;
	physx::PxF32			mStandardDeviationVelocity;
	NxModuleTurbulenceFS*	mModuleTurbulenceFS;
	VelocitySourceEffect*		mData;
};

class EffectFlameEmitter : public EffectData
{
public:
	EffectFlameEmitter(const char* parentName,
		FlameEmitterEffect* data,
		NxApexSDK& sdk,
		NxApexScene& scene,
		ParticlesScene& dscene,
		NxModuleTurbulenceFS* moduleTurbulenceFS,
		const PxTransform& rootPose,
		bool parentEnabled);

	virtual ~EffectFlameEmitter(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);


	NxModuleTurbulenceFS*	mModuleTurbulenceFS;
	FlameEmitterEffect*		mData;
};


class EffectTurbulenceFS : public EffectData
{
public:
	EffectTurbulenceFS(const char* parentName,
	                   TurbulenceFieldSamplerEffect* data,
	                   NxApexSDK& sdk,
	                   NxApexScene& scene,
	                   ParticlesScene& dscene,
					   NxModuleTurbulenceFS* moduleTurbulenceFS,
	                   const PxTransform& rootPose,
	                   bool parentEnabled);

	virtual ~EffectTurbulenceFS(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);

	NxModuleTurbulenceFS*	mModuleTurbulenceFS;
	TurbulenceFieldSamplerEffect*	mData;
};

class EffectJetFS : public EffectData
{
public:
	EffectJetFS(const char* parentName,
	            JetFieldSamplerEffect* data,
	            NxApexSDK& sdk,
	            NxApexScene& scene,
	            ParticlesScene& dscene,
	            const PxTransform& rootPose,
	            bool parentEnabled);
	virtual ~EffectJetFS(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);

	JetFieldSamplerEffect*	mData;
};


class EffectWindFS : public EffectData
{
public:
	EffectWindFS(const char* parentName,
		WindFieldSamplerEffect* data,
		NxApexSDK& sdk,
		NxApexScene& scene,
		ParticlesScene& dscene,
		const PxTransform& rootPose,
		bool parentEnabled);
	virtual ~EffectWindFS(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);

	WindFieldSamplerEffect*	mData;
};

class EffectRigidBody : public EffectData
{
public:
	EffectRigidBody(const char* parentName,
		RigidBodyEffect* data,
		NxApexSDK& sdk,
		NxApexScene& scene,
		ParticlesScene& dscene,
		const PxTransform& rootPose,
		bool parentEnabled);
	virtual ~EffectRigidBody(void);

	virtual void release(void)
	{
		delete this;
	}

	void releaseRigidBody(void);

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);

	RigidBodyEffect*	mData;
	PxRigidDynamic		*mRigidDynamic;
};

class EffectNoiseFS : public EffectData
{
public:
	EffectNoiseFS(const char* parentName,
	              NoiseFieldSamplerEffect* data,
	              NxApexSDK& sdk,
	              NxApexScene& scene,
	              ParticlesScene& dscene,
	              const PxTransform& rootPose,
	              bool parentEnabled);
	virtual ~EffectNoiseFS(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);

	NoiseFieldSamplerEffect*	mData;
};


class EffectVortexFS : public EffectData
{
public:
	EffectVortexFS(const char* parentName,
	               VortexFieldSamplerEffect* data,
	               NxApexSDK& sdk,
	               NxApexScene& scene,
	               ParticlesScene& dscene,
	               const PxTransform& rootPose,
	               bool parentEnabled);
	virtual ~EffectVortexFS(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);

	VortexFieldSamplerEffect*	mData;
};



class EffectAttractorFS : public EffectData
{
public:
	EffectAttractorFS(const char* parentName,
	                  AttractorFieldSamplerEffect* data,
	                  NxApexSDK& sdk,
	                  NxApexScene& scene,
	                  ParticlesScene& dscene,
	                  const PxTransform& rootPose,
	                  bool parentEnabled);

	virtual ~EffectAttractorFS(void);

	virtual void release(void)
	{
		delete this;
	}

	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;
	virtual bool refresh(const PxTransform& parent, bool parentEnabled, bool fromSetPose, NxApexRenderVolume* renderVolume,NxApexEmitterActor::NxApexEmitterValidateCallback *callback);

	AttractorFieldSamplerEffect*		mData;
};


class EffectPackageActor : public NxEffectPackageActor, public physx::shdfnd::UserAllocated, public ParticlesBase, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	EffectPackageActor(NxEffectPackageAsset* asset,
	                   const EffectPackageAssetParams* assetParams,
	                   const EffectPackageActorParams* actorParams,
	                   physx::apex::NxApexSDK& sdk,
	                   physx::apex::NxApexScene& scene,
	                   ParticlesScene& dynamicSystemScene,
					   NxModuleTurbulenceFS* moduleTurbulenceFS);

	virtual ~EffectPackageActor(void);

	virtual ParticlesType getParticlesType(void) const
	{
		return ParticlesBase::DST_EFFECT_PACKAGE_ACTOR;
	}

	void updateParticles(PxF32 dtime);
	void updatePoseAndBounds(bool screenCulling, bool znegative);

	virtual void setPose(const physx::PxTransform& pose);
	virtual const PxTransform& getPose(void) const;
	virtual void visualize(physx::general_renderdebug4::RenderDebug* callback, bool solid) const;

	virtual void refresh(void);
	virtual void release(void);

	virtual const char* getName(void) const;

	virtual PxU32 getEffectCount(void) const; // returns the number of effects in the effect package
	virtual EffectType getEffectType(PxU32 effectIndex) const; // return the type of effect.
	virtual NxApexActor* getEffectActor(PxU32 effectIndex) const; // return the base NxApexActor pointer
	virtual void setEmitterState(bool state); // set the state for all emitters in this effect package.
	virtual PxU32 getActiveParticleCount(void) const; // return the total number of particles still active in this effect package.
	virtual bool isStillEmitting(void) const; // return true if any emitters are still actively emitting particles.


	/**
	\brief Returns the name of the effect at this index.

	\param [in] effectIndex : The effect number to refer to; must be less than the result of getEffectCount
	*/
	virtual const char* getEffectName(PxU32 effectIndex) const;

	/**
	\brief Returns true if this sub-effect is currently enabled.

	\param [in] effectIndex : The effect number to refer to; must be less than the result of getEffectCount
	*/
	virtual bool isEffectEnabled(PxU32 effectIndex) const;

	/**
	\brief Set's the enabled state of this sub-effect

	\param [in] effectIndex : The effect number to refer to; must be less than the result of getEffectCount
	\param [in] state : Whether the effect should be enabled or not.
	*/
	virtual bool setEffectEnabled(PxU32 effectIndex, bool state);

	/**
	\brief Returns the pose of this sub-effect; returns as a a bool the active state of this effect.

	\param [in] effectIndex : The effect number to refer to; must be less than the result of getEffectCount
	\param [pose] : Contains the pose requested
	\param [worldSpace] : Whether to return the pose in world-space or in parent-relative space.
	*/
	virtual bool getEffectPose(PxU32 effectIndex, PxTransform& pose, bool worldSpace);

	/**
	\brief Sets the pose of this sub-effect; returns as a a bool the active state of this effect.

	\param [in] effectIndex : The effect number to refer to; must be less than the result of getEffectCount
	\param [pose] : Contains the pose to be set
	\param [worldSpace] : Whether to return the pose in world-space or in parent-relative space.
	*/
	virtual bool setEffectPose(PxU32 effectIndex, const PxTransform& pose, bool worldSpace);

	virtual void setCurrentScale(PxF32 scale);

	virtual PxF32 getCurrentScale(void) const
	{
		return mObjectScale;
	}

	virtual PxRigidDynamic* getEffectRigidDynamic(PxU32 effectIndex) const;

	/**
	\brief Returns the current lifetime of the particle.
	*/
	virtual PxF32 getCurrentLife(void) const;


	virtual PxF32 getDuration(void) const;

	/**
	\brief Returns the owning asset
	*/
	virtual NxApexAsset* getOwner() const
	{
		NX_READ_ZONE();
		return mAsset;
	}

	/**
	\brief Returns the range of possible values for physical Lod overwrite

	\param [out] min		The minimum lod value
	\param [out] max		The maximum lod value
	\param [out] intOnly	Only integers are allowed if this is true, gets rounded to nearest

	\note The max value can change with different graphical Lods
	\see NxApexActor::forcePhysicalLod()
	*/
	virtual void getPhysicalLodRange(physx::PxF32& min, physx::PxF32& max, bool& intOnly) const
	{
		NX_READ_ZONE();
		min = 0;
		max = 100000;
		intOnly = false;
	}

	/**
	\brief Get current physical lod.
	*/
	virtual physx::PxF32 getActivePhysicalLod() const
	{
		NX_READ_ZONE();
		return 0;
	}

	/**
	\brief Force an APEX Actor to use a certian physical Lod

	\param [in] lod	Overwrite the Lod system to use this Lod.

	\note Setting the lod value to a negative number will turn off the overwrite and proceed with regular Lod computations
	\see NxApexActor::getPhysicalLodRange()
	*/
	virtual void forcePhysicalLod(physx::PxF32 lod)
	{
		NX_WRITE_ZONE();
		PX_UNUSED(lod);
	}

	/**
	\brief Selectively enables/disables debug visualization of a specific APEX actor.  Default value it true.
	*/
	virtual void setEnableDebugVisualization(bool state)
	{
		NX_WRITE_ZONE();
		ApexActor::setEnableDebugVisualization(state);
	}

	/**
	\brief Ensure that all module-cached data is cached.
	*/
	virtual void cacheModuleData() const
	{

	}

	virtual void setEnabled(bool state)
	{
		NX_WRITE_ZONE();
		mEnabled = state;
		refresh();
	}

	bool getEnabled(void) const
	{
		NX_READ_ZONE();
		return mEnabled;
	}

#if NX_SDK_VERSION_MAJOR == 2
	// NxScene pointer may be NULL
	virtual void		setPhysXScene(NxScene* s)
	{
		mPhysXScene = s;
	}

	virtual NxScene*	getPhysXScene() const
	{
		return mPhysXScene;
	}
#elif NX_SDK_VERSION_MAJOR == 3
	virtual void		setPhysXScene(PxScene* s)
	{
		mPhysXScene = s;
	}
	virtual PxScene*	getPhysXScene() const
	{
		return mPhysXScene;
	}
#endif

	physx::PxF32 internalGetDuration(void);

	virtual bool isAlive(void) const
	{
		NX_READ_ZONE();
		return mAlive;
	}

	virtual void fadeOut(physx::PxF32 fadeTime)
	{
		NX_WRITE_ZONE();
		if (!mFadeOut)
		{
			mFadeOutTime = fadeTime;
			mFadeOutDuration = 0;

			if (mFadeIn)
			{
				PxF32 fadeLerp = mFadeInDuration / mFadeInTime;
				if (fadeLerp > 1)
				{
					fadeLerp = 1;
				}
				mFadeOutDuration = 1 - (fadeLerp * mFadeOutTime);
				mFadeIn = false;
			}

			mFadeOut = true;
		}
	}

	virtual void fadeIn(physx::PxF32 fadeTime)
	{
		NX_WRITE_ZONE();
		if (!mFadeIn)
		{
			mFadeInTime = fadeTime;
			mFadeInDuration = 0;
			mFadeIn = true;
			if (mFadeOut)
			{
				PxF32 fadeLerp = mFadeOutDuration / mFadeOutTime;
				if (fadeLerp > 1)
				{
					fadeLerp = 1;
				}
				mFadeInDuration = 1 - (fadeLerp * mFadeInTime);
				mFadeOut = false;
			}
		}
	}

	virtual void                 setPreferredRenderVolume(NxApexRenderVolume* volume);

	virtual const char * hasVolumeRenderMaterial(physx::PxU32 &index) const;

	virtual void setApexEmitterValidateCallback(NxApexEmitterActor::NxApexEmitterValidateCallback *callback)
	{
		NX_WRITE_ZONE();
		mEmitterValidateCallback = callback;
	}

	PxF32 getSampleScaleSpline(void) const
	{
		return mEffectPath ? mEffectPath->getSampleScaleSpline() : 1;
	}

	void getSamplePoseSpline(PxTransform &pose)
	{
		if ( mEffectPath )
		{
			mEffectPath->getSamplePoseSpline(pose);
		}
	}

private:

#if NX_SDK_VERSION_MAJOR == 2
	NxScene*					mPhysXScene;
#elif NX_SDK_VERSION_MAJOR == 3
	physx::PxScene*			mPhysXScene;
#endif

	EffectType getEffectType(const NxParameterized::Interface* iface);

	bool						mAlive:1;
	bool						mEnabled: 1;
	bool						mVisible: 1;
	bool						mEverVisible: 1;
	bool						mFirstFrame: 1;
	bool						mFadeOut: 1;
	PxF32						mFadeOutTime;
	PxF32						mFadeOutDuration;

	bool						mFadeIn: 1;
	PxF32						mFadeInTime;
	PxF32						mFadeInDuration;

	NxApexEmitterActor::NxApexEmitterValidateCallback *mEmitterValidateCallback;

	PxF32						mFadeTime;
	PxF32                       mNotVisibleTime;
	VisState					mVisState;
	PxF32						mOffScreenTime;

	PxTransform					mPose;
	PxF32						mObjectScale;
	const EffectPackageAssetParams*	mData;
	Array< EffectData* >		mEffects;
	PxF32                       mSimTime;
	PxF32						mCurrentLifeTime;
	EffectPath					*mEffectPath;

	physx::apex::NxApexScene	*mScene;
	NxModuleTurbulenceFS		*mModuleTurbulenceFS;
	NxEffectPackageAsset		*mAsset;
	NxApexRenderVolume			*mRenderVolume;
	bool						mRigidBodyChange;
};

} // end of particles namespace
} // end of apex namespace
} // end of physx namespace

#endif
