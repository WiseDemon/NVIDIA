/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __MODULE_PARTICLES_H__
#define __MODULE_PARTICLES_H__

#include "NxApex.h"
#include "NxModuleParticles.h"
#include "NiApexSDK.h"
#include "NiModule.h"
#include "Module.h"

#include "ApexRWLockable.h"
#include "ApexInterface.h"
#include "ApexSDKHelpers.h"
#include "ParticlesDebugRenderParams.h"
#include "ParticlesModuleParameters.h"
#include "EffectPackageGraphicsMaterialsParams.h"

#include "EffectPackageAssetParams.h"
#include "EffectPackageActorParams.h"

#include "GraphicsMaterialData.h"
#include "VolumeRenderMaterialData.h"
#include "EmitterEffect.h"
#include "RigidBodyEffect.h"
#include "HeatSourceEffect.h"
#include "SubstanceSourceEffect.h"
#include "VelocitySourceEffect.h"
#include "ForceFieldEffect.h"
#include "JetFieldSamplerEffect.h"
#include "WindFieldSamplerEffect.h"
#include "NoiseFieldSamplerEffect.h"
#include "VortexFieldSamplerEffect.h"
#include "AttractorFieldSamplerEffect.h"
#include "TurbulenceFieldSamplerEffect.h"
#include "FlameEmitterEffect.h"

#include "EffectPackageData.h"
#include "AttractorFieldSamplerData.h"
#include "JetFieldSamplerData.h"
#include "WindFieldSamplerData.h"
#include "NoiseFieldSamplerData.h"
#include "VortexFieldSamplerData.h"
#include "TurbulenceFieldSamplerData.h"
#include "HeatSourceData.h"
#include "SubstanceSourceData.h"
#include "VelocitySourceData.h"
#include "ForceFieldData.h"
#include "EmitterData.h"
#include "GraphicsEffectData.h"
#include "ParticleSimulationData.h"
#include "FlameEmitterData.h"

#include "EffectPackageIOSDatabaseParams.h"
#include "EffectPackageIOFXDatabaseParams.h"
#include "EffectPackageEmitterDatabaseParams.h"
#include "EffectPackageDatabaseParams.h"
#include "EffectPackageFieldSamplerDatabaseParams.h"

#include "ReadCheck.h"
#include "WriteCheck.h"

namespace physx
{
namespace apex
{

class NiApexScene;
class NxModuleTurbulenceFS;
class NxApexEmitterActor;
class NxApexEmitterAsset;

namespace particles
{
class ParticlesAsset;
class ParticlesAssetAuthoring;
class ParticlesScene;

typedef Array< NiModuleScene* > ModuleSceneVector;

class NxModuleParticlesDesc : public NxApexDesc
{
public:

	/**
	\brief Constructor sets to default.
	*/
	PX_INLINE NxModuleParticlesDesc()
	{
		setToDefault();
	}
	/**
	\brief (re)sets the structure to the default.
	*/
	PX_INLINE void	setToDefault()
	{
		NxApexDesc::setToDefault();
		moduleValue = 0;
	}

	/**
	Returns true if an object can be created using this descriptor.
	*/
	PX_INLINE bool	isValid() const
	{
		return NxApexDesc::isValid();
	}

	/**
	Module configurable parameter.
	*/
	physx::PxU32 moduleValue;
};


class ModuleParticles : public NxModuleParticles, public NiModule, public Module, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleParticles(NiApexSDK* sdk);
	~ModuleParticles();

	void						init(const NxModuleParticlesDesc& desc);

	// base class methods
	void						init(NxParameterized::Interface&) {}
	NxParameterized::Interface* getDefaultModuleDesc();
	void release()
	{
		Module::release();
	}
	void destroy();
	const char*					getName() const
	{
		NX_READ_ZONE();
		return Module::getName();
	}
	physx::PxU32				getNbParameters() const
	{
		NX_READ_ZONE();
		return Module::getNbParameters();
	}
	NxApexParameter**			getParameters()
	{
		NX_READ_ZONE();
		return Module::getParameters();
	}

	/**
	\brief Get the cost of one LOD aspect unit.
	*/
	virtual physx::PxF32 getLODUnitCost() const
	{
		NX_READ_ZONE();
		return 0;
	}

	/**
	\brief Set the cost of one LOD aspect unit.
	*/
	virtual void setLODUnitCost(physx::PxF32)
	{
		NX_WRITE_ZONE();
	}

	/**
	\brief Get the resource value of one unit of benefit.
	*/
	virtual physx::PxF32 getLODBenefitValue() const
	{
		NX_READ_ZONE();
		return 0;
	}

	/**
	\brief Set the resource value of one unit of benefit.
	*/
	virtual void setLODBenefitValue(physx::PxF32)
	{
		NX_WRITE_ZONE();
	}

	/**
	\brief Get enabled/disabled state of automatic LOD system.
	*/
	virtual bool getLODEnabled() const
	{
		NX_READ_ZONE();
		return false;
	}

	/**
	\brief Set enabled/disabled state of automatic LOD system.
	*/
	virtual void setLODEnabled(bool)
	{
		NX_WRITE_ZONE();
	}


	void						setIntValue(physx::PxU32 parameterIndex, physx::PxU32 value)
	{
		NX_WRITE_ZONE();
		return Module::setIntValue(parameterIndex, value);
	}

	virtual bool setEffectPackageGraphicsMaterialsDatabase(const NxParameterized::Interface* dataBase);

	virtual const NxParameterized::Interface* getEffectPackageGraphicsMaterialsDatabase() const;

	virtual bool setEffectPackageIOSDatabase(const NxParameterized::Interface* dataBase);
	virtual bool setEffectPackageIOFXDatabase(const NxParameterized::Interface* dataBase);
	virtual bool setEffectPackageEmitterDatabase(const NxParameterized::Interface* dataBase);
	virtual bool setEffectPackageDatabase(const NxParameterized::Interface* dataBase);
	virtual bool setEffectPackageFieldSamplerDatabase(const NxParameterized::Interface* dataBase);

	virtual const NxParameterized::Interface* getEffectPackageIOSDatabase(void) const
	{
		NX_READ_ZONE();
		return mEffectPackageIOSDatabaseParams;
	};
	virtual const NxParameterized::Interface* getEffectPackageIOFXDatabase(void) const
	{
		NX_READ_ZONE();
		return mEffectPackageIOFXDatabaseParams;
	};
	virtual const NxParameterized::Interface* getEffectPackageEmitterDatabase(void) const
	{
		NX_READ_ZONE();
		return mEffectPackageEmitterDatabaseParams;
	};
	virtual const NxParameterized::Interface* getEffectPackageDatabase(void) const
	{
		NX_READ_ZONE();
		return mEffectPackageDatabaseParams;
	};
	virtual const NxParameterized::Interface* getEffectPackageFieldSamplerDatabase(void) const
	{
		NX_READ_ZONE();
		return mEffectPackageFieldSamplerDatabaseParams;
	};

	bool initParticleSimulationData(ParticleSimulationData* ed);

	virtual NxParameterized::Interface* locateResource(const char* resourceName,		// the name of the resource
	        const char* nameSpace);

	virtual const char** getResourceNames(const char* nameSpace, physx::PxU32& nameCount, const char** &variants);

	virtual const NxParameterized::Interface* locateGraphicsMaterialData(const char* name) const;
	virtual const NxParameterized::Interface* locateVolumeRenderMaterialData(const char* name) const;


	NiModuleScene* 				createNiModuleScene(NiApexScene&, NiApexRenderDebug*);
	void						releaseNiModuleScene(NiModuleScene&);
	physx::PxU32				forceLoadAssets();
	NxAuthObjTypeID				getModuleID() const;
	NxApexRenderableIterator* 	createRenderableIterator(const NxApexScene&);

	NxAuthObjTypeID             getParticlesAssetTypeID() const;

	physx::PxU32				getModuleValue() const
	{
		return mModuleValue;
	}

	NxModuleTurbulenceFS* getModuleTurbulenceFS(void)
	{
		return mTurbulenceModule;
	}

	ParticlesScene* 			getParticlesScene(const NxApexScene& apexScene);

	virtual void setEnableScreenCulling(bool state, bool znegative)
	{
		NX_WRITE_ZONE();
		mEnableScreenCulling = state;
		mZnegative = znegative;
	}

	bool getEnableScreenCulling(void) const
	{
		return mEnableScreenCulling;
	}

	bool getZnegative(void) const
	{
		return mZnegative;
	}

	virtual void resetEmitterPool(void);

	virtual void setUseEmitterPool(bool state)
	{
		NX_WRITE_ZONE();
		mUseEmitterPool = state;
	}

	virtual bool getUseEmitterPool(void) const
	{
		NX_READ_ZONE();
		return mUseEmitterPool;
	}
#if NX_SDK_VERSION_MAJOR == 3
	PxMaterial *getDefaultMaterial(void) const
	{
		return mDefaultMaterial;
	}
#endif

	virtual void notifyReleaseSDK(void);

	virtual void notifyChildGone(NiModule* imodule);

protected:
	bool						mUseEmitterPool;
	bool						mEnableScreenCulling;
	bool						mZnegative;

	NxResourceList				mAuthorableObjects;
	NxResourceList				mEffectPackageAuthorableObjects;

	NxResourceList				mParticlesScenes;

	physx::PxU32				mModuleValue;

	friend class ParticlesAsset;

	/**
	\brief Used by the ParticleEffectTool to initialize the default database values for the editor
	*/
	virtual void initializeDefaultDatabases(void);

	virtual physx::apex::NxModule* getModule(const char* moduleName);

private:

	bool fixFieldSamplerCollisionFilterNames(NxParameterized::Interface *fs);

	bool fixupNamedReferences(void);

	ParticlesDebugRenderParamsFactory							mParticlesDebugRenderParamsFactory;
	ParticlesModuleParametersFactory							mParticlesModuleParametersFactory;
	EffectPackageGraphicsMaterialsParamsFactory					mEffectPackageGraphicsMaterialsParamsFactory;

	// factory definitions in support of effect packages

	EffectPackageAssetParamsFactory								mEffectPackageAssetParamsFactory;
	EffectPackageActorParamsFactory								mEffectPackageActorParamsFactory;

	RigidBodyEffectFactory										mRigidBodyFactory;
	EmitterEffectFactory										mEmitterEffectFactory;
	HeatSourceEffectFactory										mHeatSourceEffectFactory;
	SubstanceSourceEffectFactory								mSubstanceSourceEffectFactory;
	VelocitySourceEffectFactory									mVelocitySourceEffectFactory;
	ForceFieldEffectFactory										mForceFieldEffectFactory;
	JetFieldSamplerEffectFactory								mJetFieldSamplerEffectFactory;
	WindFieldSamplerEffectFactory								mWindFieldSamplerEffectFactory;
	NoiseFieldSamplerEffectFactory								mNoiseFieldSamplerEffectFactory;
	VortexFieldSamplerEffectFactory								mVortexFieldSamplerEffectFactory;
	AttractorFieldSamplerEffectFactory							mAttractorFieldSamplerEffectFactory;
	TurbulenceFieldSamplerEffectFactory							mTurbulenceFieldSamplerEffectFactory;
	FlameEmitterEffectFactory									mFlameEmitterEffectFactory;

	EffectPackageIOSDatabaseParamsFactory						mEffectPackageIOSDatabaseParamsFactory;
	EffectPackageIOFXDatabaseParamsFactory						mEffectPackageIOFXDatabaseParamsFactory;
	EffectPackageEmitterDatabaseParamsFactory					mEffectPackageEmitterDatabaseParamsFactory;
	EffectPackageDatabaseParamsFactory							mEffectPackageDatabaseParamsFactory;
	EffectPackageFieldSamplerDatabaseParamsFactory				mEffectPackageFieldSamplerDatabaseParamsFactory;

	EffectPackageDataFactory									mEffectPackageDataFactory;
	AttractorFieldSamplerDataFactory							mAttractorFieldSamplerDataFactory;
	NoiseFieldSamplerDataFactory								mNoiseFieldSamplerDataFactory;
	VortexFieldSamplerDataFactory								mVortexFieldSamplerDataFactory;
	JetFieldSamplerDataFactory									mJetFieldSamplerDataFactory;
	WindFieldSamplerDataFactory									mWindFieldSamplerDataFactory;
	TurbulenceFieldSamplerDataFactory							mTurbulenceFieldSamplerDataFactory;
	HeatSourceDataFactory										mHeatSourceDataFactory;
	SubstanceSourceDataFactory									mSubstanceSourceDataFactory;
	VelocitySourceDataFactory									mVelocitySourceDataFactory;
	ForceFieldDataFactory										mForceFieldDataFactory;
	EmitterDataFactory											mEmitterDataFactory;
	GraphicsEffectDataFactory									mGraphicsEffectDataFactory;
	ParticleSimulationDataFactory								mParticleSimulationDataFactory;
	GraphicsMaterialDataFactory									mGraphicsMaterialDataFactory;
	VolumeRenderMaterialDataFactory								mVolumeRenderMaterialDataFactory;
	FlameEmitterDataFactory										mFlameEmitterDataFactory;

	NxParameterized::Interface*									mEffectPackageIOSDatabaseParams;
	NxParameterized::Interface*									mEffectPackageIOFXDatabaseParams;
	NxParameterized::Interface*									mEffectPackageEmitterDatabaseParams;
	NxParameterized::Interface*									mEffectPackageDatabaseParams;
	NxParameterized::Interface*									mEffectPackageFieldSamplerDatabaseParams;

	ParticlesModuleParameters*									mModuleParams;
	NxParameterized::Interface*									mGraphicsMaterialsDatabase;
	NxModuleTurbulenceFS*							mTurbulenceModule;
	ModuleSceneVector											mScenes;
	Array< const char*>										mTempNames;
	Array< const char*>										mTempVariantNames;

	physx::apex::NxModule* 	mModuleBasicIos;			// Instantiate the BasicIOS module statically
	physx::apex::NxModule* 	mModuleEmitter;				// Instantiate the Emitter module statically
	physx::apex::NxModule* 	mModuleIofx;				// Instantiate the IOFX module statically
	physx::apex::NxModule* 	mModuleFieldSampler;		// Instantiate the field sampler module statically
	physx::apex::NxModule* 	mModuleBasicFS;				// Instantiate the BasicFS module statically
#	if NX_SDK_VERSION_MAJOR == 2
	physx::apex::NxModule* 	mModuleFieldBoundary;		// PhysX 2.8 only : Instantiate the FieldBoundary module
	physx::apex::NxModule* 	mModuleExplosion;			// PhysX 2.8 only : Instantiate the explosion module
	physx::apex::NxModule* 	mModuleFluidIos;			// PhysX 2.8 only : Instantiate the fluidIOS module
#	elif NX_SDK_VERSION_MAJOR == 3
	physx::apex::NxModule* 	mModuleParticleIos;			// PhysX 3.x only : Instantiate the ParticleIOS module
	physx::apex::NxModule* 	mModuleForceField;			// PhysX 3.x only : Instantiate the ForceField module
#	endif
#if NX_SDK_VERSION_MAJOR == 3
	physx::PxMaterial		*mDefaultMaterial;
#endif
};

}
}
} // end namespace physx::apex

#endif // __MODULE_PARTICLES_H__
