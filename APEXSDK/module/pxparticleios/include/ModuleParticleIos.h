/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __MODULE_PARTICLE_H__
#define __MODULE_PARTICLE_H__

#include "NxApex.h"
#include "NxModuleParticleIos.h"
#include "NiApexSDK.h"
#include "Module.h"
#include "NiModule.h"
#include "NiResourceProvider.h"
#include "ApexSharedUtils.h"
#include "ApexSDKHelpers.h"
#include "ModulePerfScope.h"
#include "ApexAuthorableObject.h"
#include "ParticleIosAsset.h"
#include "PxparticleiosParamClasses.h"
#include "ApexRWLockable.h"

namespace physx
{
namespace apex
{

class NiModuleIofx;
class NiModuleFieldSampler;
class NxParticleIosActor;

namespace pxparticleios
{

class ParticleIosScene;


/**
\brief Module descriptor for ParticleIOS module
*/
class NxModuleParticleIosDesc : public NxApexDesc
{
public:

	/**
	\brief constructor sets to default.
	*/
	PX_INLINE NxModuleParticleIosDesc() : NxApexDesc()
	{
		init();
	}

	/**
	\brief sets members to default values.
	*/
	PX_INLINE void setToDefault()
	{
		NxApexDesc::setToDefault();
		init();
	}

	/**
	\brief checks if this is a valid descriptor.
	*/
	PX_INLINE bool isValid() const
	{
		bool retVal = NxApexDesc::isValid();
		return retVal;
	}

private:

	PX_INLINE void init()
	{
	}
};

class ModuleParticleIos : public NxModuleParticleIos, public NiModule, public Module, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleParticleIos(NiApexSDK* sdk);
	~ModuleParticleIos();

	void											init(const NxModuleParticleIosDesc& desc);

	// base class methods
	void											init(NxParameterized::Interface&);
	NxParameterized::Interface* 					getDefaultModuleDesc();
	void											release()
	{
		Module::release();
	}
	void											destroy();
	const char*										getName() const
	{
		NX_READ_ZONE();
		return Module::getName();
	}
	physx::PxU32									getNbParameters() const
	{
		NX_READ_ZONE();
		return Module::getNbParameters();
	}
	NxApexParameter**								getParameters()
	{
		NX_READ_ZONE();
		return Module::getParameters();
	}
	void											setLODUnitCost(physx::PxF32 cost)
	{
		NX_WRITE_ZONE();
		Module::setLODUnitCost(cost);
	}
	physx::PxF32									getLODUnitCost() const
	{
		NX_READ_ZONE();
		return Module::getLODUnitCost();
	}
	void											setLODBenefitValue(physx::PxF32 value)
	{
		NX_WRITE_ZONE();
		Module::setLODBenefitValue(value);
	}
	physx::PxF32									getLODBenefitValue() const
	{
		NX_READ_ZONE();
		return Module::getLODBenefitValue();
	}
	void											setLODEnabled(bool enabled)
	{
		NX_WRITE_ZONE();
		Module::setLODEnabled(enabled);
	}
	bool											getLODEnabled() const
	{
		NX_READ_ZONE();
		return Module::getLODEnabled();
	}

	//NxParticleIosActor *							getApexActor( NxApexScene* scene, NxAuthObjTypeID type ) const;
	ApexActor* 										getApexActor(NxApexActor* nxactor, NxAuthObjTypeID type) const;

	void setIntValue(physx::PxU32 parameterIndex, physx::PxU32 value)
	{
		NX_WRITE_ZONE();
		return Module::setIntValue(parameterIndex, value);
	}
	NiModuleScene* 									createNiModuleScene(NiApexScene&, NiApexRenderDebug*);
	void											releaseNiModuleScene(NiModuleScene&);
	physx::PxU32									forceLoadAssets();
	NxAuthObjTypeID									getModuleID() const;
	NxApexRenderableIterator* 						createRenderableIterator(const NxApexScene&);

	virtual const char*                             getParticleIosTypeName();

	ParticleIosScene* 								getParticleIosScene(const NxApexScene& scene);
	const ParticleIosScene* 						getParticleIosScene(const NxApexScene& scene) const;

	NiModuleIofx* 									getNiModuleIofx();
	NiModuleFieldSampler* 							getNiModuleFieldSampler();

protected:

	NxResourceList								mParticleIosSceneList;
	NxResourceList								mAuthorableObjects;

	friend class ParticleIosScene;
private:
#	define PARAM_CLASS(clas) PARAM_CLASS_DECLARE_FACTORY(clas)
#	include "PxparticleiosParamClasses.inc"

	ParticleIosModuleParameters*				mModuleParams;

	NiModuleIofx*                               mIofxModule;
	NiModuleFieldSampler*                       mFieldSamplerModule;
};

}
}
} // namespace physx::apex

#endif // __MODULE_PARTICLE_H__
