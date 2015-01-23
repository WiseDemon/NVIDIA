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
#include "NxModuleFluidIos.h"
#include "NiApexSDK.h"
#include "Module.h"
#include "NiModule.h"
#include "NiResourceProvider.h"
#include "ApexSharedUtils.h"
#include "ApexSDKHelpers.h"
#include "ModulePerfScope.h"
#include "ApexAuthorableObject.h"
#include "FluidIosAsset.h"
#include "NxfluidiosParamClasses.h"
#include "ApexRWLockable.h"

class NxCompartment;

namespace physx
{
namespace apex
{

class NiModuleIofx;

namespace nxfluidios
{
class FluidIosScene;

class ModuleFluidIos : public NxModuleFluidIos, public NiModule, public Module, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleFluidIos(NiApexSDK* sdk);
	~ModuleFluidIos();

	void											init(const NxModuleFluidIosDesc& moduleParticlesDesc);

	// base class methods
	void											init(NxParameterized::Interface&) {}
	NxParameterized::Interface* 					getDefaultModuleDesc();
	void											release()
	{
		Module::release();
	}
	void											destroy();
	const char*										getName() const
	{
		return Module::getName();
	}
	physx::PxU32									getNbParameters() const
	{
		return Module::getNbParameters();
	}
	NxApexParameter**								getParameters()
	{
		return Module::getParameters();
	}
	void											setLODUnitCost(physx::PxF32 cost)
	{
		Module::setLODUnitCost(cost);
	}
	physx::PxF32									getLODUnitCost() const
	{
		return Module::getLODUnitCost();
	}
	void											setLODBenefitValue(physx::PxF32 value)
	{
		Module::setLODBenefitValue(value);
	}
	physx::PxF32									getLODBenefitValue() const
	{
		return Module::getLODBenefitValue();
	}
	void											setLODEnabled(bool enabled)
	{
		Module::setLODEnabled(enabled);
	}
	bool											getLODEnabled() const
	{
		return Module::getLODEnabled();
	}

	ApexActor* 										getApexActor(NxApexActor* nxactor, NxAuthObjTypeID type) const;


	void setIntValue(physx::PxU32 parameterIndex, physx::PxU32 value)
	{
		return Module::setIntValue(parameterIndex, value);
	}
	NiModuleScene* 									createNiModuleScene(NiApexScene&, NiApexRenderDebug*);
	void											releaseNiModuleScene(NiModuleScene&);
	physx::PxU32									forceLoadAssets();
	NxAuthObjTypeID									getModuleID() const;
	NxApexRenderableIterator* 						createRenderableIterator(const NxApexScene&);

	virtual const char*                             getFluidIosTypeName();

	void											setCompartment(const NxApexScene&, NxCompartment& comp);
	const NxCompartment* 							getCompartment(const NxApexScene&) const;

	void                                            setSPHCompartment(const NxApexScene&, NxCompartment& comp);
	const NxCompartment*                            getSPHCompartment(const NxApexScene&) const;

	FluidIosScene* 									getParticleScene(const NxApexScene& scene);
	const FluidIosScene* 							getParticleScene(const NxApexScene& scene) const;

	NiModuleIofx* 									getNiModuleIofx();

protected:
	NiModuleIofx*                               mIofxModule;

#	define PARAM_CLASS(clas) PARAM_CLASS_DECLARE_FACTORY(clas)
#	include "NxfluidiosParamClasses.inc"

	NxResourceList								mParticleSceneList;
	NxResourceList								mAuthorableObjects;

	FluidIosModuleParameters* 					mModuleParams;

	friend class FluidIosScene;
};

}
}
} // namespace physx::apex

#endif // __MODULE_PARTICLES_H__
