/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __MODULE_FORCEFIELD_H__
#define __MODULE_FORCEFIELD_H__

#include "NxApex.h"
#include "NxModuleForceField.h"
#include "NiApexSDK.h"
#include "NiModule.h"
#include "Module.h"

#include "ApexInterface.h"
#include "ApexSDKHelpers.h"
#include "ApexRWLockable.h"
#include "ReadCheck.h"
#include "WriteCheck.h"
#include "ForcefieldParamClasses.h"

namespace physx
{
namespace apex
{
class NiApexScene;
class NiModuleFieldSampler;

namespace forcefield
{

class ForceFieldAsset;
class ForceFieldAssetAuthoring;
class ForceFieldScene;

class NxModuleForceFieldDesc : public NxApexDesc
{
public:

	/**
	\brief Constructor sets to default.
	*/
	PX_INLINE NxModuleForceFieldDesc()
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


class ModuleForceField : public NxModuleForceField, public NiModule, public Module, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleForceField(NiApexSDK* sdk);
	~ModuleForceField();

	void						init(const NxModuleForceFieldDesc& explosionDesc);

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
		return Module::getName();
	}
	physx::PxU32				getNbParameters() const
	{
		return Module::getNbParameters();
	}
	NxApexParameter**			getParameters()
	{
		return Module::getParameters();
	}
	void						setLODUnitCost(physx::PxF32 cost)
	{
		Module::setLODUnitCost(cost);
	}
	physx::PxF32				getLODUnitCost() const
	{
		return Module::getLODUnitCost();
	}
	void						setLODBenefitValue(physx::PxF32 value)
	{
		Module::setLODBenefitValue(value);
	}
	physx::PxF32				getLODBenefitValue() const
	{
		return Module::getLODBenefitValue();
	}
	void						setLODEnabled(bool enabled)
	{
		Module::setLODEnabled(enabled);
	}
	bool						getLODEnabled() const
	{
		return Module::getLODEnabled();
	}
	void						setIntValue(physx::PxU32 parameterIndex, physx::PxU32 value)
	{
		return Module::setIntValue(parameterIndex, value);
	}

	NiModuleScene* 				createNiModuleScene(NiApexScene&, NiApexRenderDebug*);
	void						releaseNiModuleScene(NiModuleScene&);
	physx::PxU32				forceLoadAssets();
	NxAuthObjTypeID				getModuleID() const;
	NxApexRenderableIterator* 	createRenderableIterator(const NxApexScene&);

	NxAuthObjTypeID             getForceFieldAssetTypeID() const;

	physx::PxU32				getModuleValue() const
	{
		NX_READ_ZONE();
		return mModuleValue;
	}

	NiModuleFieldSampler*		getNiModuleFieldSampler();

protected:
	ForceFieldScene* 			getForceFieldScene(const NxApexScene& apexScene);

	NxResourceList				mAuthorableObjects;

	NxResourceList				mForceFieldScenes;

	physx::PxU32				mModuleValue;

	NiModuleFieldSampler*		mFieldSamplerModule;

	friend class ForceFieldAsset;

private:
#	define PARAM_CLASS(clas) PARAM_CLASS_DECLARE_FACTORY(clas)
#	include "ForcefieldParamClasses.inc"

	ForceFieldModuleParams*				mModuleParams;
};

}
}
} // end namespace physx::apex

#endif // __MODULE_FORCEFIELD_H__
