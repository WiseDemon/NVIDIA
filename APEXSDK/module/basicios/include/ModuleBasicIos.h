/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __MODULE_BASIC_IOS_H__
#define __MODULE_BASIC_IOS_H__

#include "NxApex.h"
#include "NxModuleBasicIos.h"
#include "NiApexSDK.h"
#include "Module.h"
#include "NiModule.h"
#include "NiResourceProvider.h"
#include "ApexSharedUtils.h"
#include "ApexSDKHelpers.h"
#include "ModulePerfScope.h"
#include "ApexAuthorableObject.h"
#include "BasicIosAsset.h"
#include "BasiciosParamClasses.h"
#include "ApexRWLockable.h"

namespace physx
{
namespace apex
{

class NiModuleIofx;
class NiModuleFieldSampler;

namespace basicios
{

class BasicIosScene;

/**
\brief Module descriptor for BasicIOS module
*/
class NxModuleBasicIosDesc : public NxApexDesc
{
public:

	/**
	\brief constructor sets to default.
	*/
	PX_INLINE NxModuleBasicIosDesc() : NxApexDesc()
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

class ModuleBasicIos : public NxModuleBasicIos, public NiModule, public Module, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleBasicIos(NiApexSDK* sdk);
	~ModuleBasicIos();

	void											init(const NxModuleBasicIosDesc& desc);

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

	//NxBasicIosActor *								getApexActor( NxApexScene* scene, NxAuthObjTypeID type ) const;
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

	virtual const char*                             getBasicIosTypeName();

	BasicIosScene* 									getBasicIosScene(const NxApexScene& scene);
	const BasicIosScene* 							getBasicIosScene(const NxApexScene& scene) const;

	NiModuleIofx* 									getNiModuleIofx();
	NiModuleFieldSampler* 							getNiModuleFieldSampler();

	const NxTestBase*								getTestBase(NxApexScene* apexScene) const;

protected:

	NxResourceList								mBasicIosSceneList;
	NxResourceList								mAuthorableObjects;

	friend class BasicIosScene;
private:

#	define PARAM_CLASS(clas) PARAM_CLASS_DECLARE_FACTORY(clas)
#	include "BasiciosParamClasses.inc"

	BasicIosModuleParameters*					mModuleParams;

	NiModuleIofx*                               mIofxModule;
	NiModuleFieldSampler*                       mFieldSamplerModule;
};

}
}
} // namespace physx::apex

#endif // __MODULE_BASIC_IOS_H__
