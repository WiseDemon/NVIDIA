/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __MODULE_IOFX_H__
#define __MODULE_IOFX_H__

#include "NxApex.h"
#include "NxModuleIofx.h"
#include "NiApexSDK.h"
#include "Module.h"
#include "NiModuleIofx.h"
#include "NiResourceProvider.h"
#include "ApexSharedUtils.h"
#include "ApexSDKHelpers.h"
#include "ModulePerfScope.h"
#include "ApexAuthorableObject.h"
#include "ApexRWLockable.h"
#include "IofxParamClasses.h"
#include "ReadCheck.h"
#include "WriteCheck.h"

namespace physx
{
namespace apex
{

class NxApexRenderVolume;

namespace iofx
{
class IofxAsset;
class IofxScene;

class NxModuleIofxDesc : public NxApexDesc
{
public:

	/**
	\brief constructor sets to default.
	*/
	PX_INLINE NxModuleIofxDesc() : NxApexDesc()
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

class ModuleIofx : public NxModuleIofx, public NiModuleIofx, public Module, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleIofx(NiApexSDK* sdk);
	~ModuleIofx();

	void							init(const NxModuleIofxDesc& ModuleIofxDesc);

	// base class methods
	void							init(NxParameterized::Interface&) {}
	NxParameterized::Interface* 	getDefaultModuleDesc();
	void							release()
	{
		Module::release();
	}
	void							destroy();
	const char*						getName() const
	{
		NX_READ_ZONE();
		return Module::getName();
	}
	physx::PxU32					getNbParameters() const
	{
		NX_READ_ZONE();
		return Module::getNbParameters();
	}
	NxApexParameter**				getParameters()
	{
		NX_READ_ZONE();
		return Module::getParameters();
	}
	void							setLODUnitCost(physx::PxF32 cost)
	{
		NX_WRITE_ZONE();
		Module::setLODUnitCost(cost);
	}
	physx::PxF32					getLODUnitCost() const
	{
		NX_READ_ZONE();
		return Module::getLODUnitCost();
	}
	void							setLODBenefitValue(physx::PxF32 value)
	{
		NX_WRITE_ZONE();
		Module::setLODBenefitValue(value);
	}
	physx::PxF32					getLODBenefitValue() const
	{
		NX_READ_ZONE();
		return Module::getLODBenefitValue();
	}
	void							setLODEnabled(bool enabled)
	{
		NX_WRITE_ZONE();
		Module::setLODEnabled(enabled);
	}
	bool							getLODEnabled() const
	{
		NX_READ_ZONE();
		return Module::getLODEnabled();
	}

	NxApexRenderableIterator*		createRenderableIterator(const NxApexScene&);

	void							disableCudaInterop()
	{
		NX_WRITE_ZONE();
		mInteropDisabled = true;
	}
	void							disableCudaModifiers()
	{
		NX_WRITE_ZONE();
		mCudaDisabled = true;
	}
	void							disableDeferredRenderableAllocation()
	{
		NX_WRITE_ZONE();
		mDeferredDisabled = true;
	}

	const NxTestBase*				getTestBase(NxApexScene* apexScene) const;

	// NiModuleIofx methods
	NiIofxManager*					createActorManager(const NxApexScene& scene, const NxIofxAsset& asset, const NiIofxManagerDesc& desc);

	void setIntValue(physx::PxU32 parameterIndex, physx::PxU32 value)
	{
		NX_WRITE_ZONE();
		return Module::setIntValue(parameterIndex, value);
	}
	physx::PxU32					forceLoadAssets();
	NxAuthObjTypeID					getModuleID() const;

	NiModuleScene* 					createNiModuleScene(NiApexScene&, NiApexRenderDebug*);
	void							releaseNiModuleScene(NiModuleScene&);

	IofxScene* 						getIofxScene(const NxApexScene& scene);
	const IofxScene* 				getIofxScene(const NxApexScene& scene) const;

	NxApexRenderVolume* 			createRenderVolume(const NxApexScene& apexScene, const PxBounds3& b, PxU32 priority, bool allIofx);
	void							releaseRenderVolume(NxApexRenderVolume& volume);

protected:

	NxResourceList								mAuthorableObjects;

#	define PARAM_CLASS(clas) PARAM_CLASS_DECLARE_FACTORY(clas)
#	include "IofxParamClasses.inc"

	IofxModuleParameters* 						mModuleParams;
	bool										mInteropDisabled;
	bool										mCudaDisabled;
	bool										mDeferredDisabled;

	NxResourceList								mIofxScenes;
	friend class IofxActor;
	friend class IofxScene;
	friend class IofxManager;
	friend class IofxManagerGPU;
};

}
}
} // namespace apex

#endif // __MODULE_PARTICLES_H__
