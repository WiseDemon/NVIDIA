/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __MODULE_BASIC_FS_H__
#define __MODULE_BASIC_FS_H__

#include "NxApex.h"
#include "NxModuleBasicFS.h"
#include "NiApexSDK.h"
#include "NiModule.h"
#include "Module.h"

#include "ApexInterface.h"
#include "ApexSDKHelpers.h"
#include "ApexRWLockable.h"
#include "BasicfsParamClasses.h"


namespace physx
{
namespace apex
{

class NiApexScene;

class NiModuleFieldSampler;
	
namespace basicfs
{

class BasicFSAsset;
class JetFSAssetAuthoring;
class AttractorFSAssetAuthoring;
class VortexFSAssetAuthoring;
class BasicFSScene;

class ModuleBasicFS : public NxModuleBasicFS, public NiModule, public Module, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleBasicFS(NiApexSDK* sdk);
	~ModuleBasicFS();

	// base class methods
	void						init(NxParameterized::Interface&);
	NxParameterized::Interface* getDefaultModuleDesc();
	void release()
	{
		Module::release();
	}
	void destroy();
	const char* getName() const
	{
		return Module::getName();
	}
	PxU32 getNbParameters() const
	{
		return Module::getNbParameters();
	}
	NxApexParameter** getParameters()
	{
		return Module::getParameters();
	}
	void setLODUnitCost(physx::PxF32 cost)
	{
		Module::setLODUnitCost(cost);
	}
	physx::PxF32 getLODUnitCost() const
	{
		return Module::getLODUnitCost();
	}
	void setLODBenefitValue(physx::PxF32 value)
	{
		Module::setLODBenefitValue(value);
	}
	physx::PxF32 getLODBenefitValue() const
	{
		return Module::getLODBenefitValue();
	}
	void setLODEnabled(bool enabled)
	{
		Module::setLODEnabled(enabled);
	}
	bool getLODEnabled() const
	{
		return Module::getLODEnabled();
	}
	void setIntValue(PxU32 parameterIndex, PxU32 value)
	{
		return Module::setIntValue(parameterIndex, value);
	}

	NiModuleScene* 				createNiModuleScene(NiApexScene&, NiApexRenderDebug*);
	void						releaseNiModuleScene(NiModuleScene&);
	NxAuthObjTypeID				getModuleID() const;
	NxApexRenderableIterator* 	createRenderableIterator(const NxApexScene&);

	NxAuthObjTypeID             getJetFSAssetTypeID() const;
	NxAuthObjTypeID             getAttractorFSAssetTypeID() const;
	NxAuthObjTypeID             getVortexFSAssetTypeID() const;
	NxAuthObjTypeID             getNoiseFSAssetTypeID() const;
	NxAuthObjTypeID             getWindFSAssetTypeID() const;

	ApexActor* 					getApexActor(NxApexActor*, NxAuthObjTypeID) const;

	NiModuleFieldSampler* 		getNiModuleFieldSampler();

	BasicFSScene* 				getBasicFSScene(const NxApexScene& apexScene); // return to protected
protected:
	NxResourceList				mAuthorableObjects;

	NxResourceList				mBasicFSScenes;

	friend class BasicFSAsset;
	friend class JetFSAsset;
	friend class AttractorFSAsset;
	friend class VortexFSAsset;
	friend class BasicFSScene;

private:

#	define PARAM_CLASS(clas) PARAM_CLASS_DECLARE_FACTORY(clas)
#	include "BasicfsParamClasses.inc"

	BasicFSModuleParameters* 			mModuleParams;

	NiModuleFieldSampler* 				mFieldSamplerModule;
};

}
}
} // end namespace physx::apex

#endif // __MODULE_BASIC_FS_H__
