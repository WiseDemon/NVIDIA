/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __MODULE_FIELD_SAMPLER_H__
#define __MODULE_FIELD_SAMPLER_H__

#include "NxApex.h"
#include "NxModuleFieldSampler.h"
#include "NiApexSDK.h"
#include "NiModule.h"
#include "Module.h"

#include "ApexInterface.h"
#include "ApexSDKHelpers.h"
#include "ApexRWLockable.h"
#include "FieldsamplerParamClasses.h"

#include "NiModuleFieldSampler.h"

namespace physx
{
namespace apex
{

class NiApexScene;

namespace fieldsampler
{

class FieldSamplerScene;

class ModuleFieldSampler : public NxModuleFieldSampler, public NiModuleFieldSampler, public Module, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleFieldSampler(NiApexSDK* sdk);
	~ModuleFieldSampler();
#if NX_SDK_VERSION_MAJOR == 2
	bool						setFieldBoundaryGroupsFilteringParams(const NxApexScene& apexScene ,
	        const NxGroupsFilteringParams64& params);

	bool						getFieldBoundaryGroupsFilteringParams(const NxApexScene& apexScene ,
	        NxGroupsFilteringParams64& params) const;

	bool						setFieldSamplerGroupsFilteringParams(const NxApexScene& apexScene ,
	        const NxGroupsFilteringParams64& params);

	bool						getFieldSamplerGroupsFilteringParams(const NxApexScene& apexScene ,
	        NxGroupsFilteringParams64& params) const;
#endif

	NiFieldSamplerManager*		getNiFieldSamplerManager(const NxApexScene& apexScene);

#if NX_SDK_VERSION_MAJOR == 3
	bool setFieldSamplerWeightedCollisionFilterCallback(const NxApexScene& apexScene,NxFieldSamplerWeightedCollisionFilterCallback *callback);
	void						enablePhysXMonitor(const NxApexScene& apexScene, bool enable);

	void						setPhysXMonitorFilterData(const NxApexScene& apexScene, physx::PxFilterData filterData);

	physx::PxU32				createForceSampleBatch(const NxApexScene& apexScene, physx::PxU32 maxCount, const physx::PxFilterData filterData);
	void						releaseForceSampleBatch(const NxApexScene& apexScene, physx::PxU32 batchId);
	void						submitForceSampleBatch(const NxApexScene& apexScene, physx::PxU32 batchId,
													   PxVec4* forces, const PxU32 forcesStride,
													   const PxVec3* positions, const PxU32 positionsStride,
													   const PxVec3* velocities, const PxU32 velocitiesStride,
													   const PxF32* mass, const PxU32 massStride,
													   const PxU32* indices, const PxU32 numIndices);
#endif


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


protected:
	fieldsampler::FieldSamplerScene* 	getFieldSamplerScene(const NxApexScene& apexScene) const;

	NxResourceList				mFieldSamplerScenes;

private:

#	define PARAM_CLASS(clas) PARAM_CLASS_DECLARE_FACTORY(clas)
#	include "FieldsamplerParamClasses.inc"

	FieldSamplerModuleParameters* 			mModuleParams;
};

}
}
} // end namespace physx::apex

#endif // __MODULE_FIELD_SAMPLER_H__
