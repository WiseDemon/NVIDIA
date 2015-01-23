/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "NxApexDefs.h"
#include "MinPhysxSdkVersion.h"

#if NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
#include "NxApex.h"
#include "ModuleFieldSampler.h"
#include "FieldSamplerScene.h"
#include "FieldSamplerManager.h"
#include "NiApexScene.h"
#include "PxMemoryBuffer.h"
#include "ModulePerfScope.h"
using namespace fieldsampler;
#endif

#include "NiApexSDK.h"
#include "PsShare.h"

#include "NxLock.h"

#include "ReadCheck.h"
#include "WriteCheck.h"

namespace physx
{
namespace apex
{

#if defined(_USRDLL)

/* Modules don't have to link against the framework, they keep their own */
NiApexSDK* gApexSdk = 0;
NxApexSDK* NxGetApexSDK()
{
	return gApexSdk;
}
NiApexSDK* NiGetApexSDK()
{
	return gApexSdk;
}

NXAPEX_API NxModule*  NX_CALL_CONV createModule(
    NiApexSDK* inSdk,
    NiModule** niRef,
    PxU32 APEXsdkVersion,
    PxU32 PhysXsdkVersion,
    NxApexCreateError* errorCode)
{
	if (APEXsdkVersion != NX_APEX_SDK_VERSION)
	{
		if (errorCode)
		{
			*errorCode = APEX_CE_WRONG_VERSION;
		}
		return NULL;
	}

	if (PhysXsdkVersion != NX_PHYSICS_SDK_VERSION)
	{
		if (errorCode)
		{
			*errorCode = APEX_CE_WRONG_VERSION;
		}
		return NULL;
	}

#if NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
	gApexSdk = inSdk;
	APEX_INIT_FOUNDATION();
	initModuleProfiling(inSdk, "FieldSampler");
	ModuleFieldSampler* impl = PX_NEW(ModuleFieldSampler)(inSdk);
	*niRef  = (NiModule*) impl;
	return (NxModule*) impl;
#else // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
	if (errorCode != NULL)
	{
		*errorCode = APEX_CE_WRONG_VERSION;
	}

	PX_UNUSED(niRef);
	PX_UNUSED(inSdk);
	return NULL; // FieldSampler Module can only compile against 283
#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
}

#else
/* Statically linking entry function */
void instantiateModuleFieldSampler()
{
#if NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
	NiApexSDK* sdk = NiGetApexSDK();
	initModuleProfiling(sdk, "FieldSampler");
	fieldsampler::ModuleFieldSampler* impl = PX_NEW(fieldsampler::ModuleFieldSampler)(sdk);
	sdk->registerExternalModule((NxModule*) impl, (NiModule*) impl);
#endif
}
#endif // `defined(_USRDLL)

namespace fieldsampler
{
/* === ModuleFieldSampler Implementation === */
#if NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED

ModuleFieldSampler::ModuleFieldSampler(NiApexSDK* sdk)
{
	name = "FieldSampler";
	mSdk = sdk;
	mApiProxy = this;
	mModuleParams = NULL;

	/* Register the NxParameterized factories */
	NxParameterized::Traits* traits = mSdk->getParameterizedTraits();
#	define PARAM_CLASS(clas) PARAM_CLASS_REGISTER_FACTORY(traits, clas)
#	include "FieldsamplerParamClasses.inc"
}

ModuleFieldSampler::~ModuleFieldSampler()
{
	releaseModuleProfiling();
}

void ModuleFieldSampler::destroy()
{
	NxParameterized::Traits* traits = mSdk->getParameterizedTraits();

	if (mModuleParams)
	{
		mModuleParams->destroy();
		mModuleParams = NULL;
	}

	Module::destroy();
	delete this;

	if (traits)
	{
		/* Remove the NxParameterized factories */

#		define PARAM_CLASS(clas) PARAM_CLASS_REMOVE_FACTORY(traits, clas)
#		include "FieldsamplerParamClasses.inc"
	}
}

void ModuleFieldSampler::init(NxParameterized::Interface&)
{
}

NxParameterized::Interface* ModuleFieldSampler::getDefaultModuleDesc()
{
	NxParameterized::Traits* traits = mSdk->getParameterizedTraits();

	if (!mModuleParams)
	{
		mModuleParams = DYNAMIC_CAST(FieldSamplerModuleParameters*)
		                (traits->createNxParameterized("FieldSamplerModuleParameters"));
		PX_ASSERT(mModuleParams);
	}
	else
	{
		mModuleParams->initDefaults();
	}

	return mModuleParams;
}

NiFieldSamplerManager* ModuleFieldSampler::getNiFieldSamplerManager(const NxApexScene& apexScene)
{
	FieldSamplerScene* scene = ModuleFieldSampler::getFieldSamplerScene(apexScene);
	return scene->getManager();
}


NxAuthObjTypeID ModuleFieldSampler::getModuleID() const
{
	return 0;
}


/* == Example Scene methods == */
NiModuleScene* ModuleFieldSampler::createNiModuleScene(NiApexScene& scene, NiApexRenderDebug* debugRender)
{
#if defined(APEX_CUDA_SUPPORT)
	NX_READ_LOCK(scene);
	if (scene.getTaskManager()->getGpuDispatcher() && scene.isUsingCuda())
	{
		return PX_NEW(FieldSamplerSceneGPU)(*this, scene, debugRender, mFieldSamplerScenes);
	}
	else
#endif
		return PX_NEW(FieldSamplerSceneCPU)(*this, scene, debugRender, mFieldSamplerScenes);
}

void ModuleFieldSampler::releaseNiModuleScene(NiModuleScene& scene)
{
	FieldSamplerScene* es = DYNAMIC_CAST(FieldSamplerScene*)(&scene);
	es->destroy();
}

fieldsampler::FieldSamplerScene* ModuleFieldSampler::getFieldSamplerScene(const NxApexScene& apexScene) const
{
	for (PxU32 i = 0 ; i < mFieldSamplerScenes.getSize() ; i++)
	{
		FieldSamplerScene* es = DYNAMIC_CAST(FieldSamplerScene*)(mFieldSamplerScenes.getResource(i));
		if (es->mApexScene == &apexScene)
		{
			return es;
		}
	}

	PX_ASSERT(!"Unable to locate an appropriate FieldSamplerScene");
	return NULL;
}

NxApexRenderableIterator* ModuleFieldSampler::createRenderableIterator(const NxApexScene& apexScene)
{
	FieldSamplerScene* es = getFieldSamplerScene(apexScene);
	if (es)
	{
		return es->createRenderableIterator();
	}

	return NULL;
}

#if NX_SDK_VERSION_MAJOR == 2
bool ModuleFieldSampler::setFieldBoundaryGroupsFilteringParams(const NxApexScene& apexScene ,
        const NxGroupsFilteringParams64& params)
{
	NX_WRITE_ZONE();
	FieldSamplerScene* scene = getFieldSamplerScene(apexScene);
	if (scene != NULL)
	{
		DYNAMIC_CAST(FieldSamplerManager*)(scene->getManager())->setFieldBoundaryGroupsFilteringParams(params);
		return true;
	}
	return false;
}

bool ModuleFieldSampler::getFieldBoundaryGroupsFilteringParams(const NxApexScene& apexScene ,
        NxGroupsFilteringParams64& params) const
{
	NX_READ_ZONE();
	FieldSamplerScene* scene = getFieldSamplerScene(apexScene);
	if (scene != NULL)
	{
		DYNAMIC_CAST(FieldSamplerManager*)(scene->getManager())->getFieldBoundaryGroupsFilteringParams(params);
		return true;
	}
	return false;
}

bool ModuleFieldSampler::setFieldSamplerGroupsFilteringParams(const NxApexScene& apexScene ,
        const NxGroupsFilteringParams64& params)
{
	NX_WRITE_ZONE();
	FieldSamplerScene* scene = getFieldSamplerScene(apexScene);
	if (scene != NULL)
	{
		DYNAMIC_CAST(FieldSamplerManager*)(scene->getManager())->setFieldSamplerGroupsFilteringParams(params);
		return true;
	}
	return false;
}

bool ModuleFieldSampler::getFieldSamplerGroupsFilteringParams(const NxApexScene& apexScene ,
        NxGroupsFilteringParams64& params) const
{
	NX_READ_ZONE();
	FieldSamplerScene* scene = getFieldSamplerScene(apexScene);
	if (scene != NULL)
	{
		DYNAMIC_CAST(FieldSamplerManager*)(scene->getManager())->getFieldSamplerGroupsFilteringParams(params);
		return true;
	}
	return false;
}
#endif

#if NX_SDK_VERSION_MAJOR == 3

bool ModuleFieldSampler::setFieldSamplerWeightedCollisionFilterCallback(const NxApexScene& apexScene,NxFieldSamplerWeightedCollisionFilterCallback *callback)
{
	NX_WRITE_ZONE();
	FieldSamplerScene* scene = getFieldSamplerScene(apexScene);
	if (scene != NULL)
	{
		DYNAMIC_CAST(FieldSamplerManager*)(scene->getManager())->setFieldSamplerWeightedCollisionFilterCallback(callback);
		return true;
	}
	return false;

}

void ModuleFieldSampler::enablePhysXMonitor(const NxApexScene& apexScene, bool enable)
{
	NX_WRITE_ZONE();
	getFieldSamplerScene(apexScene)->enablePhysXMonitor(enable);	
}

void ModuleFieldSampler::setPhysXMonitorFilterData(const NxApexScene& apexScene, physx::PxFilterData filterData)
{
	NX_WRITE_ZONE();
	getFieldSamplerScene(apexScene)->setPhysXFilterData(filterData);
}


PxU32 ModuleFieldSampler::createForceSampleBatch(const NxApexScene& apexScene, PxU32 maxCount, const physx::PxFilterData filterData)
{
	NX_WRITE_ZONE();
	FieldSamplerScene* fsScene = getFieldSamplerScene(apexScene);
	if (fsScene)
	{
		return fsScene->createForceSampleBatch(maxCount, filterData);
	}
	return (PxU32)~0;
}


void ModuleFieldSampler::releaseForceSampleBatch(const NxApexScene& apexScene, PxU32 batchId)
{
	NX_WRITE_ZONE();
	FieldSamplerScene* fsScene = getFieldSamplerScene(apexScene);
	if (fsScene)
	{
		fsScene->releaseForceSampleBatch(batchId);
	}
}


void ModuleFieldSampler::submitForceSampleBatch(const NxApexScene& apexScene, PxU32 batchId,
												PxVec4* forces, const PxU32 forcesStride,
												const PxVec3* positions, const PxU32 positionsStride, 
												const PxVec3* velocities, const PxU32 velocitiesStride,
												const PxF32* mass, const PxU32 massStride,
												const PxU32* indices, const PxU32 numIndices)
{
	NX_WRITE_ZONE();
	FieldSamplerScene* fsScene = getFieldSamplerScene(apexScene);
	if (fsScene)
	{
		fsScene->submitForceSampleBatch(batchId, forces, forcesStride, positions, positionsStride, velocities, velocitiesStride, mass, massStride, indices, numIndices);
	}
}


#endif

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED

}
}
} // end namespace physx::apex
