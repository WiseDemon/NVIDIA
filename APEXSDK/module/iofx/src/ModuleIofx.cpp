/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "NxApex.h"
#include "PsShare.h"
#include "NiApexSDK.h"
#include "ModuleIofx.h"
#include "IofxAsset.h"
#include "NiApexScene.h"
#include "PxMemoryBuffer.h"
#include "IofxSceneCPU.h"
#include "IofxSceneGPU.h"
#include "IofxActor.h"
#include "ApexRenderVolume.h"

#include "ModulePerfScope.h"
using namespace iofx;

#include "NxLock.h"

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
    physx::PxU32 APEXsdkVersion,
    physx::PxU32 PhysXsdkVersion,
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

	/* Setup common module global variables */
	gApexSdk = inSdk;
	APEX_INIT_FOUNDATION();
	initModuleProfiling(inSdk, "IOFX");
	ModuleIofx* impl = PX_NEW(ModuleIofx)(inSdk);
	*niRef  = (NiModule*) impl;
	return (NxModule*) impl;
}
#else
/* Statically linking entry function */
void instantiateModuleIofx()
{
	NiApexSDK* sdk = NiGetApexSDK();
	initModuleProfiling(sdk, "IOFX");
	iofx::ModuleIofx* impl = PX_NEW(iofx::ModuleIofx)(sdk);
	sdk->registerExternalModule((NxModule*) impl, (NiModule*) impl);
}
#endif


namespace iofx
{
/* =================== ModuleIofx =================== */


NxAuthObjTypeID IofxAsset::mAssetTypeID;
#ifdef WITHOUT_APEX_AUTHORING

class IofxAssetDummyAuthoring : public NxApexAssetAuthoring, public UserAllocated
{
public:
	IofxAssetDummyAuthoring(ModuleIofx* module, NxResourceList& list, NxParameterized::Interface* params, const char* name)
	{
		PX_UNUSED(module);
		PX_UNUSED(list);
		PX_UNUSED(params);
		PX_UNUSED(name);
	}

	IofxAssetDummyAuthoring(ModuleIofx* module, NxResourceList& list, const char* name)
	{
		PX_UNUSED(module);
		PX_UNUSED(list);
		PX_UNUSED(name);
	}

	IofxAssetDummyAuthoring(ModuleIofx* module, NxResourceList& list)
	{
		PX_UNUSED(module);
		PX_UNUSED(list);
	}

	virtual void setToolString(const char* /*toolName*/, const char* /*toolVersion*/, PxU32 /*toolChangelist*/)
	{

	}


	virtual void release()
	{
		destroy();
	}

	// internal
	void destroy()
	{
		delete this;
	}

	/**
	* \brief Returns the name of this APEX authorable object type
	*/
	virtual const char* getObjTypeName() const
	{
		return NX_IOFX_AUTHORING_TYPE_NAME;
	}

	/**
	 * \brief Prepares a fully authored Asset Authoring object for a specified platform
	 */
	virtual bool prepareForPlatform(physx::apex::NxPlatformTag)
	{
		PX_ASSERT(0);
		return false;
	}

	const char* getName(void) const
	{
		return NULL;
	}

	/**
	* \brief Save asset's NxParameterized interface, may return NULL
	*/
	virtual NxParameterized::Interface* getNxParameterized() const
	{
		PX_ASSERT(0);
		return NULL;
	}

	virtual NxParameterized::Interface* releaseAndReturnNxParameterizedInterface(void)
	{
		PX_ALWAYS_ASSERT();
		return NULL;
	}
};

typedef ApexAuthorableObject<ModuleIofx, IofxAsset, IofxAssetDummyAuthoring> IofxAO;

#else
typedef ApexAuthorableObject<ModuleIofx, IofxAsset, IofxAssetAuthoring> IofxAO;
#endif

ModuleIofx::ModuleIofx(NiApexSDK* sdk)
{
	mSdk = sdk;
	mApiProxy = this;
	name = "IOFX";
	mModuleParams = NULL;
	mInteropDisabled = false;
	mCudaDisabled = false;
	mDeferredDisabled = false;

	/* Register this module's authorable object types and create their namespaces */
	const char* pName = IofxAssetParameters::staticClassName();
	IofxAO* AO = PX_NEW(IofxAO)(this, mAuthorableObjects, pName);
	IofxAsset::mAssetTypeID = AO->getResID();

	/* Register the NxParameterized factories */
	NxParameterized::Traits* traits = mSdk->getParameterizedTraits();
#	define PARAM_CLASS(clas) PARAM_CLASS_REGISTER_FACTORY(traits, clas)
#	include "IofxParamClasses.inc"
}

NxAuthObjTypeID ModuleIofx::getModuleID() const
{
	NX_READ_ZONE();
	return IofxAsset::mAssetTypeID;
}

ModuleIofx::~ModuleIofx()
{
	releaseModuleProfiling();
}

void ModuleIofx::destroy()
{
	/* Remove the NxParameterized factories */
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
#		define PARAM_CLASS(clas) PARAM_CLASS_REMOVE_FACTORY(traits, clas)
#		include "IofxParamClasses.inc"
	}
}

NxParameterized::Interface* ModuleIofx::getDefaultModuleDesc()
{
	NX_READ_ZONE();
	NxParameterized::Traits* traits = mSdk->getParameterizedTraits();

	if (!mModuleParams)
	{
		mModuleParams = DYNAMIC_CAST(IofxModuleParameters*)
		                (traits->createNxParameterized("IofxModuleParameters"));
		PX_ASSERT(mModuleParams);
	}
	else
	{
		mModuleParams->initDefaults();
	}

	return mModuleParams;
}

void ModuleIofx::init(const NxModuleIofxDesc&)
{
	NX_WRITE_ZONE();
}

physx::PxU32 ModuleIofx::forceLoadAssets()
{
	physx::PxU32 loadedAssetCount = 0;

	for (physx::PxU32 i = 0; i < mAuthorableObjects.getSize(); i++)
	{
		NiApexAuthorableObject* ao = static_cast<NiApexAuthorableObject*>(mAuthorableObjects.getResource(i));
		loadedAssetCount += ao->forceLoadAssets();
	}
	return loadedAssetCount;
}

NiModuleScene* ModuleIofx::createNiModuleScene(NiApexScene& scene, NiApexRenderDebug* debugRender)
{
#if defined(APEX_CUDA_SUPPORT)
	physx::PxGpuDispatcher* gd;
	{
		NX_READ_LOCK(scene);
		gd = scene.getTaskManager()->getGpuDispatcher();
	}

	if (gd && gd->getCudaContextManager()->contextIsValid())
	{
		return PX_NEW(IofxSceneGPU)(*this, scene, debugRender, mIofxScenes);
	}
	else
#endif
	{
		return PX_NEW(IofxSceneCPU)(*this, scene, debugRender, mIofxScenes);
	}
}

void ModuleIofx::releaseNiModuleScene(NiModuleScene& scene)
{
	IofxScene* is = DYNAMIC_CAST(IofxScene*)(&scene);
	is->destroy();
}

IofxScene* ModuleIofx::getIofxScene(const NxApexScene& apexScene)
{
	for (physx::PxU32 i = 0 ; i < mIofxScenes.getSize() ; i++)
	{
		IofxScene* is = DYNAMIC_CAST(IofxScene*)(mIofxScenes.getResource(i));
		if (is->mApexScene == &apexScene)
		{
			return is;
		}
	}

	PX_ASSERT(!"Unable to locate an appropriate IofxScene");
	return NULL;
}

const IofxScene* ModuleIofx::getIofxScene(const NxApexScene& apexScene) const
{
	for (physx::PxU32 i = 0 ; i < mIofxScenes.getSize() ; i++)
	{
		IofxScene* is = DYNAMIC_CAST(IofxScene*)(mIofxScenes.getResource(i));
		if (is->mApexScene == &apexScene)
		{
			return is;
		}
	}

	PX_ASSERT(!"Unable to locate an appropriate IofxScene");
	return NULL;
}

NxApexRenderVolume* ModuleIofx::createRenderVolume(const NxApexScene& apexScene, const PxBounds3& b, PxU32 priority, bool allIofx)
{
	NX_WRITE_ZONE();
	IofxScene* is = getIofxScene(apexScene);
	if (is)
	{
		return PX_NEW(ApexRenderVolume)(*is, b, priority, allIofx);
	}

	return NULL;
}

void ModuleIofx::releaseRenderVolume(NxApexRenderVolume& volume)
{
	NX_WRITE_ZONE();
	ApexRenderVolume* arv = DYNAMIC_CAST(ApexRenderVolume*)(&volume);
	arv->destroy();
}


NiIofxManager* ModuleIofx::createActorManager(const NxApexScene& scene, const NxIofxAsset& asset, const NiIofxManagerDesc& desc)
{
	NX_WRITE_ZONE();
	IofxScene* is = getIofxScene(scene);
	return is ? is->createIofxManager(asset, desc) : NULL;
}

NxApexRenderableIterator* ModuleIofx::createRenderableIterator(const NxApexScene& apexScene)
{
	NX_WRITE_ZONE();
	IofxScene* is = getIofxScene(apexScene);
	if (is)
	{
		return is->createRenderableIterator();
	}

	return NULL;
}


const NxTestBase* ModuleIofx::getTestBase(NxApexScene* apexScene) const
{
#if ENABLE_TEST
	const IofxScene* scene = getIofxScene(*apexScene);
	return static_cast<const NxTestBase*>(DYNAMIC_CAST(const IofxTestScene*)(scene));
#else
	PX_UNUSED(apexScene);
	return 0;
#endif
}

}
}
} // namespace physx::apex
