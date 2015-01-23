/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef PARTICLES_EFFECT_PACKAGE_ASSET_H
#define PARTICLES_EFFECT_PACKAGE_ASSET_H

#include "NxApex.h"
#include "PsShare.h"
#include "NxEffectPackageAsset.h"
#include "NxEffectPackageActor.h"
#include "ApexSDKHelpers.h"
#include "ApexInterface.h"
#include "ModuleParticles.h"
#include "ApexAssetAuthoring.h"
#include "ApexString.h"
#include "ApexAssetTracker.h"
#include "ApexAuthorableObject.h"
#include "EffectPackageAssetParams.h"
#include "ApexRWLockable.h"

#include "ReadCheck.h"
#include "WriteCheck.h"
#include "ApexAuthorableObject.h"

namespace physx
{
namespace apex
{
namespace particles
{

class EffectPackageActor;
class ModuleParticles;

class EffectPackageAsset : public NxEffectPackageAsset, public NxApexResource, public ApexResource, public ApexRWLockable
{
	friend class EffectPackageAssetDummyAuthoring;
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	EffectPackageAsset(ModuleParticles*, NxResourceList&, const char* name);
	EffectPackageAsset(ModuleParticles*, NxResourceList&, NxParameterized::Interface*, const char*);

	~EffectPackageAsset();

	/* NxApexAsset */
	const char* 					getName() const
	{
		NX_READ_ZONE();
		return mName.c_str();
	}
	NxAuthObjTypeID					getObjTypeID() const
	{
		NX_READ_ZONE();
		return mAssetTypeID;
	}
	const char* 					getObjTypeName() const
	{
		NX_READ_ZONE();
		return getClassName();
	}

	physx::PxU32					forceLoadAssets();

	/* NxApexInterface */
	virtual void					release();

	/* NxApexResource, ApexResource */
	physx::PxU32					getListIndex() const
	{
		return m_listIndex;
	}

	void							setListIndex(class NxResourceList& list, physx::PxU32 index)
	{
		m_list = &list;
		m_listIndex = index;
	}

	/* NxEffectPackageAsset specific methods */
	void							releaseEffectPackageActor(NxEffectPackageActor&);
	const EffectPackageAssetParams&	getEffectPackageParameters() const
	{
		return *mParams;
	}
	physx::PxF32					getDefaultScale() const
	{
		return 1;
	}
	void							destroy();

	const NxParameterized::Interface* getAssetNxParameterized() const
	{
		NX_READ_ZONE();
		return mParams;
	}

	virtual PxF32 getDuration() const;
	virtual bool useUniqueRenderVolume() const;

	/**
	 * \brief Releases the ApexAsset but returns the NxParameterized::Interface and *ownership* to the caller.
	 */
	virtual NxParameterized::Interface* releaseAndReturnNxParameterizedInterface()
	{
		NX_WRITE_ZONE();
		NxParameterized::Interface* ret = mParams;
		mParams = NULL;
		release();
		return ret;
	}
	NxParameterized::Interface* getDefaultActorDesc();
	NxParameterized::Interface* getDefaultAssetPreviewDesc();
	virtual NxApexActor* createApexActor(const NxParameterized::Interface& /*parms*/, NxApexScene& /*apexScene*/);
	virtual NxApexAssetPreview* createApexAssetPreview(const ::NxParameterized::Interface& /*params*/, NxApexAssetPreviewScene* /*previewScene*/)
	{
		NX_WRITE_ZONE();
		PX_ALWAYS_ASSERT();
		return NULL;
	}

	virtual bool isValidForActorCreation(const ::NxParameterized::Interface& /*parms*/, NxApexScene& /*apexScene*/) const
	{
		NX_READ_ZONE();
		return true; // TODO implement this method
	}

	virtual bool isDirty() const
	{
		NX_READ_ZONE();
		return false;
	}

	static NxAuthObjTypeID			mAssetTypeID;

protected:
	static const char* 				getClassName()
	{
		return NX_PARTICLES_EFFECT_PACKAGE_AUTHORING_TYPE_NAME;
	}

	ModuleParticles*				mModule;

	NxResourceList					mEffectPackageActors;
	ApexSimpleString				mName;
	EffectPackageAssetParams*			mParams;
	EffectPackageActorParams*			mDefaultActorParams;

	void							initializeAssetNameTable();

	friend class ModuleParticlesE;
	friend class EffectPackageActor;
	template <class T_Module, class T_Asset, class T_AssetAuthoring> friend class physx::apex::ApexAuthorableObject;
};

#ifndef WITHOUT_APEX_AUTHORING
class EffectPackageAssetAuthoring : public EffectPackageAsset, public ApexAssetAuthoring, public NxEffectPackageAssetAuthoring
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	/* NxEffectPackageAssetAuthoring */
	EffectPackageAssetAuthoring(ModuleParticles* m, NxResourceList& l) :
		EffectPackageAsset(m, l, "EffectPackageAssetAuthoring") {}

	EffectPackageAssetAuthoring(ModuleParticles* m, NxResourceList& l, const char* name) :
		EffectPackageAsset(m, l, name) {}

	EffectPackageAssetAuthoring(ModuleParticles* m, NxResourceList& l, NxParameterized::Interface* params, const char* name) :
		EffectPackageAsset(m, l, params, name) {}

	~EffectPackageAssetAuthoring() {}

	void							destroy()
	{
		delete this;
	}

	/* NxApexAssetAuthoring */
	const char* 					getName() const
	{
		NX_READ_ZONE();
		return EffectPackageAsset::getName();
	}
	const char* 					getObjTypeName() const
	{
		NX_READ_ZONE();
		return EffectPackageAsset::getClassName();
	}
	virtual bool					prepareForPlatform(physx::apex::NxPlatformTag)
	{
		NX_WRITE_ZONE();
		APEX_INVALID_OPERATION("Not Implemented.");
		return false;
	}

	void setToolString(const char* toolName, const char* toolVersion, PxU32 toolChangelist)
	{
		NX_WRITE_ZONE();
		ApexAssetAuthoring::setToolString(toolName, toolVersion, toolChangelist);
	}

	// from ApexAssetAuthoring
	virtual void setToolString(const char* toolString);

	/* NxApexInterface */
	virtual void					release()
	{
		mModule->mSdk->releaseAssetAuthoring(*this);
	}

	NxParameterized::Interface* getNxParameterized() const
	{
		return (NxParameterized::Interface*)getAssetNxParameterized();
	}
	/**
	 * \brief Releases the ApexAsset but returns the NxParameterized::Interface and *ownership* to the caller.
	 */
	virtual NxParameterized::Interface* releaseAndReturnNxParameterizedInterface()
	{
		NX_WRITE_ZONE();
		NxParameterized::Interface* ret = mParams;
		mParams = NULL;
		release();
		return ret;
	}
};
#endif

}
}
} // end namespace physx::apex

#endif // PARTICLES_EFFECT_PACKAGE_ASSET_H
