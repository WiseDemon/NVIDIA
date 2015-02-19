/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef PARTICLE_IOS_ASSET_H
#define PARTICLE_IOS_ASSET_H

#include "NxApex.h"
#include "NxIofxAsset.h"
#include "NxParticleIosAsset.h"
#include "NiInstancedObjectSimulation.h"
#include "ApexInterface.h"
#include "ApexSDKHelpers.h"
#include "ApexAssetAuthoring.h"
#include "ApexString.h"
#include "NiResourceProvider.h"
#include "ApexAuthorableObject.h"
#include "ParticleIosAssetParam.h"
#include "ApexAssetTracker.h"
#include "PsShare.h"
#include "ApexRWLockable.h"
#include "ReadCheck.h"
#include "WriteCheck.h"
#include "ApexAuthorableObject.h"

namespace physx
{
namespace apex
{

namespace iofx
{
class NxIofxAsset;
}

namespace pxparticleios
{

class ModuleParticleIos;
class ParticleIosActor;


/**
\brief Descriptor needed to create a ParticleIOS Actor
*/
class NxParticleIosActorDesc : public NxApexDesc
{
public:
	///Radius of a particle (overrides authered value)
	physx::PxF32				radius;
	///Density of a particle (overrides authered value)
	physx::PxF32				density;

	/**
	\brief constructor sets to default.
	*/
	PX_INLINE NxParticleIosActorDesc() : NxApexDesc()
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
		if (!NxApexDesc::isValid())
		{
			return false;
		}

		return true;
	}

private:

	PX_INLINE void init()
	{
		// authored values will be used where these default values remain
		radius = 0.0f;
		density = 0.0f;
	}
};

class ParticleIosAsset : public NxParticleIosAsset,
	public NxApexResource,
	public ApexResource,
	public ApexRWLockable
{
	friend class ParticleIosAssetDummyAuthoring;
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ParticleIosAsset(ModuleParticleIos*, NxResourceList&, const char*);
	ParticleIosAsset(ModuleParticleIos* module, NxResourceList&, NxParameterized::Interface*, const char*);
	~ParticleIosAsset();

	// NxApexAsset
	void							release();
	const char*						getName(void) const
	{
		return mName.c_str();
	}
	NxAuthObjTypeID					getObjTypeID() const
	{
		return mAssetTypeID;
	}
	const char* 					getObjTypeName() const
	{
		return getClassName();
	}
	physx::PxU32					forceLoadAssets();

	NxApexActor*					createIosActor(NxApexScene& scene, NxIofxAsset* iofxAsset);
	void							releaseIosActor(NxApexActor&);
	bool							getSupportsDensity() const
	{
		NX_READ_ZONE();
		return mParams->DensityBuffer;
	}
	bool							isValidForActorCreation(const ::NxParameterized::Interface& /*actorParams*/, NxApexScene& /*apexScene*/) const
	{
		return true;
	}

	bool							isDirty() const
	{
		return false;
	}


	// Private API for this module only
	ParticleIosActor*               getIosActorInScene(NxApexScene& scene, bool mesh) const;

	// NxApexResource methods
	void							setListIndex(NxResourceList& list, physx::PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}
	physx::PxU32					getListIndex() const
	{
		return m_listIndex;
	}

	physx::PxF32					getParticleRadius() const
	{
		NX_READ_ZONE();
		return mParams->particleRadius;
	}
	//physx::PxF32					getRestDensity() const				{ return mParams->restDensity; }
	physx::PxF32					getMaxInjectedParticleCount() const
	{
		NX_READ_ZONE();
		return mParams->maxInjectedParticleCount;
	}
	physx::PxU32					getMaxParticleCount() const
	{
		NX_READ_ZONE();
		return mParams->maxParticleCount;
	}
	const char*							getParticleTypeClassName() const
	{
		return mParams->particleType->className();
	}
	const ParticleIosAssetParam* getParticleDesc() const
	{
		return mParams;
	}
	physx::PxF32					getParticleMass() const
	{
		NX_READ_ZONE();
		return mParams->particleMass;
	}

	const NxParameterized::Interface* getAssetNxParameterized() const
	{
		NX_READ_ZONE();
		return mParams;
	}
	/**
	 * \brief Releases the ApexAsset but returns the NxParameterized::Interface and *ownership* to the caller.
	 */
	virtual NxParameterized::Interface* releaseAndReturnNxParameterizedInterface(void)
	{
		NxParameterized::Interface* ret = mParams;
		mParams = NULL;
		release();
		return ret;
	}

	NxParameterized::Interface* getDefaultActorDesc()
	{
		NX_READ_ZONE();
		APEX_INVALID_OPERATION("Not yet implemented!");
		return NULL;
	}

	NxParameterized::Interface* getDefaultAssetPreviewDesc()
	{
		NX_READ_ZONE();
		APEX_INVALID_OPERATION("Not yet implemented!");
		return NULL;
	}

	virtual NxApexActor* createApexActor(const NxParameterized::Interface& /*parms*/, NxApexScene& /*apexScene*/)
	{
		NX_WRITE_ZONE();
		APEX_INVALID_OPERATION("Not yet implemented!");
		return NULL;
	}

	virtual NxApexAssetPreview* createApexAssetPreview(const NxParameterized::Interface& /*params*/, NxApexAssetPreviewScene* /*previewScene*/)
	{
		NX_WRITE_ZONE();
		APEX_INVALID_OPERATION("Not yet implemented!");
		return NULL;
	}

protected:
	virtual void					destroy();

	static NxAuthObjTypeID			mAssetTypeID;
	static const char* 				getClassName()
	{
		return NX_PARTICLE_IOS_AUTHORING_TYPE_NAME;
	}

	NxResourceList					mIosActorList;

	ModuleParticleIos*				mModule;
	ApexSimpleString				mName;

	ParticleIosAssetParam*			mParams;

	friend class ModuleParticleIos;
	friend class ParticleIosActor;
	friend class ParticleIosActorCPU;
	friend class ParticleIosActorGPU;
	template <class T_Module, class T_Asset, class T_AssetAuthoring> friend class physx::apex::ApexAuthorableObject;
	friend class ParticleIosAuthorableObject;
};

#ifndef WITHOUT_APEX_AUTHORING
class ParticleIosAssetAuthoring : public NxParticleIosAssetAuthoring, public ApexAssetAuthoring, public ParticleIosAsset
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ParticleIosAssetAuthoring(ModuleParticleIos* module, NxResourceList& list);
	ParticleIosAssetAuthoring(ModuleParticleIos* module, NxResourceList& list, const char* name);
	ParticleIosAssetAuthoring(ModuleParticleIos* module, NxResourceList& list, NxParameterized::Interface* params, const char* name);

	virtual void	release();

	const char* 	getName(void) const
	{
		NX_READ_ZONE();
		return mName.c_str();
	}
	const char* 	getObjTypeName() const
	{
		NX_READ_ZONE();
		return ParticleIosAsset::getClassName();
	}
	virtual bool	prepareForPlatform(physx::apex::NxPlatformTag)
	{
		NX_WRITE_ZONE();
		APEX_INVALID_OPERATION("Not Implemented.");
		return false;
	}
	void			setToolString(const char* toolName, const char* toolVersion, PxU32 toolChangelist)
	{
		NX_WRITE_ZONE();
		ApexAssetAuthoring::setToolString(toolName, toolVersion, toolChangelist);
	}

	void			setParticleRadius(physx::PxF32 radius)
	{
		mParams->particleRadius = radius;
	}
	//void			setRestDensity( physx::PxF32 density )			{ mParams->restDensity = density; }
	void			setMaxInjectedParticleCount(physx::PxF32 count)
	{
		mParams->maxInjectedParticleCount = count;
	}
	void			setMaxParticleCount(physx::PxU32 count)
	{
		mParams->maxParticleCount = count;
	}
	void			setParticleMass(physx::PxF32 mass)
	{
		mParams->particleMass = mass;
	}

	void			setCollisionGroupName(const char* collisionGroupName);
	void			setCollisionGroupMaskName(const char* collisionGroupMaskName);

	NxParameterized::Interface* getNxParameterized() const
	{
		return (NxParameterized::Interface*)getAssetNxParameterized();
	}
	/**
	 * \brief Releases the ApexAsset but returns the NxParameterized::Interface and *ownership* to the caller.
	 */
	virtual NxParameterized::Interface* releaseAndReturnNxParameterizedInterface(void)
	{
		NxParameterized::Interface* ret = mParams;
		mParams = NULL;
		release();
		return ret;
	}
};
#endif

}
}
} // namespace physx::apex

#endif // PARTICLE_IOS_ASSET_H
