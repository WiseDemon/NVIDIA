/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef FORCEFIELD_ASSET_H
#define FORCEFIELD_ASSET_H

#include "NxApex.h"
#include "PsShare.h"
#include "NxForceFieldAsset.h"
#include "NxForceFieldActor.h"
#include "NxForceFieldPreview.h"
#include "ApexSDKHelpers.h"
#include "ApexInterface.h"
#include "ModuleForceField.h"
#include "ApexAssetAuthoring.h"
#include "ApexString.h"
#include "ApexAssetTracker.h"
#include "ApexAuthorableObject.h"
#include "ForceFieldAssetParams.h"
#include "ReadCheck.h"
#include "WriteCheck.h"
#include "ApexAuthorableObject.h"

namespace physx
{
namespace apex
{
namespace forcefield
{

class NxForceFieldActorDesc : public NxApexDesc
{
public:
#if NX_SDK_VERSION_MAJOR == 2
	physx::NxGroupsMask64 samplerFilterData;
	physx::NxGroupsMask64 boundaryFilterData;
#else
	physx::PxFilterData samplerFilterData;
	physx::PxFilterData boundaryFilterData;
#endif
	physx::PxMat44 initialPose;
	
	//deprecated, has no effect
	physx::PxF32	 scale;
	PxActor* nxActor;
	const char* actorName;

	/**
	\brief constructor sets to default.
	*/
	PX_INLINE NxForceFieldActorDesc() : NxApexDesc()
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
		initialPose = PxMat44::createIdentity();
		scale = 1.0f;
		nxActor = NULL;
		actorName = NULL;
	}
};

/**
\brief Descriptor for a ForceField Asset
*/
class NxForceFieldPreviewDesc
{
public:
	NxForceFieldPreviewDesc() :
		mPose(physx::PxMat44()),
		mIconScale(1.0f),
		mPreviewDetail(APEX_FORCEFIELD::FORCEFIELD_DRAW_ICON)
	{
		mPose = PxMat44::createIdentity();
	};

	/**
	\brief The pose that translates from explosion preview coordinates to world coordinates.
	*/
	physx::PxMat44							mPose;
	/**
	\brief The scale of the icon.
	*/
	physx::PxF32							mIconScale;
	/**
	\brief The detail options of the preview drawing
	*/
	physx::PxU32							mPreviewDetail;
};


class ForceFieldActor;

class ForceFieldAsset : public NxForceFieldAsset, public NxApexResource, public ApexResource, public ApexRWLockable
{
	friend class ForceFieldAssetDummyAuthoring;
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ForceFieldAsset(ModuleForceField*, NxResourceList&, const char* name);
	ForceFieldAsset(ModuleForceField*, NxResourceList&, NxParameterized::Interface*, const char*);

	~ForceFieldAsset();

	/* NxApexAsset */
	const char* 					getName() const
	{
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
	virtual void					release()
	{
		mModule->mSdk->releaseAsset(*this);
	}

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

	/* NxForceFieldAsset specific methods */
	void							releaseForceFieldActor(NxForceFieldActor&);
	const ForceFieldAssetParams&	getForceFieldParameters() const
	{
		return *mParams;
	}
	physx::PxF32					getDefaultScale() const
	{
		NX_READ_ZONE();
		return mParams->defScale;
	}
	void							destroy();
	NxForceFieldPreview*			createForceFieldPreview(const NxForceFieldPreviewDesc& desc, NxApexAssetPreviewScene* previewScene);
	NxForceFieldPreview*			createForceFieldPreviewImpl(const NxForceFieldPreviewDesc& desc, ForceFieldAsset* ForceFieldAsset, NxApexAssetPreviewScene* previewScene);
	void							releaseForceFieldPreview(NxForceFieldPreview& preview);

	const NxParameterized::Interface* getAssetNxParameterized() const
	{
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
	NxParameterized::Interface* getDefaultActorDesc();
	NxParameterized::Interface* getDefaultAssetPreviewDesc();
	virtual NxApexActor* createApexActor(const NxParameterized::Interface& /*parms*/, NxApexScene& /*apexScene*/);
	virtual NxApexAssetPreview* createApexAssetPreview(const NxParameterized::Interface& params, NxApexAssetPreviewScene* previewScene);

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

protected:
	static const char* 				getClassName()
	{
		return NX_FORCEFIELD_AUTHORING_TYPE_NAME;
	}
	static NxAuthObjTypeID			mAssetTypeID;

	ModuleForceField*				mModule;

	NxResourceList					mForceFieldActors;
	ApexSimpleString				mName;
	ForceFieldAssetParams*			mParams;
	ForceFieldActorParams*			mDefaultActorParams;
	ForceFieldAssetPreviewParams*	mDefaultPreviewParams;

	GenericForceFieldKernelParams*	mGenericParams;
	RadialForceFieldKernelParams*	mRadialParams;
	ForceFieldFalloffParams*		mFalloffParams;
	ForceFieldNoiseParams*			mNoiseParams;

	void							initializeAssetNameTable();

	friend class ModuleForceField;
	friend class ForceFieldActor;
	template <class T_Module, class T_Asset, class T_AssetAuthoring> friend class physx::apex::ApexAuthorableObject;
};

#ifndef WITHOUT_APEX_AUTHORING
class ForceFieldAssetAuthoring : public ForceFieldAsset, public ApexAssetAuthoring, public NxForceFieldAssetAuthoring
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	/* NxForceFieldAssetAuthoring */
	ForceFieldAssetAuthoring(ModuleForceField* m, NxResourceList& l) :
		ForceFieldAsset(m, l, "ForceFieldAssetAuthoring") {}

	ForceFieldAssetAuthoring(ModuleForceField* m, NxResourceList& l, const char* name) :
		ForceFieldAsset(m, l, name) {}

	ForceFieldAssetAuthoring(ModuleForceField* m, NxResourceList& l, NxParameterized::Interface* params, const char* name) :
		ForceFieldAsset(m, l, params, name) {}

	~ForceFieldAssetAuthoring() {}

	void							destroy()
	{
		delete this;
	}

	/* NxApexAssetAuthoring */
	const char* 					getName(void) const
	{
		NX_READ_ZONE();
		return ForceFieldAsset::getName();
	}
	const char* 					getObjTypeName() const
	{
		return ForceFieldAsset::getClassName();
	}
	virtual bool					prepareForPlatform(physx::apex::NxPlatformTag)
	{
		APEX_INVALID_OPERATION("Not Implemented.");
		return false;
	}

	void setToolString(const char* toolName, const char* toolVersion, PxU32 toolChangelist)
	{
		ApexAssetAuthoring::setToolString(toolName, toolVersion, toolChangelist);
	}

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
	virtual NxParameterized::Interface* releaseAndReturnNxParameterizedInterface(void)
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

#endif // FORCEFIELD_ASSET_H
