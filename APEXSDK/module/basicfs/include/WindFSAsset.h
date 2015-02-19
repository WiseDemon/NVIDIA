/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __WIND_FS_ASSET_H__
#define __WIND_FS_ASSET_H__

#include "BasicFSAsset.h"
#include "WindFSAssetPreview.h"
#include "WindFSAssetParams.h"
#include "WindFSActorParams.h"
#include "ApexAuthorableObject.h"

namespace physx
{
namespace apex
{

class NxRenderMeshAsset;

namespace basicfs
{

class WindFSActor;

class WindFSAsset : public BasicFSAsset
{
	friend class BasicFSAssetDummyAuthoring;
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	WindFSAsset(ModuleBasicFS*, NxResourceList&, const char*);
	WindFSAsset(ModuleBasicFS*, NxResourceList&, NxParameterized::Interface*, const char*);
	~WindFSAsset();

	/* NxApexAsset */
	NxAuthObjTypeID			getObjTypeID() const
	{
		return mAssetTypeID;
	}
	const char* 			getObjTypeName() const
	{
		return getClassName();
	}

	/* NxApexInterface */
	virtual void			release()
	{
		mModule->mSdk->releaseAsset(*this);
	}

	// TODO: implement forceLoadAssets
	PxU32					forceLoadAssets()
	{
		return 0;
	}

	NxParameterized::Interface* getAssetNxParameterized() const
	{
		return mParams;
	}

	NxParameterized::Interface* releaseAndReturnNxParameterizedInterface(void)
	{
		NxParameterized::Interface* ret = mParams;
		mParams = NULL;
		release();
		return ret;
	}

	/* NxBasicFSAsset specific methods */
	void                    destroy();

	/**
	* \brief Apply any changes that may been made to the NxParameterized::Interface on this asset.
	*/
	virtual void applyEditingChanges(void)
	{
		APEX_INVALID_OPERATION("Not yet implemented!");
	}

	NxWindFSPreview*	createWindFSPreview(const NxWindFSPreviewDesc& desc, NxApexAssetPreviewScene* previewScene);
	NxWindFSPreview*	createWindFSPreviewImpl(const NxWindFSPreviewDesc& desc, WindFSAsset* TurboAsset, NxApexAssetPreviewScene* previewScene);
	void				releaseWindFSPreview(NxWindFSPreview& preview);

	NxParameterized::Interface* getDefaultActorDesc();
	virtual NxApexActor* createApexActor(const NxParameterized::Interface& /*parms*/, NxApexScene& /*apexScene*/);

	NxParameterized::Interface* getDefaultAssetPreviewDesc();
	virtual NxApexAssetPreview* createApexAssetPreview(const NxParameterized::Interface& /*params*/, NxApexAssetPreviewScene* previewScene);

	virtual bool isValidForActorCreation(const ::NxParameterized::Interface& /*parms*/, NxApexScene& /*apexScene*/) const
	{
		return true; // todo implement this method
	}

	/* Typical asset members */
	static const char* 			getClassName() // return to protected
	{
		return NX_WIND_FS_AUTHORING_TYPE_NAME;
	}

protected:

	static NxAuthObjTypeID		mAssetTypeID;

	WindFSAssetParams*			mParams;
	WindFSActorParams*			mDefaultActorParams;
	WindFSPreviewParams*		mDefaultPreviewParams;

	friend class ModuleBasicFS;
	friend class WindFSActor;
	friend class WindFSAssetPreview;
	template <class T_Module, class T_Asset, class T_AssetAuthoring> friend class physx::apex::ApexAuthorableObject;

};

class WindFSAssetAuthoring : public WindFSAsset, public ApexAssetAuthoring, public NxBasicFSAssetAuthoring
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	WindFSAssetAuthoring(ModuleBasicFS* m, NxResourceList& l) :
		WindFSAsset(m, l, "WindFSAssetAuthoring") {}

	WindFSAssetAuthoring(ModuleBasicFS* m, NxResourceList& l, const char* name) :
		WindFSAsset(m, l, name) {}

	WindFSAssetAuthoring(ModuleBasicFS* m, NxResourceList& l, NxParameterized::Interface* params, const char* name) :
		WindFSAsset(m, l, params, name) {}

	~WindFSAssetAuthoring() {}
	void                    destroy()
	{
		delete this;
	}

	/* NxApexAssetAuthoring */
	const char* 			getName(void) const
	{
		return WindFSAsset::getName();
	}
	const char* 			getObjTypeName() const
	{
		return WindFSAsset::getClassName();
	}
	bool					prepareForPlatform(physx::apex::NxPlatformTag)
	{
		APEX_INVALID_OPERATION("Not Implemented.");
		return false;
	}

	void					setToolString(const char* toolName, const char* toolVersion, PxU32 toolChangelist)
	{
		ApexAssetAuthoring::setToolString(toolName, toolVersion, toolChangelist);
	}

	/* NxApexInterface */
	virtual void			release()
	{
		mModule->mSdk->releaseAssetAuthoring(*this);
	}

	NxParameterized::Interface* getNxParameterized() const
	{
		return WindFSAsset::getAssetNxParameterized();
	}

	NxParameterized::Interface* releaseAndReturnNxParameterizedInterface(void)
	{
		return WindFSAsset::releaseAndReturnNxParameterizedInterface();
	}

};

}
}
} // end namespace physx::apex

#endif