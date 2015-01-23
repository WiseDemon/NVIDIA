/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef BASIC_FS_ASSET_H
#define BASIC_FS_ASSET_H

#include "NxApex.h"

#include "NxBasicFSAsset.h"
#include "ApexSDKHelpers.h"
#include "ApexInterface.h"
#include "ModuleBasicFS.h"
#include "ApexAssetAuthoring.h"
#include "ApexString.h"
#include "ApexAssetTracker.h"
#include "ApexAuthorableObject.h"
#include "ApexRWLockable.h"
#include "NiFieldBoundary.h"

namespace physx
{
namespace apex
{

class NxRenderMeshAsset;

namespace basicfs
{

class BasicFSActor;

///p,q -> p cross q = n (n - must be normalized!)
PX_INLINE void BuildPlaneBasis(const PxVec3& n, PxVec3& p, PxVec3& q)
{
	float nzSqr = n.z * n.z;
	if (nzSqr > 0.5f)
	{
		// choose p in y-z plane
		const float k = PxSqrt(n.y * n.y + nzSqr);
		// k can not be zero here
		const float rk = (1 / k);
		p.x = 0;
		p.y = -n.z * rk;
		p.z = n.y * rk;
		// set q = n cross p
		q.x = k;
		q.y = -n.x * p.z;
		q.z = n.x * p.y;
	}
	else
	{
		// choose p in x-y plane
		const float k = PxSqrt(n.x * n.x + n.y * n.y);
		// k can be zero in case n is zero
		const float rk = (k > 0) ? (1 / k) : 0;
		p.x = -n.y * rk;
		p.y = n.x * rk;
		p.z = 0;
		// set q = n cross p
		q.x = -n.z * p.y;
		q.y = n.z * p.x;
		q.z = k;
	}
}

class BasicFSAsset : public NxBasicFSAsset, public NxApexResource, public ApexResource, public ApexRWLockable
{
	friend class BasicFSAssetDummyAuthoring;
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	BasicFSAsset(ModuleBasicFS*, const char*);
	virtual ~BasicFSAsset();

	/* NxApexAsset */
	const char* 			getName() const
	{
		return mName.c_str();
	}

	// TODO: implement forceLoadAssets
	PxU32					forceLoadAssets()
	{
		return 0;
	}

	/* NxApexResource, ApexResource */
	PxU32					getListIndex() const
	{
		return m_listIndex;
	}
	void					setListIndex(class NxResourceList& list, PxU32 index)
	{
		m_list = &list;
		m_listIndex = index;
	}

	/**
	* \brief Apply any changes that may been made to the NxParameterized::Interface on this asset.
	*/
	virtual void applyEditingChanges(void)
	{
		APEX_INVALID_OPERATION("Not yet implemented!");
	}

	NxParameterized::Interface* getDefaultActorDesc() = 0;
	virtual NxApexActor* createApexActor(const NxParameterized::Interface& /*parms*/, NxApexScene& /*apexScene*/) = 0;

	NxParameterized::Interface* getDefaultAssetPreviewDesc()
	{
		APEX_INVALID_OPERATION("Not yet implemented!");
		return NULL;
	}

	virtual NxApexAssetPreview* createApexAssetPreview(const NxParameterized::Interface& /*params*/, NxApexAssetPreviewScene* /*previewScene*/)
	{
		APEX_INVALID_OPERATION("Not yet implemented!");
		return NULL;
	}

	virtual bool isValidForActorCreation(const ::NxParameterized::Interface& /*parms*/, NxApexScene& /*apexScene*/) const
	{
		return true; // todo implement this method
	}

	virtual bool isDirty() const
	{
		return false;
	}


protected:

	ModuleBasicFS* 				mModule;
	NxResourceList				mFSActors;
	ApexSimpleString			mName;

	friend class ModuleBasicFS;
	friend class BasicFSActor;
};

}
}
} // end namespace physx::apex

#endif // BASIC_FS_ASSET_H
