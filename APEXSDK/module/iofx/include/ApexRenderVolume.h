/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __APEX_RENDER_VOLUME_H__
#define __APEX_RENDER_VOLUME_H__

#include "NxApex.h"
#include "NxApexRenderVolume.h"
#include "PsArray.h"
#include "ApexInterface.h"
#include "ApexRenderable.h"
#include "ApexRWLockable.h"
#include "ReadCheck.h"
#include "WriteCheck.h"

namespace physx
{
namespace apex
{

class NxIofxAsset;
class NxIofxActor;

namespace iofx
{
class IofxScene;

class ApexRenderVolume : public NxApexRenderVolume, public ApexRenderable, public NxApexResource, public ApexResource, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ApexRenderVolume(IofxScene& scene, const PxBounds3& b, PxU32 priority, bool allIofx);
	~ApexRenderVolume();

	// NxApexResource methods
	void				release();
	void			    destroy();

	physx::PxU32	    getListIndex() const
	{
		return m_listIndex;
	}
	void	            setListIndex(NxResourceList& list, physx::PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}

	// NxApexRenderable API
	void				lockRenderResources()
	{
		ApexRenderable::renderDataLock();
	}
	void				unlockRenderResources()
	{
		ApexRenderable::renderDataUnLock();
	}
	void				updateRenderResources(bool rewriteBuffers, void* userRenderData);
	void				dispatchRenderResources(NxUserRenderer&);

	void				setOwnershipBounds(const PxBounds3& b)
	{
		NX_WRITE_ZONE();
		ApexRenderable::mRenderBounds = b;
	}
	PxBounds3			getOwnershipBounds(void) const
	{
		NX_READ_ZONE();
		return ApexRenderable::getBounds();
	}
	PxBounds3			getBounds() const;

	// methods for use by IOS or IOFX actor
	bool				addIofxActor(NxIofxActor& iofx);
	bool				removeIofxActor(const NxIofxActor& iofx);

	bool				addIofxAsset(NxIofxAsset& iofx);
	void				setPosition(const PxVec3& pos);

	bool				getAffectsAllIofx() const
	{
		NX_READ_ZONE();
		return mAllIofx;
	}
	NxIofxActor* const* getIofxActorList(PxU32& count) const
	{
		NX_READ_ZONE();
		count = mIofxActors.size();
		return count ? &mIofxActors.front() : NULL;
	}
	NxIofxAsset* const* getIofxAssetList(PxU32& count) const
	{
		NX_READ_ZONE();
		count = mIofxAssets.size();
		return count ? &mIofxAssets.front() : NULL;
	}
	PxVec3				getPosition() const
	{
		NX_READ_ZONE();
		return mRenderBounds.getCenter();
	}
	PxU32				getPriority() const
	{
		NX_READ_ZONE();
		return mPriority;
	}
	bool				affectsIofxAsset(const NxIofxAsset& iofx) const;

protected:
	// bounds is stored in ApexRenderable::mRenderBounds
	PxU32						 mPriority;
	bool						 mAllIofx;
	bool                         mPendingDelete;
	IofxScene&                   mScene;
	physx::Array<NxIofxAsset*>	 mIofxAssets;
	physx::Array<NxIofxActor*>	mIofxActors;
};

}
}
} // namespace apex

#endif // __APEX_RENDER_VOLUME_H__
