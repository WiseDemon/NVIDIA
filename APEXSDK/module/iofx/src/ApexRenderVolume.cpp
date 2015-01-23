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

#include "ApexRenderVolume.h"
#include "ModuleIofx.h"
#include "IofxAsset.h"
#include "IofxActor.h"
#include "IofxScene.h"
#include "NiApexScene.h"

#include "PsArray.h"

namespace physx
{
namespace apex
{
namespace iofx
{

ApexRenderVolume::ApexRenderVolume(IofxScene& scene, const PxBounds3& b, PxU32 priority, bool allIofx)
	: mPriority(priority)
	, mAllIofx(allIofx)
	, mPendingDelete(false)
	, mScene(scene)
{
	setOwnershipBounds(b);

	mScene.mAddedRenderVolumesLock.lock();
	mScene.mAddedRenderVolumes.pushBack(this);
	mScene.mAddedRenderVolumesLock.unlock();
}

ApexRenderVolume::~ApexRenderVolume()
{
	ApexRenderable::renderDataLock();
	while (mIofxActors.size())
	{
		NxIofxActor* iofx = mIofxActors.popBack();
		iofx->release();
	}
	ApexRenderable::renderDataUnLock();
}

void ApexRenderVolume::destroy()
{
	if (!mPendingDelete)
	{
		mPendingDelete = true;

		mScene.mDeletedRenderVolumesLock.lock();
		mScene.mDeletedRenderVolumes.pushBack(this);
		mScene.mDeletedRenderVolumesLock.unlock();
	}
}

void ApexRenderVolume::release()
{
	if (!mPendingDelete)
	{
		mScene.mModule->releaseRenderVolume(*this);
	}
}

bool ApexRenderVolume::addIofxAsset(NxIofxAsset& iofx)
{
	NX_WRITE_ZONE();
	if (mAllIofx || mPendingDelete)
	{
		return false;
	}

	ApexRenderable::renderDataLock();
	mIofxAssets.pushBack(&iofx);
	ApexRenderable::renderDataUnLock();
	return true;
}

bool ApexRenderVolume::addIofxActor(NxIofxActor& iofx)
{
	if (mPendingDelete)
	{
		return false;
	}

	ApexRenderable::renderDataLock();
	mIofxActors.pushBack(&iofx);
	ApexRenderable::renderDataUnLock();
	return true;
}

bool ApexRenderVolume::removeIofxActor(const NxIofxActor& iofx)
{
	ApexRenderable::renderDataLock();
	for (PxU32 i = 0 ; i < mIofxActors.size() ; i++)
	{
		if (mIofxActors[ i ] == &iofx)
		{
			mIofxActors.replaceWithLast(i);
			ApexRenderable::renderDataUnLock();
			return true;
		}
	}

	ApexRenderable::renderDataUnLock();
	return false;
}

void ApexRenderVolume::setPosition(const PxVec3& pos)
{
	NX_WRITE_ZONE();
	physx::PxVec3 ext = mRenderBounds.getExtents();
	ApexRenderable::mRenderBounds = physx::PxBounds3(pos - ext, pos + ext);
}

PxBounds3 ApexRenderVolume::getBounds() const
{
	if (mPendingDelete)
	{
		return PxBounds3::empty();
	}

	PxBounds3 b = PxBounds3::empty();
	physx::Array<NxIofxActor*>::ConstIterator i;
	for (i = mIofxActors.begin() ; i != mIofxActors.end() ; i++)
	{
		(*i)->lockRenderResources();
		b.include((*i)->getBounds());
		(*i)->unlockRenderResources();
	}

	return b;
}

void ApexRenderVolume::updateRenderResources(bool rewriteBuffers, void* userRenderData)
{
	URR_SCOPE;

	if (mPendingDelete)
	{
		return;
	}

	physx::Array<NxIofxActor*>::Iterator i;
	for (i = mIofxActors.begin() ; i != mIofxActors.end() ; i++)
	{
		(*i)->lockRenderResources();
		(*i)->updateRenderResources(rewriteBuffers, userRenderData);
		(*i)->unlockRenderResources();
	}
}


void ApexRenderVolume::dispatchRenderResources(NxUserRenderer& renderer)
{
	if (mPendingDelete)
	{
		return;
	}

	physx::Array<NxIofxActor*>::Iterator i;
	for (i = mIofxActors.begin() ; i != mIofxActors.end() ; i++)
	{
		(*i)->lockRenderResources();
		(*i)->dispatchRenderResources(renderer);
		(*i)->unlockRenderResources();
	}
}

// Callers must acquire render lock for thread safety
bool ApexRenderVolume::affectsIofxAsset(const NxIofxAsset& iofx) const
{
	NX_READ_ZONE();
	if (mPendingDelete)
	{
		return false;
	}

	if (mAllIofx)
	{
		return true;
	}

	physx::Array<NxIofxAsset*>::ConstIterator i;
	for (i = mIofxAssets.begin() ; i != mIofxAssets.end() ; i++)
	{
		if (&iofx == *i)
		{
			return true;
		}
	}

	return false;
}

}
}
} // namespace physx::apex
