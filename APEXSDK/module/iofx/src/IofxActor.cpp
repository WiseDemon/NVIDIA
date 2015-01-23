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
#include "NiApexSDK.h"
#include "NiApexScene.h"
#include "NxIofxActor.h"
#include "IofxActor.h"
#include "IofxSceneCPU.h"
#include "IofxSceneGPU.h"
#include "ApexRenderVolume.h"
#include "IosObjectData.h"
#include "IofxRenderData.h"

#include "ModuleIofx.h"

namespace physx
{
namespace apex
{
namespace iofx
{

IofxActor::IofxActor(NxApexAsset* renderAsset, IofxScene* iscene, IofxManager& mgr)
	: mRenderAsset(renderAsset)
	, mIofxScene(iscene)
	, mMgr(mgr)
	, mRenderVolume(NULL) // IOS will set this after creation
	, mSemantics(0)
	, mActiveRenderData(NULL)
{
	//asset.add(*this);

	mResultBounds.setEmpty();
	mResultRange.startIndex = 0;
	mResultRange.objectCount = 0;
	mResultVisibleCount = 0;

	addSelfToContext(*iscene->mApexScene->getApexContext());    // Add self to ApexScene
	addSelfToContext(*iscene);							      // Add self to IofxScene
}

IofxActor::~IofxActor()
{
}

void IofxActor::getPhysicalLodRange(physx::PxF32& min, physx::PxF32& max, bool& intOnly) const
{
	NX_READ_ZONE();
	PX_UNUSED(min);
	PX_UNUSED(max);
	PX_UNUSED(intOnly);
	APEX_INVALID_OPERATION("not implemented");
}


physx::PxF32 IofxActor::getActivePhysicalLod() const
{
	NX_READ_ZONE();
	APEX_INVALID_OPERATION("NxBasicIosActor does not support this operation");
	return -1.0f;
}


void IofxActor::forcePhysicalLod(physx::PxF32 lod)
{
	NX_WRITE_ZONE();
	PX_UNUSED(lod);
	APEX_INVALID_OPERATION("not implemented");
}

void IofxActor::release()
{
	if (mInRelease)
	{
		return;
	}
	mInRelease = true;
	destroy();
}


void IofxActor::destroy()
{
	if (mRenderVolume)
	{
		mRenderVolume->removeIofxActor(*this);
	}

	// Removes self from scenes and IOFX manager
	// should be called after mRenderVolume->removeIofxActor to avoid dead-lock!!!
	ApexActor::destroy();

	for (PxU32 i = 0 ; i < mRenderDataArray.size() ; i++)
	{
		IofxActorRenderData* renderData = mRenderDataArray[i];
		PX_DELETE(renderData);
	}
	mRenderDataArray.clear();

	delete this;
}

void IofxActor::lockRenderResources()
{
	return ApexRenderable::renderDataLock();
}

void IofxActor::unlockRenderResources()
{
	return ApexRenderable::renderDataUnLock();
}

void IofxActor::updateRenderResources(bool rewriteBuffers, void* userRenderData)
{
	URR_SCOPE;

	if (mActiveRenderData != NULL)
	{
		mActiveRenderData->updateRenderResources(rewriteBuffers, userRenderData);
	}
}

void IofxActor::dispatchRenderResources(NxUserRenderer& renderer)
{
	if (mActiveRenderData != NULL)
	{
		mActiveRenderData->dispatchRenderResources(renderer);
	}
}

bool IofxActor::prepareRenderResources(IosObjectBaseData* obj)
{
	mRenderBounds = mResultBounds;
	mRenderRange = mResultRange;
	mRenderVisibleCount = mResultVisibleCount;

	if (mRenderRange.objectCount > 0 && obj->renderData->checkSemantics(mSemantics))
	{
		const PxU32 instanceID = obj->renderData->getInstanceID();
		if (mRenderDataArray.size() <= instanceID)
		{
			mRenderDataArray.resize(instanceID + 1, NULL);
		}
		mActiveRenderData = mRenderDataArray[instanceID];
		if (mActiveRenderData == NULL)
		{
			if (mMgr.mIsMesh)
			{
				NxRenderMeshActor* renderMeshActor = loadRenderMeshActor(obj->maxObjectCount);
				mActiveRenderData = PX_NEW(IofxActorRenderDataMesh)(this, renderMeshActor);
			}
			else
			{
				NxApexAsset* spriteMaterialAsset = mRenderAsset;
				mActiveRenderData = PX_NEW(IofxActorRenderDataSprite)(this, spriteMaterialAsset);
			}
			mRenderDataArray[instanceID] = mActiveRenderData;
		}
		mActiveRenderData->setSharedRenderData(obj->renderData);
		return true;
	}
	else
	{
		mRenderBounds.setEmpty();
		mRenderRange.objectCount = 0;
		mRenderVisibleCount = 0;

		mActiveRenderData = NULL;
		return false;
	}
}

NxRenderMeshActor* IofxActor::loadRenderMeshActor(physx::PxU32 maxInstanceCount)
{
	NxRenderMeshActor* rmactor = NULL;

	NxRenderMeshActorDesc renderableMeshDesc;
	renderableMeshDesc.maxInstanceCount = maxInstanceCount;

	NxRenderMeshAsset* meshAsset = static_cast<NxRenderMeshAsset*>(mRenderAsset);
	if (meshAsset)
	{
		rmactor = meshAsset->createActor(renderableMeshDesc);
		if (rmactor)
		{
			//this call is important to prevent renderResource release in case when there are no instances!
			rmactor->setReleaseResourcesIfNothingToRender(false);
		}

		ApexActor* aa = NiGetApexSDK()->getApexActor(rmactor);
		if (aa)
		{
			aa->addSelfToContext(*this);
		}
	}
	return rmactor;
}

}
}
} // namespace physx::apex
