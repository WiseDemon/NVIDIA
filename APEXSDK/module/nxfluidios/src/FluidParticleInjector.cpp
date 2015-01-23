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
#if NX_SDK_VERSION_MAJOR == 2

#include "NxIofxAsset.h"
#include "NiApexScene.h"
#include "FluidParticleInjector.h"
#include "FluidIosActor.h"

#include "PxTask.h"
#include "PxTaskManager.h"

namespace physx
{
namespace apex
{
namespace nxfluidios
{

FluidParticleInjector::FluidParticleInjector(NxResourceList& list, FluidIosActor& actor)
	: mIosActor(&actor)
	, mIofxClient(NULL)
	, mVolume(NULL)
	, mLastRandomID(0)
	, mVolumeID(NiIofxActorID::NO_VOLUME)
	, mLODMaxDistance(0.0f)
	, mLODDistanceWeight(0.0f)
	, mLODSpeedWeight(0.0f)
	, mLODLifeWeight(0.0f)
	, mLODBias(1.0f)
	, mIsBackLogged(false)
	, mInjectedBenefit(0.0f)
	, mSimulatedCount(0)
	, mSimulatedBenefit(0.0f)
	, mLODNodeBenefit(0.0f)
	, mLODNodeResource(0.0f)
	, mResourceBudget(0)
	, mMaxInsertionCount(0)
{
	mRand.setSeed(actor.mParticleScene->getApexScene()->getSeed());

	list.add(*this);
}

void FluidParticleInjector::setObjectScale(PxF32 objectScale)
{
	PX_ASSERT(mIofxClient);
	NiIofxManagerClient::Params params;
	mIofxClient->getParams(params);
	params.objectScale = objectScale;
	mIofxClient->setParams(params);
}

void FluidParticleInjector::init(NxIofxAsset* iofxAsset)
{
	mIofxClient = mIosActor->mIofxMgr->createClient(iofxAsset, NiIofxManagerClient::Params());

	/* add this injector to the IOFX asset's context (so when the IOFX goes away our ::release() is called) */
	iofxAsset->addDependentActor(this);

	mRandomActorClassIDs.clear();
	if (iofxAsset->getMeshAssetCount() < 2)
	{
		mRandomActorClassIDs.pushBack(mIosActor->mIofxMgr->getActorClassID(mIofxClient, 0));
		return;
	}

	/* Cache actorClassIDs for this asset */
	physx::Array<PxU16> temp;
	for (PxU32 i = 0 ; i < iofxAsset->getMeshAssetCount() ; i++)
	{
		PxU32 w = iofxAsset->getMeshAssetWeight(i);
		PxU16 acid = mIosActor->mIofxMgr->getActorClassID(mIofxClient, (PxU16) i);
		for (PxU32 j = 0 ; j < w ; j++)
		{
			temp.pushBack(acid);
		}
	}

	mRandomActorClassIDs.reserve(temp.size());
	while (temp.size())
	{
		PxU32 index = (PxU32)mRand.getScaled(0, (PxF32)temp.size());
		mRandomActorClassIDs.pushBack(temp[ index ]);
		temp.replaceWithLast(index);
	}
}

void FluidParticleInjector::release()
{
	if (mInRelease)
	{
		return;
	}
	mInRelease = true;
	mIosActor->releaseInjector(*this);
}

void FluidParticleInjector::destroy()
{
	ApexActor::destroy();

	mIosActor->mIofxMgr->releaseClient(mIofxClient);

	delete this;
}

physx::PxTaskID FluidParticleInjector::getCompletionTaskID() const
{
	// Return ID of task that requires injections to be complete
	return mIosActor->mParticleScene->getApexScene()->getTaskManager()->getNamedTask(AST_LOD_COMPUTE_BENEFIT);
}


PxF32   FluidParticleInjector::getLeastBenefitValue() const
{
	return mIosActor->getLeastBenefitValue();
}

void FluidParticleInjector::reset()
{
	mInjectedParticles.clear();
	mInjectedBenefit = 0.0f;
	mIsBackLogged = false;
}


/* Emitter calls this virtual injector API to insert new particles.  This should happen in
 * a task that runs before LOD.
 */
void FluidParticleInjector::createObjects(physx::PxU32 count, const IosNewObject* createList)
{
	PX_PROFILER_PERF_SCOPE("FluidIosCreateObjects");

	if (!mIosActor->mFluid)
	{
		return;
	}

	if (mRandomActorClassIDs.size() == 0)
	{
		return;
	}

	physx::PxVec3 eyePos = mIosActor->mParticleScene->getApexScene()->getEyePosition();

	// Append new objects to our list.  We do copies because we must perform buffering for the
	// emitters.  We have to hold these new objects until there is room in the NxFluid.
	for (physx::PxU32 i = 0 ; i < count ; i++)
	{
		IosNewObject& obj = mInjectedParticles.insert();
		obj = *createList++;

		obj.lodBenefit = mIosActor->calcParticleBenefit(*this, eyePos, obj.initialPosition, obj.initialVelocity, 1.0f);
		obj.iofxActorID.set(mVolumeID, mRandomActorClassIDs[ mLastRandomID++ ]);
		mLastRandomID = mLastRandomID == mRandomActorClassIDs.size() ? 0 : mLastRandomID;
		
		mInjectedBenefit += obj.lodBenefit;
	}
}

/* Emitter calls this function to adjust their particle weights with respect to other emitters */
void FluidParticleInjector::setLODWeights(physx::PxF32 maxDistance, physx::PxF32 distanceWeight, physx::PxF32 speedWeight, physx::PxF32 lifeWeight, physx::PxF32 separationWeight, physx::PxF32 bias)
{
	PX_UNUSED(separationWeight);

	//normalize weights
	PxF32 totalWeight = distanceWeight + speedWeight + lifeWeight;
	if (totalWeight > PX_EPS_F32)
	{
		distanceWeight /= totalWeight;
		speedWeight /= totalWeight;
		lifeWeight /= totalWeight;
	}

	mLODMaxDistance = maxDistance;
	mLODDistanceWeight = distanceWeight;
	mLODSpeedWeight = speedWeight;
	mLODLifeWeight = lifeWeight;
	mLODBias = bias;
}

void FluidParticleInjector::setPreferredRenderVolume(NxApexRenderVolume* volume)
{
	mVolume = volume;
	mVolumeID = mVolume ? mIosActor->mIofxMgr->getVolumeID(mVolume) : NiIofxActorID::NO_VOLUME;
}


void FluidParticleInjector::setResourceBudget(physx::PxU32 value)
{
	PX_ASSERT(value <= mSimulatedCount + mMaxInsertionCount);
	mResourceBudget = value;

	if (mMaxInsertionCount > mResourceBudget)
	{
		mMaxInsertionCount = mResourceBudget;
	}

	PxU32 forceDeleteCount = 0;
	if (mSimulatedCount + mMaxInsertionCount > mResourceBudget)
	{
		forceDeleteCount = (mSimulatedCount + mMaxInsertionCount) - mResourceBudget;
	}
	PX_ASSERT(forceDeleteCount <= mSimulatedCount);

	mForceDeleteList.reset(forceDeleteCount);
}

physx::PxF32 FluidParticleInjector::getBenefit()
{
	PxU32 injectedCount = getInjectedParticlesCount();
	PxF32 injectedBenefit = mInjectedBenefit;

	PxU32 simulatedCount = mSimulatedCount;
	PxF32 simulatedBenefit = mSimulatedBenefit;

	PxU32 totalCount = injectedCount + simulatedCount;
	PxF32 totalBenefit = injectedBenefit + simulatedBenefit;

	mLODNodeBenefit = (totalCount > 0) ? (totalBenefit / totalCount) : 0.0f;
	return mLODNodeBenefit;
}

physx::PxF32 FluidParticleInjector::setResource(physx::PxF32 suggested, physx::PxF32 maxRemaining, physx::PxF32 relativeBenefit)
{
	PX_UNUSED(maxRemaining);
	PX_UNUSED(relativeBenefit);

	mLODNodeResource = suggested;
	return mLODNodeResource;
}

}
}
} // namespace physx::apex

#endif // NX_SDK_VERSION_MAJOR == 2