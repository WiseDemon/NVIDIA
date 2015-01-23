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
#if NX_SDK_VERSION_MAJOR == 3

#include "NxApex.h"
#include "NiApexScene.h"
#include "NiApexSDK.h"

#include "NxParticleIosActor.h"
#include "ParticleIosActorCPU.h"
#include "ParticleIosAsset.h"
#include "NxIofxAsset.h"
#include "NxIofxActor.h"
#include "ModuleParticleIos.h"
#include "ParticleIosScene.h"
#include "NiApexRenderDebug.h"
#include "NiApexAuthorableObject.h"
#include "NiFieldSamplerQuery.h"
#include "foundation/PxMath.h"
#include "ApexMirroredArray.h"

#include "PxParticleSystem.h"
#include "PxParticleCreationData.h"
#include "PxParticleReadData.h"
#include "PxParticleDeviceExclusive.h"

namespace physx
{
namespace apex
{
namespace pxparticleios
{

#pragma warning(disable: 4355) // 'this' : used in base member initializer list

ParticleIosActorCPU::ParticleIosActorCPU(
    NxResourceList& list,
    ParticleIosAsset& asset,
    ParticleIosScene& scene,
	NxIofxAsset& iofxAsset)
	: ParticleIosActor(list, asset, scene, iofxAsset, false)
	, mSimulateTask(*this)
{
	initStorageGroups(mSimulationStorage);

	mField.reserve(mMaxParticleCount);
	mLifeTime.setSize(mMaxParticleCount);
	mLifeSpan.setSize(mMaxTotalParticleCount);
	mInjector.setSize(mMaxTotalParticleCount);
	mBenefit.setSize(mMaxTotalParticleCount);

	mNewIndices.resize(mMaxParticleCount);
	mAddedParticleList.reserve(mMaxParticleCount);
	mRemovedParticleList.reserve(mMaxParticleCount);
	mInputIdToParticleIndex.setSize(mMaxParticleCount, ApexMirroredPlace::CPU);

	mIndexPool = PxParticleExt::createIndexPool(mMaxParticleCount);

	mUpdateIndexBuffer.reserve(mMaxParticleCount);
	mUpdateVelocityBuffer.reserve(mMaxParticleCount);
}
ParticleIosActorCPU::~ParticleIosActorCPU()
{
	if (mIndexPool)
	{
		mIndexPool->release();
		mIndexPool = NULL;
	}	

}

physx::PxTaskID ParticleIosActorCPU::submitTasks(physx::PxTaskManager* tm)
{
	ParticleIosActor::submitTasks(tm);
	mInjectorsCounters.setSize(mInjectorList.getSize(), ApexMirroredPlace::CPU); 

	if (mAsset->getParticleDesc()->Enable == false)
	{
		return mInjectTask.getTaskID();
	}

	const physx::PxTaskID taskID = tm->submitUnnamedTask(mSimulateTask);
	return taskID;
}

void ParticleIosActorCPU::setTaskDependencies(physx::PxTaskID taskStartAfterID, physx::PxTaskID taskFinishBeforeID)
{
	physx::PxTask* iosTask = NULL;
	if (mAsset->getParticleDesc()->Enable)
	{
		iosTask = &mSimulateTask;
	}
	ParticleIosActor::setTaskDependencies(taskStartAfterID, taskFinishBeforeID, iosTask, false);
}

namespace
{
class FieldAccessor
{
	const physx::PxVec4* mField;
public:
	explicit FieldAccessor(const physx::PxVec4* field)
	{
		mField = field;
	}

	PX_INLINE void operator()(unsigned int srcIdx, physx::PxVec3& velocityDelta)
	{
		if (mField != NULL)
		{
			velocityDelta += mField[srcIdx].getXYZ();
		}
	}
};
}

void ParticleIosActorCPU::simulateParticles()
{
	PxF32 deltaTime = mParticleIosScene->getApexScene().getPhysXSimulateTime();
	const PxVec3& eyePos = mParticleIosScene->getApexScene().getEyePosition();

	SCOPED_PHYSX_LOCK_WRITE(mParticleIosScene->getApexScene());

	mTotalElapsedTime += deltaTime;

	PxU32 totalCount = mParticleCount + mInjectedCount;
	PxU32 activeCount = mLastActiveCount + mInjectedCount;
	PxU32 targetCount = mParticleBudget;

	physx::PxU32 maxStateID = 0; //we could drop state in case targetCount = 0

	for(PxU32 i = 0; i < mInjectorList.getSize(); ++i)
	{
		mInjectorsCounters[i] = 0; 
	}

	if (targetCount > 0)
	{
		maxStateID = mParticleCount;
		for (physx::PxU32 i = 0; i < maxStateID; ++i)
		{
			mNewIndices[i] = NiIosBufferDesc::NOT_A_PARTICLE;
		}

		PxU32 boundCount = 0;
		if (activeCount > targetCount)
		{
			boundCount = activeCount - targetCount;
		}

		PxF32 benefitMin = PxMin(mLastBenefitMin, mInjectedBenefitMin);
		PxF32 benefitMax = PxMax(mLastBenefitMax, mInjectedBenefitMax);
		PX_ASSERT(benefitMin <= benefitMax);
		benefitMax *= 1.00001f;

		/*
			boundBin - the highest benefit bin that should be culled
			boundCount - before computeHistogram it's the total culled particles.
					   - after computeHistogram it's the count of culled particles in boundBin
			boundIndex - count of culled particles in boundBin (0..boundCount-1)
		 */
		PxI32 boundBin = (PxI32)computeHistogram(totalCount, benefitMin, benefitMax, boundCount);
		physx::PxF32	factor = HISTOGRAM_BIN_COUNT / (benefitMax - benefitMin);
		for (PxU32 i = 0, boundIndex = 0; i < totalCount; ++i)
		{
			PxF32 benefit = mBenefit[i];
			if (benefit > -FLT_MAX)
			{
				PX_ASSERT(benefit >= benefitMin && benefit < benefitMax);

				PxI32 bin = PxI32((benefit - benefitMin) * factor);
				if (bin < boundBin)
				{
					mBenefit[i] = -FLT_MAX;
					continue;
				}
				if (bin == boundBin && boundIndex < boundCount)
				{
					mBenefit[i] = -FLT_MAX;
					++boundIndex;
				}
			}
		}
	}

	if (mParticleCount > 0)
	{
		mRemovedParticleList.clear();
		for (physx::PxU32 i = 0 ; i < mParticleCount; ++i)
		{
			if (!(mBenefit[i] > -FLT_MAX))
			{
				mRemovedParticleList.pushBack(mInputIdToParticleIndex[i]);
				mInputIdToParticleIndex[i] = INVALID_PARTICLE_INDEX;
			}
		}
		if (mRemovedParticleList.size())
		{
			PxStrideIterator<const PxU32> indexData( &mRemovedParticleList[0] );
			((PxParticleBase*)mParticleActor)->releaseParticles(mRemovedParticleList.size(), indexData);
			mIndexPool->freeIndices(mRemovedParticleList.size(), indexData);
			mRemovedParticleList.clear();
		}
	}

	mLastActiveCount = 0;
	mLastBenefitSum  = 0.0f;
	mLastBenefitMin  = +FLT_MAX;
	mLastBenefitMax  = -FLT_MAX;

	if (targetCount > 0)
	{
		const Px3InjectorParams* injectorParamsList = DYNAMIC_CAST(ParticleIosSceneCPU*)(mParticleIosScene)->mInjectorParamsArray.begin();

		FieldAccessor fieldAccessor(mFieldSamplerQuery ? mField.getPtr() : 0);

		mAddedParticleList.clear();
		mUpdateIndexBuffer.clear();
		mUpdateVelocityBuffer.clear();
		PxParticleReadData* readData = ((PxParticleBase*)mParticleActor)->lockParticleReadData();

		bool isDensityValid = false;
		if (!mIsParticleSystem)
		{
			PxParticleFluidReadData* fluidReadData = static_cast<PxParticleFluidReadData*>(readData);
			isDensityValid = (fluidReadData->densityBuffer.ptr() != 0);
		}

		for (PxU32 dstIdx = 0, srcHole = targetCount; dstIdx < targetCount; ++dstIdx)
		{
			PxU32 srcIdx = dstIdx;
			//do we have a hole in dstIdx region?
			if (!(mBenefit[dstIdx] > -FLT_MAX))
			{
				//skip holes in srcIdx region
				while (!(mBenefit[srcHole] > -FLT_MAX))
				{
					++srcHole;
				}
				PX_ASSERT(srcHole < totalCount);
				srcIdx = srcHole++;
			}
			//do we have a new particle?
			bool isNewParticle = (srcIdx >= mParticleCount);

			PxU32  pxIdx;
			PxVec3 position;
			PxVec3 velocity;
			PxVec3 collisionNormal;
			PxU32  particleFlags;
			PxF32  density;

			if (isNewParticle)
			{
				PxStrideIterator<PxU32> indexBuffer(&pxIdx);
				if (mIndexPool->allocateIndices(1, indexBuffer) != 1)
				{
					PX_ALWAYS_ASSERT();
					continue;
				}
				mInputIdToParticleIndex[dstIdx]	= pxIdx;
			}
			else
			{
				pxIdx = mInputIdToParticleIndex[srcIdx];
				PX_ASSERT((readData->flagsBuffer[pxIdx] & PxParticleFlag::eVALID));
				if (dstIdx != srcIdx)
				{
					PX_ASSERT(dstIdx < mParticleCount || !(readData->flagsBuffer[mInputIdToParticleIndex[dstIdx]] & PxParticleFlag::eVALID));
					mInputIdToParticleIndex[dstIdx]	= pxIdx;
				}

				position = readData->positionBuffer[pxIdx],
				velocity = readData->velocityBuffer[pxIdx],
				collisionNormal = readData->collisionNormalBuffer[pxIdx],
				particleFlags = readData->flagsBuffer[pxIdx],
				density = isDensityValid ? static_cast<PxParticleFluidReadData*>(readData)->densityBuffer[pxIdx] : 0.0f;
			}

			unsigned int injIndex;
			float benefit = simulateParticle(
			                    NULL, injectorParamsList,
			                    deltaTime, eyePos,
			                    isNewParticle, srcIdx, dstIdx,
			                    mBufDesc.pmaPositionMass->getPtr(), mBufDesc.pmaVelocityLife->getPtr(), 
								mBufDesc.pmaCollisionNormalFlags->getPtr(), mBufDesc.pmaUserData->getPtr(), mBufDesc.pmaActorIdentifiers->getPtr(),
								mLifeSpan.getPtr(), mLifeTime.getPtr(), mBufDesc.pmaDensity ? mBufDesc.pmaDensity->getPtr() : NULL, mInjector.getPtr(),
			                    fieldAccessor, injIndex,
								mGridDensityParams,
								position,
								velocity,
								collisionNormal,
								particleFlags,
								density
			                );

			if (injIndex < mInjectorsCounters.getSize())
			{
				++mInjectorsCounters[injIndex]; 
			}

			if (isNewParticle)
			{
				NewParticleData data;
				data.destIndex	= pxIdx;
				data.position	= position;
				data.velocity	= velocity;
				mAddedParticleList.pushBack(data);

				mBufDesc.pmaInStateToInput->get(maxStateID) = dstIdx | NiIosBufferDesc::NEW_PARTICLE_FLAG;
				++maxStateID;
			}
			else
			{
				mUpdateIndexBuffer.pushBack(pxIdx);
				mUpdateVelocityBuffer.pushBack(velocity);

				mNewIndices[srcIdx] = dstIdx;
			}

			mBenefit[dstIdx] = benefit;
			if (benefit > -FLT_MAX)
			{
				mLastBenefitSum += benefit;
				mLastBenefitMin = PxMin(mLastBenefitMin, benefit);
				mLastBenefitMax = PxMax(mLastBenefitMax, benefit);
				++mLastActiveCount;
			}
		}

		if (readData)
		{
			readData->unlock();
		}

		if (mUpdateIndexBuffer.size())
		{
			((PxParticleBase*)mParticleActor)->setVelocities(mUpdateIndexBuffer.size(), PxStrideIterator<const PxU32>(&mUpdateIndexBuffer[0]), PxStrideIterator<const PxVec3>(&mUpdateVelocityBuffer[0]));
		}

		if (mAddedParticleList.size())
		{
			PxParticleCreationData createData;
			createData.numParticles = mAddedParticleList.size();
			createData.positionBuffer = PxStrideIterator<const PxVec3>(&mAddedParticleList[0].position, sizeof(NewParticleData));
			createData.velocityBuffer = PxStrideIterator<const PxVec3>(&mAddedParticleList[0].velocity, sizeof(NewParticleData));
			createData.indexBuffer = PxStrideIterator<const PxU32>(&mAddedParticleList[0].destIndex, sizeof(NewParticleData));
			bool ok = ((PxParticleBase*)mParticleActor)->createParticles(createData);
			PX_ASSERT(ok);
			PX_UNUSED(ok);
		}

		//update stateToInput
		for (PxU32 i = 0; i < mParticleCount; ++i)
		{
			PxU32 srcIdx = mBufDesc.pmaOutStateToInput->get(i);
			PX_ASSERT(srcIdx < mParticleCount);
			mBufDesc.pmaInStateToInput->get(i) = mNewIndices[srcIdx];
		}
	}
	mParticleCount = targetCount;

	/* Oh! Manager of the IOFX! do your thing */
	mIofxMgr->updateEffectsData(deltaTime, mParticleCount, mParticleCount, maxStateID);
}

physx::PxU32 ParticleIosActorCPU::computeHistogram(physx::PxU32 dataCount, physx::PxF32 dataMin, physx::PxF32 dataMax, physx::PxU32& bound)
{
	const PxF32* dataArray = mBenefit.getPtr();

	PxU32 histogram[HISTOGRAM_BIN_COUNT];

	//clear Histogram
	for (PxU32 i = 0; i < HISTOGRAM_BIN_COUNT; ++i)
	{
		histogram[i] = 0;
	}

	physx::PxF32	factor = HISTOGRAM_BIN_COUNT / (dataMax - dataMin);
	//accum Histogram
	for (PxU32 i = 0; i < dataCount; ++i)
	{
		PxF32 data = dataArray[i];
		if (data >= dataMin && data < dataMax)
		{
			PxI32 bin = PxI32((data - dataMin) * factor);
			++histogram[bin];
		}
	}
	//compute CDF from Histogram
	PxU32 countSum = 0;
	for (PxU32 i = 0; i < HISTOGRAM_BIN_COUNT; ++i)
	{
		PxU32 count = histogram[i];
		countSum += count;
		histogram[i] = countSum;
	}

	PX_ASSERT(countSum == mLastActiveCount + mInjectedCount);

	//binary search in CDF
	PxU32 beg = 0;
	PxU32 end = HISTOGRAM_BIN_COUNT;
	while (beg < end)
	{
		PxU32 mid = beg + ((end - beg) >> 1);
		if (bound > histogram[mid])
		{
			beg = mid + 1;
		}
		else
		{
			end = mid;
		}
	}

	PX_ASSERT(histogram[beg] >= bound);
	if (beg > 0)
	{
		bound -= histogram[beg - 1];
	}

	return beg;
}

}
}
} // end namespace physx::apex

#endif // NX_SDK_VERSION_MAJOR == 3
