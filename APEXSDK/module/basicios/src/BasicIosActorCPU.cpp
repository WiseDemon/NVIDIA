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
#include "NiApexScene.h"
#include "NiApexSDK.h"

#include "NxBasicIosActor.h"
#include "BasicIosActorCPU.h"
#include "BasicIosAsset.h"
#include "NxIofxAsset.h"
#include "NxIofxActor.h"
#include "ModuleBasicIos.h"
#include "BasicIosSceneCPU.h"
#include "NiApexRenderDebug.h"
#include "NiApexAuthorableObject.h"
#include "NiFieldSamplerQuery.h"
#include "foundation/PxMath.h"
#include "ApexMirroredArray.h"

namespace physx
{
namespace apex
{
namespace basicios
{

#pragma warning(disable: 4355) // 'this' : used in base member initializer list

BasicIosActorCPU::BasicIosActorCPU(
    NxResourceList& list,
    BasicIosAsset& asset,
    BasicIosScene& scene,
    physx::apex::NxIofxAsset& iofxAsset)
	: BASIC_IOS_ACTOR(list, asset, scene, iofxAsset, false)
	, mSimulateTask(*this)
{
	initStorageGroups(mSimulationStorage);

	mLifeTime.setSize(mMaxParticleCount);
	mLifeSpan.setSize(mMaxTotalParticleCount);
	mInjector.setSize(mMaxTotalParticleCount);
	mBenefit.setSize(mMaxTotalParticleCount);

	if (mAsset->mParams->collisionWithConvex)
	{
		mConvexPlanes.reserve(MAX_CONVEX_PLANES_COUNT);
		mConvexVerts.reserve(MAX_CONVEX_VERTS_COUNT);
		mConvexPolygonsData.reserve(MAX_CONVEX_POLYGONS_DATA_SIZE);
	}
	if (mAsset->mParams->collisionWithTriangleMesh)
	{
		mTrimeshVerts.reserve(MAX_TRIMESH_VERTS_COUNT);
		mTrimeshIndices.reserve(MAX_TRIMESH_INDICES_COUNT);
	}

	mNewIndices.resize(mMaxParticleCount);
}
BasicIosActorCPU::~BasicIosActorCPU()
{
}

void BasicIosActorCPU::submitTasks()
{
	BasicIosActor::submitTasks();

	mInjectorsCounters.setSize(mInjectorList.getSize(), ApexMirroredPlace::CPU); 
	physx::PxTaskManager* tm = mBasicIosScene->getApexScene().getTaskManager();
	tm->submitUnnamedTask(mSimulateTask);
}

void BasicIosActorCPU::setTaskDependencies()
{
	BasicIosActor::setTaskDependencies(&mSimulateTask, false);
}

void BasicIosActorCPU::fetchResults()
{
	BASIC_IOS_ACTOR::fetchResults();
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

void BasicIosActorCPU::simulateParticles()
{
	physx::PxF32 deltaTime = mBasicIosScene->getApexScene().getPhysXSimulateTime();
	const physx::PxVec3& eyePos = mBasicIosScene->getApexScene().getEyePosition();

	mTotalElapsedTime += deltaTime;

	PxVec3 gravity = -mUp;

	physx::PxU32 totalCount = mParticleCount + mInjectedCount;
	physx::PxU32 activeCount = mLastActiveCount + mInjectedCount;
	physx::PxU32 targetCount = mParticleBudget;

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

		physx::PxU32 boundCount = 0;
		if (activeCount > targetCount)
		{
			boundCount = activeCount - targetCount;
		}

		PxF32 benefitMin = PxMin(mLastBenefitMin, mInjectedBenefitMin);
		PxF32 benefitMax = PxMax(mLastBenefitMax, mInjectedBenefitMax);
		PX_ASSERT(benefitMin <= benefitMax);
		benefitMax *= 1.00001f;

		physx::PxU32 boundBin = computeHistogram(totalCount, benefitMin, benefitMax, boundCount);
		for (physx::PxU32 i = 0, boundIndex = 0; i < totalCount; ++i)
		{
			physx::PxF32 benefit = mBenefit[i];
			if (benefit > -FLT_MAX)
			{
				PX_ASSERT(benefit >= benefitMin && benefit < benefitMax);

				physx::PxU32 bin = physx::PxU32((benefit - benefitMin) * HISTOGRAM_BIN_COUNT / (benefitMax - benefitMin));
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

		checkBenefit(totalCount);
		checkHoles(totalCount);
	}
	mLastActiveCount = 0;
	mLastBenefitSum  = 0.0f;
	mLastBenefitMin  = +FLT_MAX;
	mLastBenefitMax  = -FLT_MAX;

	if (targetCount > 0)
	{
		const InjectorParams* injectorParamsList = DYNAMIC_CAST(BasicIosSceneCPU*)(mBasicIosScene)->mInjectorParamsArray.begin();

		FieldAccessor fieldAccessor(mFieldSamplerQuery ? mField.getPtr() : 0);

		SimulationParams simParams;
		mSimulationParamsHandle.fetch(mSimulationStorage, simParams);

		for (physx::PxU32 dest = 0, srcHole = targetCount; dest < targetCount; ++dest)
		{
			physx::PxU32 src = dest;
			//do we have a hole in dest region?
			if (!(mBenefit[dest] > -FLT_MAX))
			{
				//skip holes in src region
				while (!(mBenefit[srcHole] > -FLT_MAX))
				{
					++srcHole;
				}
				PX_ASSERT(srcHole < totalCount);
				src = srcHole++;
			}
			//do we have a new particle?
			bool isNewParticle = (src >= mParticleCount);

			unsigned int injIndex;
			float benefit = simulateParticle(
			                    &simParams, mSimulationStorage, injectorParamsList,
			                    deltaTime, gravity, eyePos,
			                    isNewParticle, src, dest,
			                    mBufDesc.pmaPositionMass->getPtr(), mBufDesc.pmaVelocityLife->getPtr(), mBufDesc.pmaActorIdentifiers->getPtr(),
								mLifeSpan.getPtr(), mLifeTime.getPtr(), mInjector.getPtr(), mBufDesc.pmaCollisionNormalFlags->getPtr(), mBufDesc.pmaUserData->getPtr(),
			                    fieldAccessor, injIndex
			                );

			if (injIndex < mInjectorsCounters.getSize())
			{
				++mInjectorsCounters[injIndex]; 
			}
			
			if (!isNewParticle)
			{
				mNewIndices[src] = dest;
			}
			else
			{
				mBufDesc.pmaInStateToInput->get(maxStateID) = dest | NiIosBufferDesc::NEW_PARTICLE_FLAG;
				++maxStateID;
			}

			mBenefit[dest] = benefit;
			if (benefit > -FLT_MAX)
			{
				mLastBenefitSum += benefit;
				mLastBenefitMin = PxMin(mLastBenefitMin, benefit);
				mLastBenefitMax = PxMax(mLastBenefitMax, benefit);
				++mLastActiveCount;
			}
		}

		//update stateToInput
		for (physx::PxU32 i = 0; i < mParticleCount; ++i)
		{
			physx::PxU32 src = mBufDesc.pmaOutStateToInput->get(i);
			PX_ASSERT( src < mParticleCount );
			mBufDesc.pmaInStateToInput->get(i) = mNewIndices[src];
		}
	}
	checkInState(totalCount);

	mParticleCount = targetCount;

	/* Oh! Manager of the IOFX! do your thing */
	mIofxMgr->updateEffectsData(deltaTime, mParticleCount, mParticleCount, maxStateID);
}

physx::PxU32 BasicIosActorCPU::computeHistogram(physx::PxU32 dataCount, physx::PxF32 dataMin, physx::PxF32 dataMax, physx::PxU32& bound)
{
	const physx::PxF32* dataArray = mBenefit.getPtr();

	physx::PxU32 histogram[HISTOGRAM_BIN_COUNT];

	//clear Histogram
	for (physx::PxU32 i = 0; i < HISTOGRAM_BIN_COUNT; ++i)
	{
		histogram[i] = 0;
	}
	//accum Histogram
	for (physx::PxU32 i = 0; i < dataCount; ++i)
	{
		physx::PxF32 data = dataArray[i];
		if (data >= dataMin && data < dataMax)
		{
			physx::PxU32 bin = physx::PxU32((data - dataMin) * HISTOGRAM_BIN_COUNT / (dataMax - dataMin));
			++histogram[bin];
		}
	}
	//compute CDF from Histogram
	physx::PxU32 countSum = 0;
	for (physx::PxU32 i = 0; i < HISTOGRAM_BIN_COUNT; ++i)
	{
		physx::PxU32 count = histogram[i];
		countSum += count;
		histogram[i] = countSum;
	}

	//binary search in CDF
	physx::PxU32 beg = 0;
	physx::PxU32 end = HISTOGRAM_BIN_COUNT;
	while (beg < end)
	{
		physx::PxU32 mid = beg + ((end - beg) >> 1);
		if (bound > histogram[mid])
		{
			beg = mid + 1;
		}
		else
		{
			end = mid;
		}
	}

	checkHistogram(bound, histogram[beg], histogram[HISTOGRAM_BIN_COUNT - 1]);

	if (beg > 0)
	{
		bound -= histogram[beg - 1];
	}

	return beg;
}

}
}
} // namespace physx::apex
