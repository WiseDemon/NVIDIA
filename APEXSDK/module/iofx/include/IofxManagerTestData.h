/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef IOFX_MANAGER_TEST_DATA_H
#define IOFX_MANAGER_TEST_DATA_H

#include "NxApex.h"
#include "PsArray.h"

namespace physx
{
namespace apex
{
namespace iofx
{

#ifdef APEX_TEST

struct IofxManagerTestData
{
	bool mIsCPUTest;
	bool mIsGPUTest;

	//Common data
	physx::shdfnd::Array<physx::PxU32>				mInStateToInput;
	physx::shdfnd::Array<physx::PxU32>				mOutStateToInput;
	physx::shdfnd::Array<physx::PxU32>				mCountPerActor;
	physx::shdfnd::Array<physx::PxU32>				mStartPerActor;
	physx::shdfnd::Array<physx::PxVec4>				mPositionMass;

	physx::PxU32									mNumParticles;
	physx::PxU32									mMaxInputID;
	physx::PxU32									mMaxStateID;
	physx::PxU32									mCountActorIDs;

	unsigned int									mNOT_A_PARTICLE;
	unsigned int									mNEW_PARTICLE_FLAG;
	unsigned int									mSTATE_ID_MASK;

	//GPU specific data
	physx::shdfnd::Array<physx::PxU32>				mSortedActorIDs;
	physx::shdfnd::Array<physx::PxU32>				mSortedStateIDs;
	physx::shdfnd::Array<physx::PxU32>				mActorStart;
	physx::shdfnd::Array<physx::PxU32>				mActorEnd;
	physx::shdfnd::Array<physx::PxU32>				mActorVisibleEnd;
	physx::shdfnd::Array<physx::PxVec4>				mMinBounds;
	physx::shdfnd::Array<physx::PxVec4>				mMaxBounds;

	IofxManagerTestData() : mIsCPUTest(false), mIsGPUTest(false) {}
};

#endif

}
}
} // namespace physx::apex

#endif // IOFX_MANAGER_TEST_DATA_H
