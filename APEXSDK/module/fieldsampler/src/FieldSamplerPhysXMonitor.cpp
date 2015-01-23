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
#include "MinPhysxSdkVersion.h"

#if NX_SDK_VERSION_MAJOR == 3

#include "PsSort.h"
#include "FieldSamplerPhysXMonitor.h"
#include "NxApexReadWriteLock.h"
#include "FieldSamplerScene.h"
#include "FieldSamplerManager.h"
#include "NiFieldSamplerQuery.h"

#include "extensions/PxShapeExt.h"

namespace physx
{
namespace apex
{
namespace fieldsampler
{

#pragma warning(disable: 4355) // 'this' : used in base member initializer list

FieldSamplerPhysXMonitor::FieldSamplerPhysXMonitor(FieldSamplerScene& scene)
	: mFieldSamplerScene(&scene)
	, mNumPS(0)
	, mNumRB(0)
	, mEnable(false)
	, mTaskRunAfterActorUpdate(*this)
{
	mFilterData.setToDefault();

	mScene = mFieldSamplerScene->getModulePhysXScene();
	mParams = static_cast<FieldSamplerPhysXMonitorParams*>(NiGetApexSDK()->getParameterizedTraits()->createNxParameterized(FieldSamplerPhysXMonitorParams::staticClassName()));
	mFieldSamplerManager = DYNAMIC_CAST(FieldSamplerManager*)(mFieldSamplerScene->getManager());

	mRBIndex.reserve(mParams->maxRBCount);
	mParticleSystems.resize(mParams->maxPSCount);

	mPSOutField.resize(mParams->maxParticleCount);
	mOutVelocities.resize(mParams->maxParticleCount);
	mOutIndices.resize(mParams->maxParticleCount);

	mRBOutField.resize(mParams->maxRBCount);

	mRBActors.resize(mParams->maxRBCount);
}


FieldSamplerPhysXMonitor::~FieldSamplerPhysXMonitor()
{
	for (physx::PxU32 i = 0; i < mPSFieldSamplerQuery.size(); i++)
	{
		if (mPSFieldSamplerQuery[i])
		{
			mPSFieldSamplerQuery[i]->release();
		}
	}
	for (physx::PxU32 i = 0; i < mRBFieldSamplerQuery.size(); i++)
	{
		if (mRBFieldSamplerQuery[i])
		{
			mRBFieldSamplerQuery[i]->release();
		}
	}
	if(mParams)
	{
		mParams->destroy();
	}
}


void FieldSamplerPhysXMonitor::setPhysXScene(PxScene* scene)
{
	mScene = scene;
}


void FieldSamplerPhysXMonitor::getParticles(physx::PxU32 taskId)
{
	SCOPED_PHYSX3_LOCK_READ(mFieldSamplerScene->getApexScene().getPhysXScene());
	physx::PxF32 deltaTime = mFieldSamplerScene->getApexScene().getPhysXSimulateTime();
	mPCount = 0;
	mNumPS = mScene->getActors(physx::PxActorTypeSelectionFlag::ePARTICLE_SYSTEM, &mParticleSystems[0], mParams->maxPSCount);
	for(PxU32 i = 0; i < mNumPS; i++)
	if (!mFieldSamplerManager->isUnhandledParticleSystem(mParticleSystems[i]) && mParticleSystems[i]->isParticleBase())
	{
		if (mPSFieldSamplerQuery.size() == i)
		{
			NiFieldSamplerQueryDesc queryDesc;
			queryDesc.maxCount = mParams->maxParticleCount;
			queryDesc.samplerFilterData = mFilterData;
			mPSFieldSamplerQuery.pushBack( mFieldSamplerManager->createFieldSamplerQuery(queryDesc) );
			mPSFieldSamplerTaskID.pushBack(0);
			mParticleReadData.pushBack(0);
			mPSMass.pushBack(0.f);
		}

		PxParticleSystem* particleSystem = DYNAMIC_CAST(PxParticleSystem*)((mParticleSystems[i]));
		mParticleReadData[i] = particleSystem->lockParticleReadData();
		physx::PxU32 numParticles;
		if (mParticleReadData[i])
		{
			numParticles = mParticleReadData[i]->validParticleRange;
			if(mPCount + numParticles >= mParams->maxParticleCount) break;
		
			NiFieldSamplerQueryData queryData;
			queryData.timeStep = deltaTime;
			queryData.count = numParticles;
			queryData.isDataOnDevice = false;
			
			//hack for PhysX particle stride calculation
			physx::PxStrideIterator<const physx::PxVec3> positionIt(mParticleReadData[i]->positionBuffer);
#ifdef WIN64
			queryData.positionStrideBytes = (PxU32)(-(PxI64)&*positionIt + (PxI64)&*(++positionIt));
#else
			queryData.positionStrideBytes = (PxU32)(-(PxI32)&*positionIt + (PxI32)&*(++positionIt));
#endif
			queryData.velocityStrideBytes = queryData.positionStrideBytes;
			queryData.massStrideBytes = 0;
			queryData.pmaInPosition = (PxF32*)&*(mParticleReadData[i]->positionBuffer);
			queryData.pmaInVelocity = (PxF32*)&*(mParticleReadData[i]->velocityBuffer);
			queryData.pmaInIndices = 0;
			mPSMass[i] = particleSystem->getParticleMass();
			queryData.pmaInMass = &mPSMass[i];
			queryData.pmaOutField = &mPSOutField[mPCount];
			mPSFieldSamplerTaskID[i] = mPSFieldSamplerQuery[i]->submitFieldSamplerQuery(queryData, taskId);

			mPCount += numParticles;
		}
	}
}


void FieldSamplerPhysXMonitor::updateParticles()
{
	PxU32 pCount = 0;
	SCOPED_PHYSX3_LOCK_WRITE(mFieldSamplerScene->getApexScene().getPhysXScene());
	for(PxU32 i = 0; i < mNumPS; i++)
	if (!mFieldSamplerManager->isUnhandledParticleSystem(mParticleSystems[i]))
	{
		PxParticleSystem* particleSystem = DYNAMIC_CAST(PxParticleSystem*)((mParticleSystems[i]));
		PxU32 numParticles = PxMin(mParticleReadData[i]->validParticleRange, mParams->maxParticleCount);

		PxU32 numUpdates = 0;

		if (numParticles > 0)
		{
			for (PxU32 w = 0; w <= (mParticleReadData[i]->validParticleRange-1) >> 5; w++)
			{
				for (PxU32 b = mParticleReadData[i]->validParticleBitmap[w]; b; b &= b-1) 
				{
					PxU32 index = (w << 5 | shdfnd::lowestSetBit(b));

					PxVec3 diffVel = mPSOutField[pCount + index].getXYZ();
					if (!diffVel.isZero())
					{
						const PxVec3& sourceVelocity = mParticleReadData[i]->velocityBuffer[index];
						mOutVelocities[numUpdates] = sourceVelocity + diffVel;
						mOutIndices[numUpdates] = index;
						numUpdates++;
					}
				}
			}
		}
		pCount += numParticles;
		// return ownership of the buffers back to the SDK
		mParticleReadData[i]->unlock();
	
		if(pCount <= mParams->maxParticleCount && numUpdates > 0)
		{
			physx::PxStrideIterator<PxU32> indices(&mOutIndices[0]);
			physx::PxStrideIterator<PxVec3> outVelocities(&mOutVelocities[0]);
			particleSystem->setVelocities(numUpdates, indices, outVelocities);
		}		
	}
}


void FieldSamplerPhysXMonitor::getRigidBodies(physx::PxU32 taskId)
{
	SCOPED_PHYSX_LOCK_READ(mFieldSamplerScene->getApexScene());
	physx::PxF32 deltaTime = mFieldSamplerScene->getApexScene().getPhysXSimulateTime();

	NiFieldSamplerQueryData queryData;
	queryData.timeStep = deltaTime;
	queryData.pmaInIndices = 0;

	PxU32 rbCount = mScene->getActors(physx::PxActorTypeSelectionFlag::eRIGID_DYNAMIC, &mRBActors[0], mParams->maxRBCount);
	Array<PxShape*> shapes;
	mNumRB = 0;
	mRBIndex.clear();
	PxF32 weight = 1.f;

	for (PxU32 i = 0; i < rbCount; i++)
	{
		physx::PxRigidDynamic* rb = (physx::PxRigidDynamic*)mRBActors[i];
		if (rb->getRigidDynamicFlags() == PxRigidDynamicFlag::eKINEMATIC) 
		{
			continue;
		}
		const physx::PxVec3& cmPos = rb->getGlobalPose().p;
		const physx::PxVec3& velocity = rb->getLinearVelocity();
		const physx::PxVec3& rotation = rb->getAngularVelocity();
		physx::PxF32 mass = rb->getMass();

		const PxU32 numShapes = rb->getNbShapes();
		shapes.resize(numShapes);
		if (numShapes == 0) 
		{
			continue;
		}
		rb->getShapes(&shapes[0], numShapes);
		for (PxU32 j = 0; j < numShapes && mNumRB < mParams->maxRBCount; j++)
		{
			PxFilterData filterData = shapes[j]->getQueryFilterData();
			if (mFieldSamplerManager->getFieldSamplerGroupsFiltering(mFilterData, filterData, weight))
			{
				PxFilterData* current = mRBFilterData.find(filterData);
				if (current == mRBFilterData.end())
				{
					mRBFilterData.pushBack(filterData);
					current = &mRBFilterData.back();
					NiFieldSamplerQueryDesc queryDesc;
					queryDesc.maxCount = mParams->maxParticleCount;
					queryDesc.samplerFilterData = filterData;
					mRBFieldSamplerQuery.pushBack( mFieldSamplerManager->createFieldSamplerQuery(queryDesc) );
				}

				ShapeData* sd = PX_NEW(ShapeData)();
				sd->fdIndex = (PxU32)(current - &mRBFilterData[0]);
				sd->rbIndex = i;
				sd->mass = mass / numShapes;
				sd->pos = PxShapeExt::getWorldBounds(*shapes[j], *rb).getCenter();
				sd->vel = velocity + rotation.cross(sd->pos - cmPos);
				mRBIndex.pushBack(sd);
				++mNumRB;
			}
		}		
	}

	if (mNumRB == 0) 
	{
		return;
	}

	
	sort(&mRBIndex[0], mNumRB, ShapeData::sortPredicate);

	mRBInPosition.resize(mNumRB);
	mRBInVelocity.resize(mNumRB);

	PxU32 current(mRBIndex[0]->fdIndex);
	PxU32 currentCount = 0;
	PxU32 fdCount = 0;
	for (PxU32 i = 0; i <= mNumRB; i++)
	{
		if (i == mNumRB || current != mRBIndex[i]->fdIndex)
		{
			queryData.count = currentCount;
			queryData.isDataOnDevice = false;
			queryData.massStrideBytes = sizeof(PxVec4);
			queryData.positionStrideBytes = sizeof(PxVec4);
			queryData.velocityStrideBytes = queryData.positionStrideBytes;

			queryData.pmaInPosition = (PxF32*)(&mRBInPosition[fdCount]);
			queryData.pmaInVelocity = (PxF32*)(&mRBInVelocity[fdCount]);
			queryData.pmaInMass = &(mRBInPosition[fdCount].w);
			queryData.pmaOutField = &mRBOutField[fdCount];
	
			mRBFieldSamplerQuery[current]->submitFieldSamplerQuery(queryData, taskId);
			
			fdCount += currentCount;
			if (i != mNumRB) 
			{
				current = mRBIndex[i]->fdIndex;
				currentCount = 1;
			}
		}
		else
		{
			currentCount++;
		}
		if (i < mNumRB)
		{
			mRBInPosition[i] = PxVec4 (mRBIndex[i]->pos, mRBIndex[i]->mass);
			mRBInVelocity[i] = PxVec4 (mRBIndex[i]->vel, 0.f);
		}
	}
}

void FieldSamplerPhysXMonitor::updateRigidBodies()
{
	SCOPED_PHYSX_LOCK_WRITE(mFieldSamplerScene->getApexScene());

	for(PxU32 i = 0; i < mNumRB; i++)
	{
		physx::PxRigidDynamic* rb = (physx::PxRigidDynamic*)mRBActors[mRBIndex[i]->rbIndex];
		const physx::PxVec3 velocity = mRBOutField[i].getXYZ();
		const physx::PxVec3 rotation = mRBOutField[i].getXYZ().cross(mRBInPosition[i].getXYZ() - rb->getGlobalPose().p);
		if (!velocity.isZero() || !rotation.isZero())
		{
			rb->setLinearVelocity(rb->getLinearVelocity() + velocity);
			rb->setAngularVelocity(rb->getAngularVelocity() + rotation);
		}

		PX_DELETE(mRBIndex[i]);
	}
}


void FieldSamplerPhysXMonitor::update()
{	
	physx::PxTaskManager* tm = mFieldSamplerScene->getApexScene().getTaskManager();
	physx::PxU32 taskId = tm->getNamedTask(FSST_PHYSX_MONITOR_LOAD);
	if(mScene)
	{
		getParticles(taskId);

		getRigidBodies(taskId);

		//getCloth(task);
	}
	if(mNumPS > 0 || mNumRB > 0)
	{	
		tm->submitNamedTask(&mTaskRunAfterActorUpdate, FSST_PHYSX_MONITOR_UPDATE);
		mTaskRunAfterActorUpdate.startAfter(tm->getNamedTask(FSST_PHYSX_MONITOR_FETCH));
		mTaskRunAfterActorUpdate.finishBefore(tm->getNamedTask(AST_PHYSX_SIMULATE));
	}	
}

void FieldSamplerPhysXMonitor::updatePhysX()
{
	updateParticles();
	updateRigidBodies();
}


}
}
} // end namespace physx::apex

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED

