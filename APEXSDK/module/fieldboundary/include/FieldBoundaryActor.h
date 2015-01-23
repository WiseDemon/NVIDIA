/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FIELDBOUNDARY_ACTOR_H__
#define __FIELDBOUNDARY_ACTOR_H__

#include "NxApex.h"

#include "NxFieldBoundaryAsset.h"
#include "NxFieldBoundaryActor.h"
#include "FieldBoundaryAsset.h"
#include "ApexActor.h"
#include "ApexInterface.h"
#include "ApexRWLockable.h"
#include "NiFieldBoundary.h"

class NxForceFieldShapeGroup;

namespace physx
{
namespace apex
{
namespace fieldboundary
{

class FieldBoundaryAsset;
class FieldBoundaryScene;

class FieldBoundaryActor : public NxFieldBoundaryActor, public ApexActor, public ApexActorSource, public NxApexResource, public ApexResource, public NiFieldBoundary, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	/* NxFieldBoundaryActor methods */
	FieldBoundaryActor(const NxFieldBoundaryActorDesc&, FieldBoundaryAsset&, NxResourceList&, FieldBoundaryScene&);
	~FieldBoundaryActor();
	NxFieldBoundaryAsset* 	getFieldBoundaryAsset() const;
	physx::PxMat34Legacy				getGlobalPose() const
	{
		return mPose;
	}
	void				setGlobalPose(const physx::PxMat34Legacy& pose);
	physx::PxVec3				getScale() const
	{
		return mScale;
	}
	void				setScale(const physx::PxVec3& scale);
	void                updatePoseAndBounds();  // Called by FieldBoundaryScene::fetchResults()

	/* NxApexActorSource; templates for generating NxActors and NxShapes */
	void				setActorTemplate(const NxActorDescBase*);
	void				setShapeTemplate(const NxShapeDesc*);
	void				setBodyTemplate(const NxBodyDesc*);
	bool				getActorTemplate(NxActorDescBase& dest) const;
	bool				getShapeTemplate(NxShapeDesc& dest) const;
	bool				getBodyTemplate(NxBodyDesc& dest) const;

	/* NxApexResource, ApexResource */
	void				release();
	physx::PxU32				getListIndex() const
	{
		return m_listIndex;
	}
	void				setListIndex(class NxResourceList& list, physx::PxU32 index)
	{
		m_list = &list;
		m_listIndex = index;
	}

	/* NxApexActor, ApexActor */
	void                destroy();
	NxApexAsset*		getOwner() const;
	void				setPhysXScene(NxScene*);
	NxScene*			getPhysXScene() const;
	void				getPhysicalLodRange(physx::PxF32& min, physx::PxF32& max, bool& intOnly) const;
	physx::PxF32		getActivePhysicalLod() const;
	void				forcePhysicalLod(physx::PxF32 lod);
	/**
	\brief Selectively enables/disables debug visualization of a specific APEX actor.  Default value it true.
	*/
	virtual void setEnableDebugVisualization(bool state)
	{
		ApexActor::setEnableDebugVisualization(state);
	}


	void* 				getShapeGroupPtr() const
	{
		return (void*)mShapeGroup;
	}
	NxApexAsset*		getNxApexAsset(void)
	{
		return (NxApexAsset*) mAsset;
	}

	/* NiFieldBoundary */
	virtual bool updateFieldBoundary(physx::Array<NiFieldShapeDesc>& shapes);


protected:
	NxApexFieldBoundaryFlag			mType;
	physx::PxMat34Legacy			mPose;
	physx::PxVec3					mScale;
	FieldBoundaryAsset*				mAsset;
	FieldBoundaryScene*				mScene;
	NxForceFieldShapeGroup*			mShapeGroup;

	bool							mShapesChanged;

	friend class FieldBoundaryScene;
};

}
}
} // end namespace physx::apex

#endif
