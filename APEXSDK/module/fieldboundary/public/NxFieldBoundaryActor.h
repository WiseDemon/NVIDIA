/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_FIELD_BOUNDARY_ACTOR_H
#define NX_FIELD_BOUNDARY_ACTOR_H

#include "NxApex.h"

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

class NxFieldBoundaryAsset;


/**
\brief The field boundary actor class.
*/
class NxFieldBoundaryActor : public NxApexActor, public NxApexActorSource
{
protected:
	virtual ~NxFieldBoundaryActor() {}

public:
	/**
	\brief Returns the asset the instance has been created from.
	*/
	virtual NxFieldBoundaryAsset* 	getFieldBoundaryAsset() const = 0;

	/**
	\brief Gets the FieldBoundary actor's 3D (possibly nonuniform) scale
	*/
	virtual physx::PxVec3			getScale() const = 0;

	/**
	\brief Sets the FieldBoundary actor's 3D (possibly nonuniform) scale
	*/
	virtual void					setScale(const physx::PxVec3& scale) = 0;

	/**
	\brief Returns a void pointer pointed to an internal NxForceFieldShapeGroup (needed by some force field features).
	*/
	virtual void*					getShapeGroupPtr() const = 0;
};

PX_POP_PACK

}
} // namespace physx::apex

#endif // NX_FIELD_BOUNDARY_ACTOR_H
