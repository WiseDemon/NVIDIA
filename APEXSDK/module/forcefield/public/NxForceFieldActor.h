/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_FORCE_FIELD_ACTOR_H
#define NX_FORCE_FIELD_ACTOR_H

#include "NxApex.h"


namespace physx
{
namespace apex
{


PX_PUSH_PACK_DEFAULT

class NxForceFieldAsset;
class ForceFieldAssetParams;

class NxForceFieldActor : public NxApexActor
{
protected:
	virtual ~NxForceFieldActor() {}

public:
	/**
	Return true if the force field actor is enabled.
	*/
	virtual bool					isEnable() = 0;

	/**
	Disable force field actor. Default status is enable. Can switch it freely.
	A disabled explosion actor still exists there, but has no effect to the scene.
	*/
	virtual bool					disable() = 0;

	/**
	Enable force field actor. Default status is enable. Can switch it freely.
	A disabled explosion actor still exists there, but has no effect to the scene.
	*/
	virtual bool					enable() = 0;

	/**
	Gets location and orientation of the force field.
	*/
	virtual physx::PxMat44			getPose() const = 0;

	/**
	Sets location and orientation of the force field.
	*/
	virtual void					setPose(const physx::PxMat44& pose) = 0;

	/**
	Gets the force field actor's scale.
	*/
	PX_DEPRECATED virtual physx::PxF32			getScale() const = 0;

	/**
	Sets the force field actor's scale. (0.0f, +inf)
	*/
	PX_DEPRECATED virtual void					setScale(physx::PxF32 scale) = 0;

	/**
	Gets the force field actor's scale.
	*/
	PX_DEPRECATED virtual physx::PxF32			getCurrentScale() const = 0;

	/**
	Sets the force field actor's scale. (0.0f, +inf)
	*/
	PX_DEPRECATED virtual void					setCurrentScale(physx::PxF32 scale) = 0;



	/**
	Retrieves the name string for the force field actor.
	*/
	virtual const char*				getName() const = 0;

	/**
	Set a name string for the force field actor that can be retrieved with getName().
	*/
	virtual void					setName(const char* name) = 0;

	/**
	Set strength for the force field actor.
	*/
	virtual void					setStrength(const physx::PxF32 strength) = 0;

	/**
	Set lifetime for the force field actor.
	*/
	virtual void					setLifetime(const physx::PxF32 lifetime) = 0;

	/**
	Set falloff type (linear, steep, scurve, custom, none) for the force field actor.
	Only works for radial force field types.
	*/
	PX_DEPRECATED virtual void		setFalloffType(const char* type) = 0;

	/**
	Set falloff multiplier for the force field actor.
	Only works for radial force field types.
	*/
	PX_DEPRECATED virtual void		setFalloffMultiplier(const physx::PxF32 multiplier) = 0;

	/**
	Returns the asset the actor has been created from.
	*/
	virtual NxForceFieldAsset* 	    getForceFieldAsset() const = 0;

};

PX_POP_PACK

} // namespace apex
} // namespace physx

#endif // NX_FORCE_FIELD_ACTOR_H
