/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_PARTICLE_IOS_ASSET_H
#define NX_PARTICLE_IOS_ASSET_H

#include "NxApex.h"
#include <limits.h>

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

#define NX_PARTICLE_IOS_AUTHORING_TYPE_NAME "ParticleIosAsset"

/**
 \brief APEX Particle System Asset
 */
class NxParticleIosAsset : public NxIosAsset
{
public:
	///Get the radius of a particle
	virtual physx::PxF32						getParticleRadius() const = 0;
	///Get the rest density of a particle
	//virtual physx::PxF32						getRestDensity() const = 0;
	///Get the maximum number of particles that are allowed to be newly created on each frame
	virtual physx::PxF32						getMaxInjectedParticleCount() const	= 0;
	///Get the maximum number of particles that this IOS can simulate
	virtual physx::PxU32						getMaxParticleCount() const = 0;
	///Get the mass of a particle
	virtual physx::PxF32						getParticleMass() const = 0;

protected:
	virtual ~NxParticleIosAsset()	{}
};

/**
 \brief APEX Particle System Asset Authoring class
 */
class NxParticleIosAssetAuthoring : public NxApexAssetAuthoring
{
public:
	///Set the radius of a particle
	virtual void setParticleRadius(physx::PxF32) = 0;
	///Set the rest density of a particle
	//virtual void setRestDensity( physx::PxF32 ) = 0;
	///Set the maximum number of particles that are allowed to be newly created on each frame
	virtual void setMaxInjectedParticleCount(physx::PxF32 count) = 0;
	///Set the maximum number of particles that this IOS can simulate
	virtual void setMaxParticleCount(physx::PxU32 count) = 0;
	///Set the mass of a particle
	virtual void setParticleMass(physx::PxF32) = 0;

	///Set the (NRP) name for the collision group.
	virtual void setCollisionGroupName(const char* collisionGroupName) = 0;
	///Set the (NRP) name for the collision group mask.
	virtual void setCollisionGroupMaskName(const char* collisionGroupMaskName) = 0;

protected:
	virtual ~NxParticleIosAssetAuthoring()	{}
};

PX_POP_PACK

}
} // namespace physx::apex

#endif // NX_PARTICLE_IOS_ASSET_H
