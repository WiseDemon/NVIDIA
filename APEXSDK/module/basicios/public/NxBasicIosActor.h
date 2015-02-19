/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_BASIC_IOS_ACTOR_H
#define NX_BASIC_IOS_ACTOR_H

#include "NxApex.h"

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

/**
\brief BasicIOS Actor. A simple actor that simulates a particle system.
 */
class NxBasicIosActor : public NxApexActor
{
public:
	/* Get constant data */
	/// Get the particle radius
	virtual physx::PxF32						getParticleRadius() const = 0;
	/// Get the particle rest density
	virtual physx::PxF32						getRestDensity() const = 0;

	/* Get run time data */
	/// Get the current number of particles
	virtual physx::PxU32						getParticleCount() const = 0;

protected:

	virtual ~NxBasicIosActor()	{}
};

PX_POP_PACK

}
} // namespace physx::apex

#endif // NX_BASIC_IOS_ACTOR_H
