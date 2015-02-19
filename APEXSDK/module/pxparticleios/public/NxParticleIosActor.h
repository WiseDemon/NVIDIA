/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_PARTICLE_IOS_ACTOR_H
#define NX_PARTICLE_IOS_ACTOR_H

#include "NxApex.h"

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

/**
\brief ParticleIOS Actor. A simple actor that simulates a particle system.
 */
class NxParticleIosActor : public NxApexActor
{
public:
	// This actor is not publically visible

protected:
	virtual ~NxParticleIosActor()	{}
};

PX_POP_PACK

}
} // namespace physx::apex

#endif // NX_PARTICLE_IOS_ACTOR_H
