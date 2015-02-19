/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_FLUID_IOS_ACTOR_H
#define NX_FLUID_IOS_ACTOR_H

#include "foundation/Px.h"
#include "NxApexDefs.h"

#if NX_SDK_VERSION_MAJOR == 2

#include "NxApexActor.h"
#include "Nxp.h"
namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

class NxFluidIosAsset;
/**
\brief Fluid IOS Actor. PhysX-based partice system.
 * This actor class does not have any set methods because actor creation will often times
 * result in immmediate NxFluid creation, and most NxFluid parameters are not run time
 * configurable.  Thus all parameters must be specified up front in the asset file.
 */
class NxFluidIosActor : public NxApexActor
{
public:
	/* Get configuration data */
	/// Get the collision group of this fluid actor
	virtual NxCollisionGroup			getCollisionGroup() const = 0;

	/* Get run time data */
	///Get the current particle count
	virtual physx::PxU32				getParticleCount() const = 0;

protected:
	virtual ~NxFluidIosActor()	{}
};

PX_POP_PACK

}
} // namespace physx::apex

#endif // NX_SDK_VERSION_MAJOR

#endif // NX_FLUID_IOS_ACTOR_H
