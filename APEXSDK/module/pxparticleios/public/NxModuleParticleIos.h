/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_MODULE_PARTICLE_IOS_H
#define NX_MODULE_PARTICLE_IOS_H

#include "NxApex.h"
#include <limits.h>

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

class NxParticleIosAsset;
class NxParticleIosAssetAuthoring;
class NxApexCudaTestManager;
class NxApexScene;

/**
\brief ParticleIOS Module - Manages PhysX 3.0 PxParticleSystem and PxParticleFluid simulations
*/
class NxModuleParticleIos : public NxModule
{
protected:
	virtual											~NxModuleParticleIos() {}

public:
	/// Get ParticleIOS authoring type name
	virtual const char*								getParticleIosTypeName() = 0;
};


PX_POP_PACK

}
} // namespace physx::apex

#endif // NX_MODULE_PARTICLE_IOS_H
