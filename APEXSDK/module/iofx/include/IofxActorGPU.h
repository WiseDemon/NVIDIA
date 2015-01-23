/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __IOFX_ACTOR_GPU_H__
#define __IOFX_ACTOR_GPU_H__

#include "NxApex.h"
#include "NxIofxActor.h"
#include "ApexInterface.h"
#include "ApexActor.h"
#include "IofxActor.h"

namespace physx
{
namespace apex
{

class NiApexScene;

namespace iofx
{


class NxModifier;
class IofxAsset;
class IofxScene;
class IofxManager;

class IofxActorGPU : public IofxActor
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	IofxActorGPU(NxApexAsset* renderAsset, IofxScene* iscene, IofxManager& mgr)
		: IofxActor(renderAsset, iscene, mgr)
	{
	}
	~IofxActorGPU()
	{
	}
};


}
}
} // namespace apex

#endif // __IOFX_ACTOR_GPU_H__
