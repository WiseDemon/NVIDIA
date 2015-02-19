/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_EXPLOSION_ASSET_H
#define NX_EXPLOSION_ASSET_H

#include "NxApex.h"
#include "NxExplosionPreview.h"

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

#define NX_EXPLOSION_AUTHORING_TYPE_NAME "ExplosionAsset"

class NxExplosionActor;

//----------------
//Explosion Asset
//----------------
class NxExplosionAsset : public NxApexAsset
{
protected:
	virtual   ~NxExplosionAsset() {}

public:
	virtual physx::PxF32		getDefaultScale() const = 0;
	virtual physx::PxU32		getFieldBoundariesCount() const = 0;
	virtual const char*			getFieldBoundariesName(physx::PxU32 boundIndex) = 0;

	virtual void				releaseExplosionActor(NxExplosionActor&) = 0;
};

//--------------------
//Explosion Authoring
//--------------------
class NxExplosionAssetAuthoring : public NxApexAssetAuthoring
{
protected:
	virtual ~NxExplosionAssetAuthoring() {}

public:
	//Add Field Boundaries to explosion
	virtual void							addFieldBoundaryName(const char*) = 0;
};


PX_POP_PACK

}
} // end namespace physx::apex

#endif // NX_EXPLOSION_ASSET_H
