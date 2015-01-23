/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_FORCE_FIELD_ASSET_H
#define NX_FORCE_FIELD_ASSET_H

#include "NxApex.h"
#include "NxForceFieldPreview.h"

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

#define NX_FORCEFIELD_AUTHORING_TYPE_NAME "ForceFieldAsset"

class NxForceFieldActor;

//----------------
//ForceField Asset
//----------------
class NxForceFieldAsset : public NxApexAsset
{
protected:
	/**
	\brief force field asset default destructor.
	*/
	virtual   ~NxForceFieldAsset() {}

public:
	/**
	\brief returns the default scale of the asset.
	*/
	virtual physx::PxF32		getDefaultScale() const = 0;

	/**
	\brief release an actor created from this asset.
	*/
	virtual void				releaseForceFieldActor(NxForceFieldActor&) = 0;
};

//--------------------
//ForceField Authoring
//--------------------
class NxForceFieldAssetAuthoring : public NxApexAssetAuthoring
{
protected:
	/**
	\brief force field asset authoring default destructor.
	*/
	virtual ~NxForceFieldAssetAuthoring() {}

public:
};


PX_POP_PACK

} // namespace apex
} // namespace physx

#endif // NX_FORCE_FIELD_ASSET_H
