/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_MODULE_FORCE_FIELD_H
#define NX_MODULE_FORCE_FIELD_H

#include "NxApex.h"

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

class NxForceFieldAsset;
class NxForceFieldAssetAuthoring;

class NxModuleForceField : public NxModule
{
public:
	/**
	\brief The module ID value of the force field.
	*/
	virtual physx::PxU32 getModuleValue() const = 0;

protected:
	/**
	\brief Force field module default destructor.
	*/
	virtual ~NxModuleForceField() {}
};



PX_POP_PACK

} // namespace apex
} // namespace physx

#endif // NX_MODULE_FORCE_FIELD_H
