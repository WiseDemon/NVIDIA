/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_FIELD_BOUNDARY_SHARED_H
#define NX_FIELD_BOUNDARY_SHARED_H

#include "NxApex.h"

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

//-----------------
//Force Field ENUMs
//-----------------
/**
\brief enums for defining the force field type.
*/
enum NxApexForceFieldType
{
	/**
	\brief scales the force by the mass of the particle or body.
	*/
	NX_APEX_FF_TYPE_GRAVITATIONAL			= 0,
	/**
	\brief does not scale the value from the force field by the mass of the actors.
	*/
	NX_APEX_FF_TYPE_OTHER					= 1,
	/**
	\brief used to disable force field interaction with a specific feature
	*/
	NX_APEX_FF_TYPE_NO_INTERACTION			= 2
};

/**
\brief
*/
enum NxApexForceFieldMode
{
	/**
	\brief This actor is an explosion.
	*/
	NX_APEX_FFM_EXPLOSION		= 1,
	/**
	\brief This actor is an implosion.
	*/
	NX_APEX_FFM_IMPLOSION		= 2,
	/**
	\brief This actor is a shockwave.
	*/
	NX_APEX_FFM_SHOCKWAVE		= 3
};

PX_POP_PACK

} // namespace apex
} // namespace physx

#endif // NX_FIELD_BOUNDARY_SHARED_H
