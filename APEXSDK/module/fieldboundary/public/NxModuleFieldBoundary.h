/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_MODULE_FIELD_BOUNDARY_H
#define NX_MODULE_FIELD_BOUNDARY_H

#include "NxApex.h"

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

class NxFieldBoundaryAsset;
class NxFieldBoundaryAssetAuthoring;

/**
\brief Force field module class.
*/
class NxModuleFieldBoundary : public NxModule
{
public:
protected:
	/**
	\brief default destructor for the APEX force field module.
	*/
	virtual ~NxModuleFieldBoundary() {}
};


PX_POP_PACK

} // namespace apex
} // namespace physx

#endif // NX_MODULE_FIELD_BOUNDARY_H
