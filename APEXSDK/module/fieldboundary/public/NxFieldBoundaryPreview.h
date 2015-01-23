/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_FIELD_BOUNDARY_PREVIEW_H
#define NX_FIELD_BOUNDARY_PREVIEW_H

#include "NxApex.h"

namespace physx
{
namespace apex
{
PX_PUSH_PACK_DEFAULT

namespace APEX_FIELD_BOUNDARY
{
/**
*/
static const physx::PxU32 FIELD_BOUNDARY_DRAW_ICON = 0x01;
/**
*/
static const physx::PxU32 FIELD_BOUNDARY_DRAW_ICON_BOLD = 0x02;
/**
*/
static const physx::PxU32 FIELD_BOUNDARY_DRAW_BOUNDARIES = 0x04;
} //namespace APEX_FIELD_BOUNDARY

class NxFieldBoundaryPreview : public NxApexAssetPreview
{
public:
	/**
	*/
	virtual void setDetailLevel(physx::PxU32) const = 0;
protected:
	NxFieldBoundaryPreview() {}
};

PX_POP_PACK
} //namespace apex
} //namespace physx

#endif //NX_FIELD_BOUNDARY_PREVIEW_H
