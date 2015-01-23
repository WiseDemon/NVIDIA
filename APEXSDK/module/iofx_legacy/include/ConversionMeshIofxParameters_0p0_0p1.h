/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __CONVERSIONMESHIOFXPARAMETERS_0P0_0P1H__
#define __CONVERSIONMESHIOFXPARAMETERS_0P0_0P1H__

#include "ParamConversionTemplate.h"
#include "MeshIofxParameters_0p0.h"
#include "MeshIofxParameters_0p1.h"

namespace physx
{
namespace apex
{
namespace legacy
{

typedef ParamConversionTemplate<MeshIofxParameters_0p0, MeshIofxParameters_0p1, 0, 1> ConversionMeshIofxParameters_0p0_0p1Parent;

class ConversionMeshIofxParameters_0p0_0p1: ConversionMeshIofxParameters_0p0_0p1Parent
{
public:
	static NxParameterized::Conversion* Create(NxParameterized::Traits* t)
	{
		void* buf = t->alloc(sizeof(ConversionMeshIofxParameters_0p0_0p1));
		return buf ? PX_PLACEMENT_NEW(buf, ConversionMeshIofxParameters_0p0_0p1)(t) : 0;
	}

protected:
	ConversionMeshIofxParameters_0p0_0p1(NxParameterized::Traits* t) : ConversionMeshIofxParameters_0p0_0p1Parent(t) {}

	bool convert()
	{
		//TODO:
		//	Write custom conversion code here using mNewData and mLegacyData members.
		//
		//	Note that
		//		- mNewData was initialized with default values
		//		- same-named/same-typed members were copied from mLegacyData to mNewData
		//		- included references were moved to mNewData
		//			(and updated to preferred versions according to getPreferredVersions)
		//
		//	For more info see the versioning wiki.

		return true;
	}
};

}
}
} //end of physx::apex:: namespace

#endif
