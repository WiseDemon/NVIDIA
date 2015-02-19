/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef CONVERSIONBASICIOSASSETPARAM_0P3_0P4H_H
#define CONVERSIONBASICIOSASSETPARAM_0P3_0P4H_H

#include "ParamConversionTemplate.h"
#include "BasicIOSAssetParam_0p3.h"
#include "BasicIOSAssetParam_0p4.h"

namespace physx
{
namespace apex
{
namespace legacy
{

typedef ParamConversionTemplate<BasicIOSAssetParam_0p3, BasicIOSAssetParam_0p4, 3, 4> ConversionBasicIOSAssetParam_0p3_0p4Parent;

class ConversionBasicIOSAssetParam_0p3_0p4: ConversionBasicIOSAssetParam_0p3_0p4Parent
{
public:
	static NxParameterized::Conversion* Create(NxParameterized::Traits* t)
	{
		void* buf = t->alloc(sizeof(ConversionBasicIOSAssetParam_0p3_0p4));
		return buf ? PX_PLACEMENT_NEW(buf, ConversionBasicIOSAssetParam_0p3_0p4)(t) : 0;
	}

protected:
	ConversionBasicIOSAssetParam_0p3_0p4(NxParameterized::Traits* t) : ConversionBasicIOSAssetParam_0p3_0p4Parent(t) {}

	const NxParameterized::PrefVer* getPreferredVersions() const
	{
		static NxParameterized::PrefVer prefVers[] =
		{
			//TODO:
			//	Add your preferred versions for included references here.
			//	Entry format is
			//		{ (const char*)longName, (PxU32)preferredVersion }

			{ 0, 0 } // Terminator (do not remove!)
		};

		return prefVers;
	}

	bool convert()
	{
		// copy 'collisionGroupMaskName' to 'collisionFilterDataName'
		NxParameterized::Handle handle(*mNewData, "collisionFilterDataName");
		handle.setParamString(mLegacyData->collisionGroupMaskName);

		return true;
	}
};

} // namespace legacy
} // namespace apex
} // namespace physx

#endif
