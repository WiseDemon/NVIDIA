/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "PsShare.h"
#include "NxApex.h"
#include "ApexLegacyModule.h"
#include "ApexRWLockable.h"

// AUTO_GENERATED_INCLUDES_BEGIN
#include "BasicIOSAssetParam_0p0.h"
#include "BasicIOSAssetParam_0p1.h"
#include "BasicIOSAssetParam_0p2.h"
#include "ConversionBasicIOSAssetParam_0p1_0p2.h"
#include "BasicIOSAssetParam_0p3.h"
#include "ConversionBasicIOSAssetParam_0p2_0p3.h"
#include "ConversionBasicIOSAssetParam_0p0_0p1.h"
#include "BasicIOSAssetParam_0p4.h"
#include "ConversionBasicIOSAssetParam_0p3_0p4.h"
#include "BasicIOSAssetParam_0p5.h"
#include "ConversionBasicIOSAssetParam_0p4_0p5.h"
#include "BasicIOSAssetParam_0p6.h"
#include "ConversionBasicIOSAssetParam_0p5_0p6.h"
#include "BasicIOSAssetParam_0p7.h"
#include "ConversionBasicIOSAssetParam_0p6_0p7.h"
#include "BasicIOSAssetParam_0p8.h"
#include "ConversionBasicIOSAssetParam_0p7_0p8.h"
#include "BasicIOSAssetParam_0p9.h"
#include "ConversionBasicIOSAssetParam_0p8_0p9.h"
#include "BasicIOSAssetParam_1p0.h"
#include "ConversionBasicIOSAssetParam_0p9_1p0.h"
#include "BasicIOSAssetParam_1p1.h"
#include "ConversionBasicIOSAssetParam_1p0_1p1.h"
#include "BasicIOSAssetParam_1p2.h"
#include "ConversionBasicIOSAssetParam_1p1_1p2.h"
#include "BasicIOSAssetParam_1p3.h"
#include "ConversionBasicIOSAssetParam_1p2_1p3.h"
#include "BasicIOSAssetParam_1p4.h"
#include "ConversionBasicIOSAssetParam_1p3_1p4.h"
// AUTO_GENERATED_INCLUDES_END

namespace physx
{
namespace apex
{
namespace legacy
{

// AUTO_GENERATED_OBJECTS_BEGIN
static BasicIOSAssetParam_0p0Factory factory_BasicIOSAssetParam_0p0;
static BasicIOSAssetParam_0p1Factory factory_BasicIOSAssetParam_0p1;
static BasicIOSAssetParam_0p2Factory factory_BasicIOSAssetParam_0p2;
static BasicIOSAssetParam_0p3Factory factory_BasicIOSAssetParam_0p3;
static BasicIOSAssetParam_0p4Factory factory_BasicIOSAssetParam_0p4;
static BasicIOSAssetParam_0p5Factory factory_BasicIOSAssetParam_0p5;
static BasicIOSAssetParam_0p6Factory factory_BasicIOSAssetParam_0p6;
static BasicIOSAssetParam_0p7Factory factory_BasicIOSAssetParam_0p7;
static BasicIOSAssetParam_0p8Factory factory_BasicIOSAssetParam_0p8;
static BasicIOSAssetParam_0p9Factory factory_BasicIOSAssetParam_0p9;
static BasicIOSAssetParam_1p0Factory factory_BasicIOSAssetParam_1p0;
static BasicIOSAssetParam_1p1Factory factory_BasicIOSAssetParam_1p1;
static BasicIOSAssetParam_1p2Factory factory_BasicIOSAssetParam_1p2;
static BasicIOSAssetParam_1p3Factory factory_BasicIOSAssetParam_1p3;
static BasicIOSAssetParam_1p4Factory factory_BasicIOSAssetParam_1p4;
// AUTO_GENERATED_OBJECTS_END

static LegacyClassEntry ModuleBasicIOSLegacyObjects[] =
{
	// AUTO_GENERATED_TABLE_BEGIN
	{
		0,
		1,
		&factory_BasicIOSAssetParam_0p0,
		BasicIOSAssetParam_0p0::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_0p0_0p1::Create,
		0
	},
	{
		1,
		2,
		&factory_BasicIOSAssetParam_0p1,
		BasicIOSAssetParam_0p1::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_0p1_0p2::Create,
		0
	},
	{
		2,
		3,
		&factory_BasicIOSAssetParam_0p2,
		BasicIOSAssetParam_0p2::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_0p2_0p3::Create,
		0
	},
	{
		3,
		4,
		&factory_BasicIOSAssetParam_0p3,
		BasicIOSAssetParam_0p3::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_0p3_0p4::Create,
		0
	},
	{
		4,
		5,
		&factory_BasicIOSAssetParam_0p4,
		BasicIOSAssetParam_0p4::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_0p4_0p5::Create,
		0
	},
	{
		5,
		6,
		&factory_BasicIOSAssetParam_0p5,
		BasicIOSAssetParam_0p5::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_0p5_0p6::Create,
		0
	},
	{
		6,
		7,
		&factory_BasicIOSAssetParam_0p6,
		BasicIOSAssetParam_0p6::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_0p6_0p7::Create,
		0
	},
	{
		7,
		8,
		&factory_BasicIOSAssetParam_0p7,
		BasicIOSAssetParam_0p7::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_0p7_0p8::Create,
		0
	},
	{
		8,
		9,
		&factory_BasicIOSAssetParam_0p8,
		BasicIOSAssetParam_0p8::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_0p8_0p9::Create,
		0
	},
	{
		9,
		65536,
		&factory_BasicIOSAssetParam_0p9,
		BasicIOSAssetParam_0p9::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_0p9_1p0::Create,
		0
	},
	{
		65536,
		65537,
		&factory_BasicIOSAssetParam_1p0,
		BasicIOSAssetParam_1p0::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_1p0_1p1::Create,
		0
	},
	{
		65537,
		65538,
		&factory_BasicIOSAssetParam_1p1,
		BasicIOSAssetParam_1p1::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_1p1_1p2::Create,
		0
	},
	{
		65538,
		65539,
		&factory_BasicIOSAssetParam_1p2,
		BasicIOSAssetParam_1p2::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_1p2_1p3::Create,
		0
	},
	{
		65539,
		65540,
		&factory_BasicIOSAssetParam_1p3,
		BasicIOSAssetParam_1p3::freeParameterDefinitionTable,
		ConversionBasicIOSAssetParam_1p3_1p4::Create,
		0
	},
	// AUTO_GENERATED_TABLE_END

	{ 0, 0, 0, 0, 0, 0} // Terminator
};

class ModuleBasicIOSLegacy : public ApexLegacyModule, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleBasicIOSLegacy(NiApexSDK* sdk);

protected:
	void releaseLegacyObjects();

private:

	// Add custom conversions here

};

DEFINE_INSTANTIATE_MODULE(ModuleBasicIOSLegacy)

ModuleBasicIOSLegacy::ModuleBasicIOSLegacy(NiApexSDK* inSdk)
{
	name = "BasicIOS_Legacy";
	mSdk = inSdk;
	mApiProxy = this;

	// Register legacy stuff

	NxParameterized::Traits* t = mSdk->getParameterizedTraits();
	if (!t)
	{
		return;
	}

	// Register auto-generated objects
	registerLegacyObjects(ModuleBasicIOSLegacyObjects);

	// Register custom conversions here
}

void ModuleBasicIOSLegacy::releaseLegacyObjects()
{
	//Release legacy stuff

	NxParameterized::Traits* t = mSdk->getParameterizedTraits();
	if (!t)
	{
		return;
	}

	// Unregister auto-generated objects
	unregisterLegacyObjects(ModuleBasicIOSLegacyObjects);

	// Unregister custom conversions here
}

}
}
} // end namespace physx::apex
