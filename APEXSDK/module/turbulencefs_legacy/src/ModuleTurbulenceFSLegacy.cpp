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
#include "TurbulenceFSAssetParams_0p0.h"
#include "TurbulenceFSAssetParams_0p1.h"
#include "ConversionTurbulenceFSAssetParams_0p0_0p1.h"
#include "TurbulenceFSAssetParams_0p2.h"
#include "ConversionTurbulenceFSAssetParams_0p1_0p2.h"
#include "TurbulenceFSAssetParams_0p3.h"
#include "ConversionTurbulenceFSAssetParams_0p2_0p3.h"
#include "TurbulenceFSAssetParams_0p4.h"
#include "ConversionTurbulenceFSAssetParams_0p3_0p4.h"
#include "TurbulenceFSAssetParams_0p5.h"
#include "ConversionTurbulenceFSAssetParams_0p4_0p5.h"
#include "HeatSourceAssetParams_0p0.h"
#include "HeatSourceAssetParams_0p1.h"
#include "ConversionHeatSourceAssetParams_0p0_0p1.h"
#include "TurbulenceFSAssetParams_0p6.h"
#include "ConversionTurbulenceFSAssetParams_0p5_0p6.h"
#include "TurbulenceFSAssetParams_0p7.h"
#include "ConversionTurbulenceFSAssetParams_0p6_0p7.h"
#include "TurbulenceFSAssetParams_0p8.h"
#include "ConversionTurbulenceFSAssetParams_0p7_0p8.h"
#include "TurbulenceFSAssetParams_0p9.h"
#include "ConversionTurbulenceFSAssetParams_0p8_0p9.h"
#include "TurbulenceFSAssetParams_1p0.h"
#include "ConversionTurbulenceFSAssetParams_0p9_1p0.h"
#include "HeatSourceAssetParams_0p2.h"
#include "ConversionHeatSourceAssetParams_0p1_0p2.h"
#include "SubstanceSourceAssetParams_0p0.h"
#include "SubstanceSourceAssetParams_0p1.h"
#include "ConversionSubstanceSourceAssetParams_0p0_0p1.h"
#include "TurbulenceFSAssetParams_1p1.h"
#include "ConversionTurbulenceFSAssetParams_1p0_1p1.h"
#include "TurbulenceFSAssetParams_1p2.h"
#include "ConversionTurbulenceFSAssetParams_1p1_1p2.h"
#include "TurbulenceFSAssetParams_1p3.h"
#include "ConversionTurbulenceFSAssetParams_1p2_1p3.h"
#include "TurbulenceFSAssetParams_1p4.h"
#include "ConversionTurbulenceFSAssetParams_1p3_1p4.h"
// AUTO_GENERATED_INCLUDES_END

namespace physx
{
namespace apex
{
namespace legacy
{

// AUTO_GENERATED_OBJECTS_BEGIN
static TurbulenceFSAssetParams_0p0Factory factory_TurbulenceFSAssetParams_0p0;
static TurbulenceFSAssetParams_0p1Factory factory_TurbulenceFSAssetParams_0p1;
static TurbulenceFSAssetParams_0p2Factory factory_TurbulenceFSAssetParams_0p2;
static TurbulenceFSAssetParams_0p3Factory factory_TurbulenceFSAssetParams_0p3;
static TurbulenceFSAssetParams_0p4Factory factory_TurbulenceFSAssetParams_0p4;
static TurbulenceFSAssetParams_0p5Factory factory_TurbulenceFSAssetParams_0p5;
static HeatSourceAssetParams_0p0Factory factory_HeatSourceAssetParams_0p0;
static HeatSourceAssetParams_0p1Factory factory_HeatSourceAssetParams_0p1;
static TurbulenceFSAssetParams_0p6Factory factory_TurbulenceFSAssetParams_0p6;
static TurbulenceFSAssetParams_0p7Factory factory_TurbulenceFSAssetParams_0p7;
static TurbulenceFSAssetParams_0p8Factory factory_TurbulenceFSAssetParams_0p8;
static TurbulenceFSAssetParams_0p9Factory factory_TurbulenceFSAssetParams_0p9;
static TurbulenceFSAssetParams_1p0Factory factory_TurbulenceFSAssetParams_1p0;
static HeatSourceAssetParams_0p2Factory factory_HeatSourceAssetParams_0p2;
static SubstanceSourceAssetParams_0p0Factory factory_SubstanceSourceAssetParams_0p0;
static SubstanceSourceAssetParams_0p1Factory factory_SubstanceSourceAssetParams_0p1;
static TurbulenceFSAssetParams_1p1Factory factory_TurbulenceFSAssetParams_1p1;
static TurbulenceFSAssetParams_1p2Factory factory_TurbulenceFSAssetParams_1p2;
static TurbulenceFSAssetParams_1p3Factory factory_TurbulenceFSAssetParams_1p3;
static TurbulenceFSAssetParams_1p4Factory factory_TurbulenceFSAssetParams_1p4;
// AUTO_GENERATED_OBJECTS_END

static LegacyClassEntry ModuleTurbulenceFSLegacyObjects[] = {
	// AUTO_GENERATED_TABLE_BEGIN
	{
		0,
		1,
		&factory_TurbulenceFSAssetParams_0p0,
		TurbulenceFSAssetParams_0p0::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_0p0_0p1::Create,
		0
	},
	{
		1,
		2,
		&factory_TurbulenceFSAssetParams_0p1,
		TurbulenceFSAssetParams_0p1::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_0p1_0p2::Create,
		0
	},
	{
		2,
		3,
		&factory_TurbulenceFSAssetParams_0p2,
		TurbulenceFSAssetParams_0p2::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_0p2_0p3::Create,
		0
	},
	{
		3,
		4,
		&factory_TurbulenceFSAssetParams_0p3,
		TurbulenceFSAssetParams_0p3::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_0p3_0p4::Create,
		0
	},
	{
		4,
		5,
		&factory_TurbulenceFSAssetParams_0p4,
		TurbulenceFSAssetParams_0p4::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_0p4_0p5::Create,
		0
	},
	{
		0,
		1,
		&factory_HeatSourceAssetParams_0p0,
		HeatSourceAssetParams_0p0::freeParameterDefinitionTable,
		ConversionHeatSourceAssetParams_0p0_0p1::Create,
		0
	},
	{
		5,
		6,
		&factory_TurbulenceFSAssetParams_0p5,
		TurbulenceFSAssetParams_0p5::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_0p5_0p6::Create,
		0
	},
	{
		6,
		7,
		&factory_TurbulenceFSAssetParams_0p6,
		TurbulenceFSAssetParams_0p6::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_0p6_0p7::Create,
		0
	},
	{
		7,
		8,
		&factory_TurbulenceFSAssetParams_0p7,
		TurbulenceFSAssetParams_0p7::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_0p7_0p8::Create,
		0
	},
	{
		8,
		9,
		&factory_TurbulenceFSAssetParams_0p8,
		TurbulenceFSAssetParams_0p8::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_0p8_0p9::Create,
		0
	},
	{
		9,
		65536,
		&factory_TurbulenceFSAssetParams_0p9,
		TurbulenceFSAssetParams_0p9::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_0p9_1p0::Create,
		0
	},
	{
		1,
		2,
		&factory_HeatSourceAssetParams_0p1,
		HeatSourceAssetParams_0p1::freeParameterDefinitionTable,
		ConversionHeatSourceAssetParams_0p1_0p2::Create,
		0
	},
	{
		0,
		1,
		&factory_SubstanceSourceAssetParams_0p0,
		SubstanceSourceAssetParams_0p0::freeParameterDefinitionTable,
		ConversionSubstanceSourceAssetParams_0p0_0p1::Create,
		0
	},
	{
		65536,
		65537,
		&factory_TurbulenceFSAssetParams_1p0,
		TurbulenceFSAssetParams_1p0::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_1p0_1p1::Create,
		0
	},
	{
		65537,
		65538,
		&factory_TurbulenceFSAssetParams_1p1,
		TurbulenceFSAssetParams_1p1::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_1p1_1p2::Create,
		0
	},
	{
		65538,
		65539,
		&factory_TurbulenceFSAssetParams_1p2,
		TurbulenceFSAssetParams_1p2::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_1p2_1p3::Create,
		0
	},
	{
		65539,
		65540,
		&factory_TurbulenceFSAssetParams_1p3,
		TurbulenceFSAssetParams_1p3::freeParameterDefinitionTable,
		ConversionTurbulenceFSAssetParams_1p3_1p4::Create,
		0
	},
	// AUTO_GENERATED_TABLE_END

	{ 0, 0, 0, 0, 0, 0} // Terminator
};

class ModuleTurbulenceFSLegacy : public ApexLegacyModule, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleTurbulenceFSLegacy( NiApexSDK* sdk );

protected:
	void releaseLegacyObjects();

private:

	// Add custom conversions here

};

DEFINE_INSTANTIATE_MODULE(ModuleTurbulenceFSLegacy)

ModuleTurbulenceFSLegacy::ModuleTurbulenceFSLegacy( NiApexSDK* inSdk )
{
	name = "TurbulenceFS_Legacy";
	mSdk = inSdk;
	mApiProxy = this;

	// Register legacy stuff

	NxParameterized::Traits *t = mSdk->getParameterizedTraits();
	if( !t )
		return;

	// Register auto-generated objects
	registerLegacyObjects(ModuleTurbulenceFSLegacyObjects);

	// Register custom conversions here
}

void ModuleTurbulenceFSLegacy::releaseLegacyObjects()
{
	//Release legacy stuff

	NxParameterized::Traits *t = mSdk->getParameterizedTraits();
	if( !t )
		return;

	// Unregister auto-generated objects
	unregisterLegacyObjects(ModuleTurbulenceFSLegacyObjects);

	// Unregister custom conversions here
}

} // namespace legacy
}
} // end namespace physx::apex
