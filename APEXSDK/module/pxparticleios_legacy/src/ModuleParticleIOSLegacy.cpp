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
#include "ParticleIosAssetParam_0p0.h"
#include "ParticleIosAssetParam_0p1.h"
#include "ConversionParticleIosAssetParam_0p0_0p1.h"
#include "ParticleIosAssetParam_0p2.h"
#include "ConversionParticleIosAssetParam_0p1_0p2.h"
#include "SimpleParticleSystemParams_0p0.h"
#include "SimpleParticleSystemParams_0p1.h"
#include "ConversionSimpleParticleSystemParams_0p0_0p1.h"
#include "SimpleParticleSystemParams_0p2.h"
#include "ConversionSimpleParticleSystemParams_0p1_0p2.h"
#include "SimpleParticleSystemParams_0p3.h"
#include "ConversionSimpleParticleSystemParams_0p2_0p3.h"
#include "ParticleIosAssetParam_0p3.h"
#include "ConversionParticleIosAssetParam_0p2_0p3.h"
#include "ParticleIosAssetParam_0p4.h"
#include "ConversionParticleIosAssetParam_0p3_0p4.h"
// AUTO_GENERATED_INCLUDES_END

namespace physx
{
namespace apex
{
namespace legacy
{

// AUTO_GENERATED_OBJECTS_BEGIN
static ParticleIosAssetParam_0p0Factory factory_ParticleIosAssetParam_0p0;
static ParticleIosAssetParam_0p1Factory factory_ParticleIosAssetParam_0p1;
static ParticleIosAssetParam_0p2Factory factory_ParticleIosAssetParam_0p2;
static SimpleParticleSystemParams_0p0Factory factory_SimpleParticleSystemParams_0p0;
static SimpleParticleSystemParams_0p1Factory factory_SimpleParticleSystemParams_0p1;
static SimpleParticleSystemParams_0p2Factory factory_SimpleParticleSystemParams_0p2;
static SimpleParticleSystemParams_0p3Factory factory_SimpleParticleSystemParams_0p3;
static ParticleIosAssetParam_0p3Factory factory_ParticleIosAssetParam_0p3;
static ParticleIosAssetParam_0p4Factory factory_ParticleIosAssetParam_0p4;
// AUTO_GENERATED_OBJECTS_END

static LegacyClassEntry ModuleParticleIOSLegacyObjects[] = {
	// AUTO_GENERATED_TABLE_BEGIN
	{
		0,
		1,
		&factory_ParticleIosAssetParam_0p0,
		ParticleIosAssetParam_0p0::freeParameterDefinitionTable,
		ConversionParticleIosAssetParam_0p0_0p1::Create,
		0
	},
	{
		1,
		2,
		&factory_ParticleIosAssetParam_0p1,
		ParticleIosAssetParam_0p1::freeParameterDefinitionTable,
		ConversionParticleIosAssetParam_0p1_0p2::Create,
		0
	},
	{
		0,
		1,
		&factory_SimpleParticleSystemParams_0p0,
		SimpleParticleSystemParams_0p0::freeParameterDefinitionTable,
		ConversionSimpleParticleSystemParams_0p0_0p1::Create,
		0
	},
	{
		1,
		2,
		&factory_SimpleParticleSystemParams_0p1,
		SimpleParticleSystemParams_0p1::freeParameterDefinitionTable,
		ConversionSimpleParticleSystemParams_0p1_0p2::Create,
		0
	},
	{
		2,
		3,
		&factory_SimpleParticleSystemParams_0p2,
		SimpleParticleSystemParams_0p2::freeParameterDefinitionTable,
		ConversionSimpleParticleSystemParams_0p2_0p3::Create,
		0
	},
	{
		2,
		3,
		&factory_ParticleIosAssetParam_0p2,
		ParticleIosAssetParam_0p2::freeParameterDefinitionTable,
		ConversionParticleIosAssetParam_0p2_0p3::Create,
		0
	},
	{
		3,
		4,
		&factory_ParticleIosAssetParam_0p3,
		ParticleIosAssetParam_0p3::freeParameterDefinitionTable,
		ConversionParticleIosAssetParam_0p3_0p4::Create,
		0
	},
	// AUTO_GENERATED_TABLE_END

	{ 0, 0, 0, 0, 0, 0} // Terminator
};

class ModuleParticleIOSLegacy : public ApexLegacyModule, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleParticleIOSLegacy( NiApexSDK* sdk );

protected:
	void releaseLegacyObjects();

private:

	// Add custom conversions here

};

DEFINE_INSTANTIATE_MODULE(ModuleParticleIOSLegacy)

ModuleParticleIOSLegacy::ModuleParticleIOSLegacy( NiApexSDK* inSdk )
{
	name = "ParticleIOS_Legacy";
	mSdk = inSdk;
	mApiProxy = this;

	// Register legacy stuff

	NxParameterized::Traits *t = mSdk->getParameterizedTraits();
	if( !t )
		return;

	// Register auto-generated objects
	registerLegacyObjects(ModuleParticleIOSLegacyObjects);

	// Register custom conversions here
}

void ModuleParticleIOSLegacy::releaseLegacyObjects()
{
	//Release legacy stuff

	NxParameterized::Traits *t = mSdk->getParameterizedTraits();
	if( !t )
		return;

	// Unregister auto-generated objects
	unregisterLegacyObjects(ModuleParticleIOSLegacyObjects);

	// Unregister custom conversions here
}

} // namespace legacy
}
} // end namespace physx::apex
