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
#include "NoiseFSAssetParams_0p0.h"
#include "NoiseFSAssetParams_0p1.h"
#include "ConversionNoiseFSAssetParams_0p0_0p1.h"
#include "AttractorFSAssetParams_0p0.h"
#include "AttractorFSAssetParams_0p1.h"
#include "ConversionAttractorFSAssetParams_0p0_0p1.h"
#include "JetFSAssetParams_0p0.h"
#include "JetFSAssetParams_0p1.h"
#include "ConversionJetFSAssetParams_0p0_0p1.h"
#include "NoiseFSAssetParams_0p2.h"
#include "ConversionNoiseFSAssetParams_0p1_0p2.h"
#include "VortexFSAssetParams_0p0.h"
#include "VortexFSAssetParams_0p1.h"
#include "ConversionVortexFSAssetParams_0p0_0p1.h"
#include "WindFSAssetParams_0p0.h"
#include "WindFSAssetParams_0p1.h"
#include "ConversionWindFSAssetParams_0p0_0p1.h"
#include "VortexFSAssetParams_0p2.h"
#include "ConversionVortexFSAssetParams_0p1_0p2.h"
// AUTO_GENERATED_INCLUDES_END

namespace physx
{
namespace apex
{
namespace legacy
{

// AUTO_GENERATED_OBJECTS_BEGIN
static NoiseFSAssetParams_0p0Factory factory_NoiseFSAssetParams_0p0;
static NoiseFSAssetParams_0p1Factory factory_NoiseFSAssetParams_0p1;
static AttractorFSAssetParams_0p0Factory factory_AttractorFSAssetParams_0p0;
static AttractorFSAssetParams_0p1Factory factory_AttractorFSAssetParams_0p1;
static JetFSAssetParams_0p0Factory factory_JetFSAssetParams_0p0;
static JetFSAssetParams_0p1Factory factory_JetFSAssetParams_0p1;
static NoiseFSAssetParams_0p2Factory factory_NoiseFSAssetParams_0p2;
static VortexFSAssetParams_0p0Factory factory_VortexFSAssetParams_0p0;
static VortexFSAssetParams_0p1Factory factory_VortexFSAssetParams_0p1;
static WindFSAssetParams_0p0Factory factory_WindFSAssetParams_0p0;
static WindFSAssetParams_0p1Factory factory_WindFSAssetParams_0p1;
static VortexFSAssetParams_0p2Factory factory_VortexFSAssetParams_0p2;
// AUTO_GENERATED_OBJECTS_END

static LegacyClassEntry ModuleBasicFSLegacyObjects[] = {
	// AUTO_GENERATED_TABLE_BEGIN
	{
		0,
		1,
		&factory_NoiseFSAssetParams_0p0,
		NoiseFSAssetParams_0p0::freeParameterDefinitionTable,
		ConversionNoiseFSAssetParams_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_AttractorFSAssetParams_0p0,
		AttractorFSAssetParams_0p0::freeParameterDefinitionTable,
		ConversionAttractorFSAssetParams_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_JetFSAssetParams_0p0,
		JetFSAssetParams_0p0::freeParameterDefinitionTable,
		ConversionJetFSAssetParams_0p0_0p1::Create,
		0
	},
	{
		1,
		2,
		&factory_NoiseFSAssetParams_0p1,
		NoiseFSAssetParams_0p1::freeParameterDefinitionTable,
		ConversionNoiseFSAssetParams_0p1_0p2::Create,
		0
	},
	{
		0,
		1,
		&factory_VortexFSAssetParams_0p0,
		VortexFSAssetParams_0p0::freeParameterDefinitionTable,
		ConversionVortexFSAssetParams_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_WindFSAssetParams_0p0,
		WindFSAssetParams_0p0::freeParameterDefinitionTable,
		ConversionWindFSAssetParams_0p0_0p1::Create,
		0
	},
	{
		1,
		2,
		&factory_VortexFSAssetParams_0p1,
		VortexFSAssetParams_0p1::freeParameterDefinitionTable,
		ConversionVortexFSAssetParams_0p1_0p2::Create,
		0
	},
	// AUTO_GENERATED_TABLE_END

	{ 0, 0, 0, 0, 0, 0} // Terminator
};

class ModuleBasicFSLegacy : public ApexLegacyModule, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleBasicFSLegacy( NiApexSDK* sdk );

protected:
	void releaseLegacyObjects();

private:

	// Add custom conversions here

};

	DEFINE_INSTANTIATE_MODULE(ModuleBasicFSLegacy)

ModuleBasicFSLegacy::ModuleBasicFSLegacy( NiApexSDK* inSdk )
{
	name = "BasicFS_Legacy";
	mSdk = inSdk;
	mApiProxy = this;

	// Register legacy stuff

	NxParameterized::Traits *t = mSdk->getParameterizedTraits();
	if( !t )
		return;

	// Register auto-generated objects
	registerLegacyObjects(ModuleBasicFSLegacyObjects);

	// Register custom conversions here
}

void ModuleBasicFSLegacy::releaseLegacyObjects()
{
	//Release legacy stuff

	NxParameterized::Traits *t = mSdk->getParameterizedTraits();
	if( !t )
		return;

	// Unregister auto-generated objects
	unregisterLegacyObjects(ModuleBasicFSLegacyObjects);

	// Unregister custom conversions here
}

}
}
} // end namespace physx::apex
