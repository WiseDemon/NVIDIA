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
#include "IofxAssetParameters_0p0.h"
#include "IofxAssetParameters_0p1.h"
#include "ConversionIofxAssetParameters_0p0_0p1.h"
#include "SpriteIofxParameters_0p0.h"
#include "SpriteIofxParameters_0p1.h"
#include "ConversionSpriteIofxParameters_0p0_0p1.h"
#include "MeshIofxParameters_0p0.h"
#include "MeshIofxParameters_0p1.h"
#include "ConversionMeshIofxParameters_0p0_0p1.h"
#include "RotationModifierParams_0p0.h"
#include "RotationModifierParams_0p1.h"
#include "ConversionRotationModifierParams_0p0_0p1.h"
#include "SpriteIofxParameters_0p2.h"
#include "ConversionSpriteIofxParameters_0p1_0p2.h"
#include "MeshIofxParameters_0p2.h"
#include "ConversionMeshIofxParameters_0p1_0p2.h"
#include "IofxAssetParameters_0p2.h"
#include "ConversionIofxAssetParameters_0p1_0p2.h"
#include "SpriteIofxParameters_0p3.h"
#include "ConversionSpriteIofxParameters_0p2_0p3.h"
#include "MeshIofxParameters_0p3.h"
#include "ConversionMeshIofxParameters_0p2_0p3.h"
#include "RotationModifierParams_0p2.h"
#include "ConversionRotationModifierParams_0p1_0p2.h"
#include "OrientScaleAlongScreenVelocityModifierParams_0p0.h"
#include "OrientScaleAlongScreenVelocityModifierParams_0p1.h"
#include "ConversionOrientScaleAlongScreenVelocityModifierParams_0p0_0p1.h"
#include "SpriteIofxParameters_0p4.h"
#include "ConversionSpriteIofxParameters_0p3_0p4.h"
#include "MeshIofxParameters_0p4.h"
#include "ConversionMeshIofxParameters_0p3_0p4.h"
// AUTO_GENERATED_INCLUDES_END

namespace physx
{
namespace apex
{
namespace legacy
{

// AUTO_GENERATED_OBJECTS_BEGIN
static IofxAssetParameters_0p0Factory factory_IofxAssetParameters_0p0;
static IofxAssetParameters_0p1Factory factory_IofxAssetParameters_0p1;
static SpriteIofxParameters_0p0Factory factory_SpriteIofxParameters_0p0;
static SpriteIofxParameters_0p1Factory factory_SpriteIofxParameters_0p1;
static MeshIofxParameters_0p0Factory factory_MeshIofxParameters_0p0;
static MeshIofxParameters_0p1Factory factory_MeshIofxParameters_0p1;
static RotationModifierParams_0p0Factory factory_RotationModifierParams_0p0;
static RotationModifierParams_0p1Factory factory_RotationModifierParams_0p1;
static SpriteIofxParameters_0p2Factory factory_SpriteIofxParameters_0p2;
static MeshIofxParameters_0p2Factory factory_MeshIofxParameters_0p2;
static IofxAssetParameters_0p2Factory factory_IofxAssetParameters_0p2;
static SpriteIofxParameters_0p3Factory factory_SpriteIofxParameters_0p3;
static MeshIofxParameters_0p3Factory factory_MeshIofxParameters_0p3;
static RotationModifierParams_0p2Factory factory_RotationModifierParams_0p2;
static OrientScaleAlongScreenVelocityModifierParams_0p0Factory factory_OrientScaleAlongScreenVelocityModifierParams_0p0;
static OrientScaleAlongScreenVelocityModifierParams_0p1Factory factory_OrientScaleAlongScreenVelocityModifierParams_0p1;
static SpriteIofxParameters_0p4Factory factory_SpriteIofxParameters_0p4;
static MeshIofxParameters_0p4Factory factory_MeshIofxParameters_0p4;
// AUTO_GENERATED_OBJECTS_END

static LegacyClassEntry ModuleIOFXLegacyObjects[] =
{
	// AUTO_GENERATED_TABLE_BEGIN
	{
		0,
		1,
		&factory_IofxAssetParameters_0p0,
		IofxAssetParameters_0p0::freeParameterDefinitionTable,
		ConversionIofxAssetParameters_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_SpriteIofxParameters_0p0,
		SpriteIofxParameters_0p0::freeParameterDefinitionTable,
		ConversionSpriteIofxParameters_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_MeshIofxParameters_0p0,
		MeshIofxParameters_0p0::freeParameterDefinitionTable,
		ConversionMeshIofxParameters_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_RotationModifierParams_0p0,
		RotationModifierParams_0p0::freeParameterDefinitionTable,
		ConversionRotationModifierParams_0p0_0p1::Create,
		0
	},
	{
		1,
		2,
		&factory_SpriteIofxParameters_0p1,
		SpriteIofxParameters_0p1::freeParameterDefinitionTable,
		ConversionSpriteIofxParameters_0p1_0p2::Create,
		0
	},
	{
		1,
		2,
		&factory_MeshIofxParameters_0p1,
		MeshIofxParameters_0p1::freeParameterDefinitionTable,
		ConversionMeshIofxParameters_0p1_0p2::Create,
		0
	},
	{
		1,
		2,
		&factory_IofxAssetParameters_0p1,
		IofxAssetParameters_0p1::freeParameterDefinitionTable,
		ConversionIofxAssetParameters_0p1_0p2::Create,
		0
	},
	{
		2,
		3,
		&factory_SpriteIofxParameters_0p2,
		SpriteIofxParameters_0p2::freeParameterDefinitionTable,
		ConversionSpriteIofxParameters_0p2_0p3::Create,
		0
	},
	{
		2,
		3,
		&factory_MeshIofxParameters_0p2,
		MeshIofxParameters_0p2::freeParameterDefinitionTable,
		ConversionMeshIofxParameters_0p2_0p3::Create,
		0
	},
	{
		1,
		2,
		&factory_RotationModifierParams_0p1,
		RotationModifierParams_0p1::freeParameterDefinitionTable,
		ConversionRotationModifierParams_0p1_0p2::Create,
		0
	},
	{
		0,
		1,
		&factory_OrientScaleAlongScreenVelocityModifierParams_0p0,
		OrientScaleAlongScreenVelocityModifierParams_0p0::freeParameterDefinitionTable,
		ConversionOrientScaleAlongScreenVelocityModifierParams_0p0_0p1::Create,
		0
	},
	{
		3,
		4,
		&factory_SpriteIofxParameters_0p3,
		SpriteIofxParameters_0p3::freeParameterDefinitionTable,
		ConversionSpriteIofxParameters_0p3_0p4::Create,
		0
	},
	{
		3,
		4,
		&factory_MeshIofxParameters_0p3,
		MeshIofxParameters_0p3::freeParameterDefinitionTable,
		ConversionMeshIofxParameters_0p3_0p4::Create,
		0
	},
	// AUTO_GENERATED_TABLE_END

	{ 0, 0, 0, 0, 0, 0} // Terminator
};

class ModuleIOFXLegacy : public ApexLegacyModule, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleIOFXLegacy(NiApexSDK* sdk);

protected:
	void releaseLegacyObjects();

private:

	// Add custom conversions here

};

DEFINE_INSTANTIATE_MODULE(ModuleIOFXLegacy)

ModuleIOFXLegacy::ModuleIOFXLegacy(NiApexSDK* inSdk)
{
	name = "IOFX_Legacy";
	mSdk = inSdk;
	mApiProxy = this;

	// Register legacy stuff

	NxParameterized::Traits* t = mSdk->getParameterizedTraits();
	if (!t)
	{
		return;
	}

	// Register auto-generated objects
	registerLegacyObjects(ModuleIOFXLegacyObjects);

	// Register custom conversions here
}

void ModuleIOFXLegacy::releaseLegacyObjects()
{
	//Release legacy stuff

	NxParameterized::Traits* t = mSdk->getParameterizedTraits();
	if (!t)
	{
		return;
	}

	// Unregister auto-generated objects
	unregisterLegacyObjects(ModuleIOFXLegacyObjects);

	// Unregister custom conversions here
}

}
}
} // end namespace physx::apex
