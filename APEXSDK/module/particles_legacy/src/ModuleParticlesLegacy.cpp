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
#include "GraphicsMaterialData_0p0.h"
#include "GraphicsMaterialData_0p1.h"
#include "ConversionGraphicsMaterialData_0p0_0p1.h"
#include "GraphicsMaterialData_0p2.h"
#include "ConversionGraphicsMaterialData_0p1_0p2.h"
#include "GraphicsMaterialData_0p3.h"
#include "ConversionGraphicsMaterialData_0p2_0p3.h"
#include "EffectPackageAssetParams_0p0.h"
#include "EffectPackageAssetParams_0p1.h"
#include "ConversionEffectPackageAssetParams_0p0_0p1.h"
#include "VolumeRenderMaterialData_0p0.h"
#include "VolumeRenderMaterialData_0p1.h"
#include "ConversionVolumeRenderMaterialData_0p0_0p1.h"
#include "GraphicsMaterialData_0p4.h"
#include "ConversionGraphicsMaterialData_0p3_0p4.h"
#include "EffectPackageDatabaseParams_0p0.h"
#include "EffectPackageDatabaseParams_0p1.h"
#include "ConversionEffectPackageDatabaseParams_0p0_0p1.h"
// AUTO_GENERATED_INCLUDES_END

#include "PsShare.h"
#include "NxApex.h"
#include "ApexLegacyModule.h"

namespace physx
{
namespace apex
{
namespace legacy
{

// AUTO_GENERATED_OBJECTS_BEGIN
static GraphicsMaterialData_0p0Factory factory_GraphicsMaterialData_0p0;
static GraphicsMaterialData_0p1Factory factory_GraphicsMaterialData_0p1;
static GraphicsMaterialData_0p2Factory factory_GraphicsMaterialData_0p2;
static GraphicsMaterialData_0p3Factory factory_GraphicsMaterialData_0p3;
static EffectPackageAssetParams_0p0Factory factory_EffectPackageAssetParams_0p0;
static EffectPackageAssetParams_0p1Factory factory_EffectPackageAssetParams_0p1;
static VolumeRenderMaterialData_0p0Factory factory_VolumeRenderMaterialData_0p0;
static VolumeRenderMaterialData_0p1Factory factory_VolumeRenderMaterialData_0p1;
static GraphicsMaterialData_0p4Factory factory_GraphicsMaterialData_0p4;
static EffectPackageDatabaseParams_0p0Factory factory_EffectPackageDatabaseParams_0p0;
static EffectPackageDatabaseParams_0p1Factory factory_EffectPackageDatabaseParams_0p1;
// AUTO_GENERATED_OBJECTS_END

static LegacyClassEntry ModuleParticlesLegacyObjects[] = {
	// AUTO_GENERATED_TABLE_BEGIN
	{
		0,
		1,
		&factory_GraphicsMaterialData_0p0,
		GraphicsMaterialData_0p0::freeParameterDefinitionTable,
		ConversionGraphicsMaterialData_0p0_0p1::Create,
		0
	},
	{
		1,
		2,
		&factory_GraphicsMaterialData_0p1,
		GraphicsMaterialData_0p1::freeParameterDefinitionTable,
		ConversionGraphicsMaterialData_0p1_0p2::Create,
		0
	},
	{
		2,
		3,
		&factory_GraphicsMaterialData_0p2,
		GraphicsMaterialData_0p2::freeParameterDefinitionTable,
		ConversionGraphicsMaterialData_0p2_0p3::Create,
		0
	},
	{
		0,
		1,
		&factory_EffectPackageAssetParams_0p0,
		EffectPackageAssetParams_0p0::freeParameterDefinitionTable,
		ConversionEffectPackageAssetParams_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_VolumeRenderMaterialData_0p0,
		VolumeRenderMaterialData_0p0::freeParameterDefinitionTable,
		ConversionVolumeRenderMaterialData_0p0_0p1::Create,
		0
	},
	{
		3,
		4,
		&factory_GraphicsMaterialData_0p3,
		GraphicsMaterialData_0p3::freeParameterDefinitionTable,
		ConversionGraphicsMaterialData_0p3_0p4::Create,
		0
	},
	{
		0,
		1,
		&factory_EffectPackageDatabaseParams_0p0,
		EffectPackageDatabaseParams_0p0::freeParameterDefinitionTable,
		ConversionEffectPackageDatabaseParams_0p0_0p1::Create,
		0
	},
	// AUTO_GENERATED_TABLE_END

	{ 0, 0, 0, 0, 0, 0} // Terminator
};

class ModuleParticlesLegacy : public ApexLegacyModule, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleParticlesLegacy( NiApexSDK* sdk );

protected:
	void releaseLegacyObjects();

private:

	// Add custom conversions here

};

	DEFINE_INSTANTIATE_MODULE(ModuleParticlesLegacy)

ModuleParticlesLegacy::ModuleParticlesLegacy( NiApexSDK* inSdk )
{
	name = "Particles_Legacy";
	mSdk = inSdk;
	mApiProxy = this;

	// Register legacy stuff

	NxParameterized::Traits *t = mSdk->getParameterizedTraits();
	if( !t )
		return;

	// Register auto-generated objects
	registerLegacyObjects(ModuleParticlesLegacyObjects);

	// Register custom conversions here
}

void ModuleParticlesLegacy::releaseLegacyObjects()
{
	//Release legacy stuff

	NxParameterized::Traits *t = mSdk->getParameterizedTraits();
	if( !t )
		return;

	// Unregister auto-generated objects
	unregisterLegacyObjects(ModuleParticlesLegacyObjects);

	// Unregister custom conversions here
}

}
}
} // end namespace physx::apex
