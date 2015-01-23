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
#include "ForceFieldAssetParams_0p0.h"
#include "ForceFieldAssetParams_0p1.h"
#include "ConversionForceFieldAssetParams_0p0_0p1.h"
#include "GenericForceFieldKernelParams_0p0.h"
#include "GenericForceFieldKernelParams_0p1.h"
#include "ConversionGenericForceFieldKernelParams_0p0_0p1.h"
// AUTO_GENERATED_INCLUDES_END

namespace physx
{
namespace apex
{
namespace legacy
{

// AUTO_GENERATED_OBJECTS_BEGIN
static ForceFieldAssetParams_0p0Factory factory_ForceFieldAssetParams_0p0;
static ForceFieldAssetParams_0p1Factory factory_ForceFieldAssetParams_0p1;
static GenericForceFieldKernelParams_0p0Factory factory_GenericForceFieldKernelParams_0p0;
static GenericForceFieldKernelParams_0p1Factory factory_GenericForceFieldKernelParams_0p1;
// AUTO_GENERATED_OBJECTS_END

static LegacyClassEntry ModuleForceFieldLegacyObjects[] = {
	// AUTO_GENERATED_TABLE_BEGIN
	{
		0,
		1,
		&factory_ForceFieldAssetParams_0p0,
		ForceFieldAssetParams_0p0::freeParameterDefinitionTable,
		ConversionForceFieldAssetParams_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_GenericForceFieldKernelParams_0p0,
		GenericForceFieldKernelParams_0p0::freeParameterDefinitionTable,
		ConversionGenericForceFieldKernelParams_0p0_0p1::Create,
		0
	},
	// AUTO_GENERATED_TABLE_END

	{ 0, 0, 0, 0, 0, 0} // Terminator
};

class ModuleForceFieldLegacy : public ApexLegacyModule, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleForceFieldLegacy( NiApexSDK* sdk );

protected:
	void releaseLegacyObjects();

private:

	// Add custom conversions here

};

	DEFINE_INSTANTIATE_MODULE(ModuleForceFieldLegacy)

ModuleForceFieldLegacy::ModuleForceFieldLegacy( NiApexSDK* inSdk )
{
	name = "ForceField_Legacy";
	mSdk = inSdk;
	mApiProxy = this;

	// Register legacy stuff

	NxParameterized::Traits *t = mSdk->getParameterizedTraits();
	if( !t )
		return;

	// Register auto-generated objects
	registerLegacyObjects(ModuleForceFieldLegacyObjects);

	// Register custom conversions here
}

void ModuleForceFieldLegacy::releaseLegacyObjects()
{
	//Release legacy stuff

	NxParameterized::Traits *t = mSdk->getParameterizedTraits();
	if( !t )
		return;

	// Unregister auto-generated objects
	unregisterLegacyObjects(ModuleForceFieldLegacyObjects);

	// Unregister custom conversions here
}

}
}
} // end namespace physx::apex
