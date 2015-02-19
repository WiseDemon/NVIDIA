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
#include "ShapeCapsuleParams_0p0.h"
#include "ShapeCapsuleParams_0p1.h"
#include "ConversionShapeCapsuleParams_0p0_0p1.h"
#include "ShapeBoxParams_0p0.h"
#include "ShapeBoxParams_0p1.h"
#include "ConversionShapeBoxParams_0p0_0p1.h"
#include "ShapeSphereParams_0p0.h"
#include "ShapeSphereParams_0p1.h"
#include "ConversionShapeSphereParams_0p0_0p1.h"
#include "ShapeConvexParams_0p0.h"
#include "ShapeConvexParams_0p1.h"
#include "ConversionShapeConvexParams_0p0_0p1.h"
// AUTO_GENERATED_INCLUDES_END

namespace physx
{
namespace apex
{
namespace legacy
{
// AUTO_GENERATED_OBJECTS_BEGIN
static ShapeCapsuleParams_0p0Factory factory_ShapeCapsuleParams_0p0;
static ShapeCapsuleParams_0p1Factory factory_ShapeCapsuleParams_0p1;
static ShapeBoxParams_0p0Factory factory_ShapeBoxParams_0p0;
static ShapeBoxParams_0p1Factory factory_ShapeBoxParams_0p1;
static ShapeSphereParams_0p0Factory factory_ShapeSphereParams_0p0;
static ShapeSphereParams_0p1Factory factory_ShapeSphereParams_0p1;
static ShapeConvexParams_0p0Factory factory_ShapeConvexParams_0p0;
static ShapeConvexParams_0p1Factory factory_ShapeConvexParams_0p1;
// AUTO_GENERATED_OBJECTS_END

static LegacyClassEntry ModuleFieldBoundaryLegacyObjects[] =
{
	// AUTO_GENERATED_TABLE_BEGIN
	{
		0,
		1,
		&factory_ShapeCapsuleParams_0p0,
		ShapeCapsuleParams_0p0::freeParameterDefinitionTable,
		ConversionShapeCapsuleParams_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_ShapeBoxParams_0p0,
		ShapeBoxParams_0p0::freeParameterDefinitionTable,
		ConversionShapeBoxParams_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_ShapeSphereParams_0p0,
		ShapeSphereParams_0p0::freeParameterDefinitionTable,
		ConversionShapeSphereParams_0p0_0p1::Create,
		0
	},
	{
		0,
		1,
		&factory_ShapeConvexParams_0p0,
		ShapeConvexParams_0p0::freeParameterDefinitionTable,
		ConversionShapeConvexParams_0p0_0p1::Create,
		0
	},
	// AUTO_GENERATED_TABLE_END

	{ 0, 0, 0, 0, 0, 0} // Terminator
};

class ModuleFieldBoundaryLegacy : public ApexLegacyModule, public ApexRWLockable
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	ModuleFieldBoundaryLegacy(NiApexSDK* sdk);

protected:
	void releaseLegacyObjects();

private:

	// Add custom conversions here

};

DEFINE_INSTANTIATE_MODULE(ModuleFieldBoundaryLegacy)

ModuleFieldBoundaryLegacy::ModuleFieldBoundaryLegacy(NiApexSDK* inSdk)
{
	name = "FieldBoundary_Legacy";
	mSdk = inSdk;
	mApiProxy = this;

	// Register legacy stuff

	NxParameterized::Traits* t = mSdk->getParameterizedTraits();
	if (!t)
	{
		return;
	}

	// Register auto-generated objects
	registerLegacyObjects(ModuleFieldBoundaryLegacyObjects);

	// Register custom conversions here
}

void ModuleFieldBoundaryLegacy::releaseLegacyObjects()
{
	//Release legacy stuff

	NxParameterized::Traits* t = mSdk->getParameterizedTraits();
	if (!t)
	{
		return;
	}

	// Unregister auto-generated objects
	unregisterLegacyObjects(ModuleFieldBoundaryLegacyObjects);

	// Unregister custom conversions here
}

}
}
} // end namespace physx::apex
