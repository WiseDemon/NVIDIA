/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NX_MODULE_FLUID_IOS_H
#define NX_MODULE_FLUID_IOS_H

#include "NxApex.h"
#include <limits.h>

class NxCompartment;

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

class NxFluidIosAsset;
class NxFluidIosAssetAuthoring;

/**
\brief APEX Particles module descriptor. Used for initializing the module.
*/
class NxModuleFluidIosDesc : public NxApexDesc
{
public:

	/**
	\brief constructor sets to default.
	*/
	PX_INLINE NxModuleFluidIosDesc() : NxApexDesc()
	{
		init();
	}

	/**
	\brief sets members to default values.
	*/
	PX_INLINE void setToDefault()
	{
		NxApexDesc::setToDefault();
		init();
	}

	/**
	\brief checks if this is a valid descriptor.
	*/
	PX_INLINE bool isValid() const
	{
		bool retVal = NxApexDesc::isValid();
		return retVal;
	}

private:

	PX_INLINE void init()
	{
	}
};

/**
\brief APEX FluidIOS module. A particle system based upon PhysX SDK 2.8 particles
*/
class NxModuleFluidIos : public NxModule
{
protected:
	virtual									~NxModuleFluidIos() {}

public:
	/// Initializer. \sa NxModuleFluidIosDesc
	//virtual void							init( const NxModuleFluidIosDesc & moduleFluidIosDesc ) = 0;
	//virtual void init( ::NxParameterized::Interface &desc ) = 0;
	/**
	\brief Sets the compartment that will be used for non-SPH calculations in the given ApexScene.

	\sa NxCompartment
	*/
	virtual void                            setCompartment(const NxApexScene&, NxCompartment& comp) = 0;
	/**
	\brief Gets the compartment that is used for non-SPH calculations in the given ApexScene.

	\sa NxCompartment
	*/
	virtual const NxCompartment*            getCompartment(const NxApexScene&) const = 0;

	/**
	\brief Sets the compartment that will be used for SPH calculations in the given ApexScene.

	If none provided, the compartment for non-SPH calculations will br used
	\sa NxCompartment
	*/
	virtual void                            setSPHCompartment(const NxApexScene&, NxCompartment& comp) = 0;
	/**
	\brief Gets the compartment that is used for SPH calculations in the given ApexScene.

	\sa NxCompartment
	*/
	virtual const NxCompartment*            getSPHCompartment(const NxApexScene&) const = 0;
};


PX_POP_PACK

}
} // namespace physx::apex

#endif // NX_MODULE_FLUID_IOS_H
