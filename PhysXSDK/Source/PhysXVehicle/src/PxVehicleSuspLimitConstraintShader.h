/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

#ifndef PX_VEHICLE_SUSP_LIMIT_CONSTRAINT_SHADER_H
#define PX_VEHICLE_SUSP_LIMIT_CONSTRAINT_SHADER_H
/** \addtogroup vehicle
  @{
*/

#include "extensions/PxConstraintExt.h"
#include "PxConstraintDesc.h"
#include "PxConstraint.h"
#include "PxTransform.h"
#include "PsAllocator.h"

#ifndef PX_DOXYGEN
namespace physx
{
#endif

class PxVehicleConstraintShader : public PxConstraintConnector
{
//= ATTENTION! =====================================================================================
// Changing the data layout of this class breaks the binary serialization format.  See comments for 
// PX_BINARY_SERIAL_VERSION.  If a modification is required, please adjust the getBinaryMetaData 
// function.  If the modification is made on a custom branch, please change PX_BINARY_SERIAL_VERSION
// accordingly.
//==================================================================================================
public:

	friend class PxVehicleWheels;

	PxVehicleConstraintShader(PxVehicleWheels* vehicle, PxConstraint* constraint = NULL)
		: mConstraint(constraint),
		  mVehicle(vehicle)
	{
	}
	~PxVehicleConstraintShader()
	{
	}

	static void getBinaryMetaData(PxOutputStream& stream);

	void release()
	{
		if(mConstraint)
		{
			mConstraint->release();
		}
	}

	virtual void			onComShift(PxU32 actor)	{ PX_UNUSED(actor); }

	virtual void			onOriginShift(const PxVec3& shift) { PX_UNUSED(shift); }

	virtual void*			prepareData()	
	{
		return &mData;
	}

	virtual bool			updatePvdProperties(physx::debugger::comm::PvdDataStream& pvdConnection,
		const PxConstraint* c,
		PxPvdUpdateType::Enum updateType) const	 { PX_UNUSED(c); PX_UNUSED(updateType); PX_UNUSED(&pvdConnection); return true;}

	virtual void			onConstraintRelease()
	{
		mVehicle->mOnConstraintReleaseCounter--;
		if(0==mVehicle->mOnConstraintReleaseCounter)
		{
			PX_FREE(mVehicle);
		}
	}

	virtual void*			getExternalReference(PxU32& typeID) { typeID = PxConstraintExtIDs::eVEHICLE_SUSP_LIMIT; return this; }
	virtual PxBase* getSerializable() { return NULL; }


	static PxU32 vehicleSuspLimitConstraintSolverPrep(
		Px1DConstraint* constraints,
		PxVec3& body0WorldOffset,
		PxU32 maxConstraints,
		PxConstraintInvMassScale&,
		const void* constantBlock,
		const PxTransform& bodyAToWorld,
		const PxTransform& bodyBToWorld
		)
	{
		PX_UNUSED(maxConstraints);
		PX_UNUSED(body0WorldOffset);
		PX_UNUSED(bodyBToWorld);
		PX_ASSERT(bodyAToWorld.isValid()); PX_ASSERT(bodyBToWorld.isValid());

		const VehicleConstraintData* data = (const VehicleConstraintData*)constantBlock;
		PxU32 numActive=0;

		//Susp limit constraints.
		for(PxU32 i=0;i<4;i++)
		{
			if(data->mSuspLimitData.mActiveFlags[i])
			{
				Px1DConstraint& p=constraints[numActive];
				p.linear0=bodyAToWorld.q.rotate(data->mSuspLimitData.mDirs[i]);
				p.angular0=bodyAToWorld.q.rotate(data->mSuspLimitData.mCMOffsets[i].cross(data->mSuspLimitData.mDirs[i]));
				p.geometricError=data->mSuspLimitData.mErrors[i];
				p.linear1=PxVec3(0);
				p.angular1=PxVec3(0);
				p.minImpulse=-FLT_MAX;
				p.maxImpulse=0;
				p.velocityTarget=0;		
				numActive++;
			}
		}

		//Sticky tire friction constraints.
		for(PxU32 i=0;i<4;i++)
		{
			if(data->mStickyTireForwardData.mActiveFlags[i])
			{
				Px1DConstraint& p=constraints[numActive];
				p.linear0=data->mStickyTireForwardData.mDirs[i];
				p.angular0=data->mStickyTireForwardData.mCMOffsets[i].cross(data->mStickyTireForwardData.mDirs[i]);
				p.geometricError=0.0f;
				p.linear1=PxVec3(0);
				p.angular1=PxVec3(0);
				p.minImpulse=-FLT_MAX;
				p.maxImpulse=FLT_MAX;
				p.velocityTarget=data->mStickyTireForwardData.mTargetSpeeds[i];	
				p.mods.spring.damping = 1000.0f;
				p.flags = Px1DConstraintFlag::eSPRING | Px1DConstraintFlag::eACCELERATION_SPRING;
				numActive++;
			}
		}

		//Sticky tire friction constraints.
		for(PxU32 i=0;i<4;i++)
		{
			if(data->mStickyTireSideData.mActiveFlags[i])
			{
				Px1DConstraint& p=constraints[numActive];
				p.linear0=data->mStickyTireSideData.mDirs[i];
				p.angular0=data->mStickyTireSideData.mCMOffsets[i].cross(data->mStickyTireSideData.mDirs[i]);
				p.geometricError=0.0f;
				p.linear1=PxVec3(0);
				p.angular1=PxVec3(0);
				p.minImpulse=-FLT_MAX;
				p.maxImpulse=FLT_MAX;
				p.velocityTarget=data->mStickyTireSideData.mTargetSpeeds[i];	
				p.mods.spring.damping = 1000.0f;
				p.flags = Px1DConstraintFlag::eSPRING | Px1DConstraintFlag::eACCELERATION_SPRING;
				numActive++;
			}
		}


		return numActive;
	}

	static void visualiseConstraint(PxConstraintVisualizer &viz,
		const void* constantBlock,
		const PxTransform& body0Transform,
		const PxTransform& body1Transform,
		PxU32 flags){ PX_UNUSED(&viz); PX_UNUSED(constantBlock); PX_UNUSED(body0Transform); 
					  PX_UNUSED(body1Transform); PX_UNUSED(flags); 
					  PX_ASSERT(body0Transform.isValid()); PX_ASSERT(body1Transform.isValid()); }

public:

	struct SuspLimitConstraintData
	{
		PxVec3 mCMOffsets[4];
		PxVec3 mDirs[4];
		PxReal mErrors[4];
		bool mActiveFlags[4];
	};
	struct StickyTireConstraintData
	{
		PxVec3 mCMOffsets[4];
		PxVec3 mDirs[4];
		PxReal mTargetSpeeds[4];
		bool mActiveFlags[4];
	};

	struct VehicleConstraintData
	{
		SuspLimitConstraintData mSuspLimitData;
		StickyTireConstraintData mStickyTireForwardData;
		StickyTireConstraintData mStickyTireSideData;
	};
	VehicleConstraintData mData;

	PxConstraint* mConstraint;
	
	PX_INLINE void setPxConstraint(PxConstraint* pxConstraint)
	{
		mConstraint = pxConstraint;
	}

	PX_INLINE PxConstraint* getPxConstraint()
	{
		return mConstraint;
	}

	PxConstraintConnector* getConnector()
	{
		return this;
	}

private:

	PxVehicleWheels* mVehicle;

#if !defined(PX_X64)
	PxU32 mPad[2];
#else
	PxU32 mPad[1];
#endif
};
PX_COMPILE_TIME_ASSERT(0==(sizeof(PxVehicleConstraintShader)& 0x0f));


/**
\brief Default implementation of PxVehicleComputeTireForce
@see PxVehicleComputeTireForce, PxVehicleTireForceCalculator
*/
void PxVehicleComputeTireForceDefault
 (const void* shaderData, 
 const PxF32 tireFriction,
 const PxF32 longSlip, const PxF32 latSlip, const PxF32 camber,
 const PxF32 wheelOmega, const PxF32 wheelRadius, const PxF32 recipWheelRadius,
 const PxF32 restTireLoad, const PxF32 normalisedTireLoad, const PxF32 tireLoad,
 const PxF32 gravity, const PxF32 recipGravity,
 PxF32& wheelTorque, PxF32& tireLongForceMag, PxF32& tireLatForceMag, PxF32& tireAlignMoment);


/**
\brief Structure containing shader data for each tire of a vehicle and a shader function that computes individual tire forces 
*/
class PxVehicleTireForceCalculator
{
public:

	PxVehicleTireForceCalculator()
		: mShader(PxVehicleComputeTireForceDefault)
	{
	}

	/**
	\brief Array of shader data - one data entry per tire.
	Default values are pointers to PxVehicleTireData (stored in PxVehicleWheelsSimData) and are set in PxVehicleDriveTank::setup or PxVehicleDrive4W::setup
	@see PxVehicleComputeTireForce, PxVehicleComputeTireForceDefault, PxVehicleWheelsSimData, PxVehicleDriveTank::setup, PxVehicleDrive4W::setup
	*/
	const void** mShaderData;

	/**
	\brief Shader function.
	Default value is PxVehicleComputeTireForceDefault and is set in  PxVehicleDriveTank::setup or PxVehicleDrive4W::setup
	@see PxVehicleComputeTireForce, PxVehicleComputeTireForceDefault, PxVehicleWheelsSimData, PxVehicleDriveTank::setup, PxVehicleDrive4W::setup
	*/
	PxVehicleComputeTireForce mShader;

#ifndef PX_X64
	PxU32 mPad[2];
#endif
};

PX_COMPILE_TIME_ASSERT(0==(sizeof(PxVehicleTireForceCalculator) & 15));


#ifndef PX_DOXYGEN
} // namespace physx
#endif


/** @} */
#endif //PX_VEHICLE_SUSP_LIMIT_CONSTRAINT_SHADER_H
