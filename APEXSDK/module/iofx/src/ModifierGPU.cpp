/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "Modifier.h"

#if defined(APEX_CUDA_SUPPORT)

#include "ApexCudaWrapper.h"

#include "ModifierData.h"

namespace physx
{
namespace apex
{

#define MODIFIER_DECL
#define CURVE_TYPE physx::apex::iofx::Curve
#define EVAL_CURVE(curve, value) 0
#define PARAMS_NAME(name) name ## ParamsGPU

#include "ModifierSrc.h"

#undef MODIFIER_DECL
#undef CURVE_TYPE
#undef EVAL_CURVE
//#undef PARAMS_NAME

namespace iofx
{

class ModifierParamsMapperGPU_Adapter
{
private:
	ModifierParamsMapperGPU_Adapter& operator=(const ModifierParamsMapperGPU_Adapter&);

	ModifierParamsMapperGPU& _mapper;
	InplaceStorage& _storage;
	physx::PxU8* _params;

public:
	ModifierParamsMapperGPU_Adapter(ModifierParamsMapperGPU& mapper)
		: _mapper(mapper), _storage(mapper.getStorage()), _params(0) {}

	PX_INLINE InplaceStorage& getStorage()
	{
		return _storage;
	}

	PX_INLINE void beginParams(void* params, size_t , size_t , physx::PxU32)
	{
		_params = (physx::PxU8*)params;
	}
	PX_INLINE void endParams()
	{
		_params = 0;
	}

	template <typename T>
	PX_INLINE void mapValue(size_t offset, T value)
	{
		PX_ASSERT(_params != 0);
		*(T*)(_params + offset) = value;
	}

	PX_INLINE void mapCurve(size_t offset, const NxCurve* nxCurve)
	{
		PX_ASSERT(_params != 0);
		Curve& curve = *(Curve*)(_params + offset);

		physx::PxU32 numPoints;
		const NxVec2R* nxPoints = nxCurve->getControlPoints(numPoints);

		curve.resize(_storage, numPoints);
		for (physx::PxU32 i = 0; i < numPoints; ++i)
		{
			const NxVec2R& nxPoint = nxPoints[i];
			curve.setPoint(_storage, CurvePoint(nxPoint.x, nxPoint.y), i);
		}
	}
};

#define _MODIFIER(name) \
	void name ## Modifier :: mapParamsGPU(ModifierParamsMapperGPU& mapper) const \
	{ \
		ModifierParamsMapperGPU_Adapter adapter(mapper); \
		InplaceHandle< PARAMS_NAME(name) > paramsHandle; \
		paramsHandle.alloc( adapter.getStorage() ); \
		PARAMS_NAME(name) params; \
		mapParams( adapter, &params ); \
		paramsHandle.update( adapter.getStorage(), params ); \
		mapper.onParams( paramsHandle, PARAMS_NAME(name)::RANDOM_COUNT ); \
	} \
	 
#include "ModifierList.h"

}
}
} // namespace physx::apex

#endif
