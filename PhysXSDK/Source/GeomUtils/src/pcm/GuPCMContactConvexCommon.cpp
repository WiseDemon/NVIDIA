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

#include "GuVecTriangle.h"
#include "GuPCMContactConvexCommon.h"
#include "GuConvexEdgeFlags.h"
#include "GuBarycentricCoordinates.h"
#include "GuFeatureCode.h"

namespace physx
{

namespace Gu
{

/*
	This function adds the newly created manifold contacts to a new patch or existing patches 
*/
void PCMConvexVsMeshContactGeneration::addContactsToPatch(const Ps::aos::Vec3VArg patchNormal, const PxU32 previousNumContacts, const PxU32 _numContacts)
{
	using namespace Ps::aos;
	const Vec3V patchNormalInTriangle = mMeshToConvex.rotateInv(patchNormal);
	PxU32 numContacts = _numContacts;

	if(numContacts > 4)
	{
		//if the current created manifold contacts are more than 4 points, we will reduce the total numContacts to 4
		numContacts = Gu::SinglePersistentContactManifold::reduceContacts(&mManifoldContacts[previousNumContacts], numContacts);
		mNumContacts = previousNumContacts + numContacts;  
	}   

	//calculate the maxPen and transform the patch normal and localPointB into mesh's local space
	FloatV maxPen = FMax();
	for(PxU32 i = previousNumContacts; i<mNumContacts; ++i)
	{
		const FloatV pen = V4GetW(mManifoldContacts[i].mLocalNormalPen);
		mManifoldContacts[i].mLocalNormalPen = V4SetW(patchNormalInTriangle, pen);
		mManifoldContacts[i].mLocalPointB = mMeshToConvex.transformInv(mManifoldContacts[i].mLocalPointB);
		maxPen = FMin(maxPen, pen);
	}


	//get rid of duplicate manifold contacts for the remaining contacts
	for(PxU32 i = previousNumContacts; i<mNumContacts; ++i)
	{
		for(PxU32 j=i+1; j<mNumContacts; ++j)
		{
			Vec3V dif = V3Sub(mManifoldContacts[j].mLocalPointB, mManifoldContacts[i].mLocalPointB);
			FloatV d = V3Dot(dif, dif);
			if(FAllGrtr(mSqReplaceBreakingThreshold, d))
			{
				mManifoldContacts[j] = mManifoldContacts[mNumContacts-1];
				mNumContacts--;
				j--;
			}
		}
	}

	//Based on the patch normal and add the newly avaiable manifold points to the corresponding patch
	addManifoldPointToPatch(patchNormalInTriangle, maxPen, previousNumContacts);
	
	PX_ASSERT(mNumContactPatch <PCM_MAX_CONTACTPATCH_SIZE);
	if(mNumContacts >= 16)
	{
		PX_ASSERT(mNumContacts <= 64);
		processContacts(GU_SINGLE_MANIFOLD_CACHE_SIZE);
	}
}
void PCMConvexVsMeshContactGeneration::generateLastContacts()
{
	using namespace Ps::aos;
	// Process delayed contacts
	PxU32 nbEntries = mDeferredContacts.GetNbEntries();

	if(nbEntries)
	{
		nbEntries /= sizeof(PCMDeferredPolyData)/sizeof(PxU32);

		const PCMDeferredPolyData* PX_RESTRICT cd = (const PCMDeferredPolyData*)mDeferredContacts.GetEntries();
		for(PxU32 i=0;i<nbEntries;i++)
		{
			const PCMDeferredPolyData& currentContact = cd[i];  

			const PxU32 ref0 = currentContact.mInds[0];
			const PxU32 ref1 = currentContact.mInds[1];  
			const PxU32 ref2 = currentContact.mInds[2];

			PxU8 triFlags = currentContact.triFlags;

			
			bool needsProcessing =  (((triFlags & ETD_CONVEX_EDGE_01) != 0 || mEdgeCache.get(CachedEdge(ref0, ref1)) == NULL)) && 
									(((triFlags & ETD_CONVEX_EDGE_12) != 0 || mEdgeCache.get(CachedEdge(ref1, ref2)) == NULL)) && 
									(((triFlags & ETD_CONVEX_EDGE_20) != 0 || mEdgeCache.get(CachedEdge(ref2, ref0)) == NULL));
			

			if(needsProcessing)
			{

				Gu::TriangleV localTriangle(currentContact.mVerts);
				Vec3V patchNormal;
				const PxU32 previousNumContacts = mNumContacts;
				//the localTriangle is in the convex space
				//Generate contacts - we didn't generate contacts with any neighbours
				generatePolyDataContactManifold(localTriangle, currentContact.mFeatureIndex, currentContact.mTriangleIndex, triFlags, mManifoldContacts, mNumContacts, mContactDist, patchNormal);

				PxU32 numContacts = mNumContacts - previousNumContacts;

				if(numContacts > 0)
				{
					addContactsToPatch(patchNormal, previousNumContacts, numContacts);
				}
				
			}
		}
	}

}

bool PCMConvexVsMeshContactGeneration::processTriangle(const PxVec3* verts, PxU32 triangleIndex, PxU8 triFlags, const PxU32* vertInds)
{
	using namespace Ps::aos;


	const Mat33V identity =  M33Identity();
	const FloatV zero = FZero();

	const Vec3V v0 = V3LoadU(verts[0]);
	const Vec3V v1 = V3LoadU(verts[1]);
	const Vec3V v2 = V3LoadU(verts[2]);

	const Vec3V v10 = V3Sub(v1, v0);
	const Vec3V v20 = V3Sub(v2, v0);

	const Vec3V n = V3Normalize(V3Cross(v10, v20));//(p1 - p0).cross(p2 - p0).getNormalized();
	const FloatV d = V3Dot(v0, n);//d = -p0.dot(n);

	const FloatV dist = FSub(V3Dot(mHullCenterMesh, n), d);//p.dot(n) + d;
	
	// Backface culling
	if(FAllGrtr(zero, dist))
		return false;


	//tranform verts into the box local space
	const Vec3V locV0 = mMeshToConvex.transform(v0);
	const Vec3V locV1 = mMeshToConvex.transform(v1);
	const Vec3V locV2 = mMeshToConvex.transform(v2);

	Gu::TriangleV localTriangle(locV0, locV1, locV2);

	{

		SupportLocalImpl<Gu::TriangleV> localTriMap(localTriangle, mConvexTransform, identity, identity, true);

		const PxU32 previousNumContacts = mNumContacts;
		Vec3V patchNormal;

		generateTriangleFullContactManifold(localTriangle, triangleIndex, vertInds, triFlags, mPolyData, &localTriMap, mPolyMap, mManifoldContacts, mNumContacts, mContactDist, patchNormal);
		
		
		PxU32 numContacts = mNumContacts - previousNumContacts;

		if(numContacts > 0)
		{
			bool inActiveEdge0 = (triFlags & ETD_CONVEX_EDGE_01) == 0;
			bool inActiveEdge1 = (triFlags & ETD_CONVEX_EDGE_12) == 0;
			bool inActiveEdge2 = (triFlags & ETD_CONVEX_EDGE_20) == 0;

			if(inActiveEdge0)
				mEdgeCache.addData(CachedEdge(vertInds[0], vertInds[1]));
			if(inActiveEdge1)
				mEdgeCache.addData(CachedEdge(vertInds[1], vertInds[2]));
			if(inActiveEdge2)
				mEdgeCache.addData(CachedEdge(vertInds[2], vertInds[0]));

			addContactsToPatch(patchNormal, previousNumContacts, numContacts);
		}
	}

	

	return true;
}


bool PCMConvexVsMeshContactGeneration::processTriangle(const Gu::PolygonalData& polyData, SupportLocal* polyMap, const PxVec3* verts, const PxU32 triangleIndex, PxU8 triFlags,const Ps::aos::FloatVArg inflation, const bool isDoubleSided, 
													   const Ps::aos::PsTransformV& convexTransform, const Ps::aos::PsMatTransformV& meshToConvex, Gu::MeshPersistentContact* manifoldContacts, PxU32& numContacts)
{
	using namespace Ps::aos;


	const Mat33V identity =  M33Identity();
	const FloatV zero = FZero();

	const Vec3V v0 = V3LoadU(verts[0]);
	const Vec3V v1 = V3LoadU(verts[1]);
	const Vec3V v2 = V3LoadU(verts[2]);

	//tranform verts into the box local space
	const Vec3V locV0 = meshToConvex.transform(v0);
	const Vec3V locV1 = meshToConvex.transform(v1);
	const Vec3V locV2 = meshToConvex.transform(v2);

	const Vec3V v10 = V3Sub(locV1, locV0);
	const Vec3V v20 = V3Sub(locV2, locV0);

	const Vec3V n = V3Normalize(V3Cross(v10, v20));//(p1 - p0).cross(p2 - p0).getNormalized();
	const FloatV d = V3Dot(locV0, n);//d = -p0.dot(n);

	const FloatV dist = FSub(V3Dot(polyMap->shapeSpaceCenterOfMass, n), d);//p.dot(n) + d;
	
	// Backface culling
	const bool culled = !isDoubleSided && (FAllGrtr(zero, dist));
	if(culled)
		return false;


	Gu::TriangleV localTriangle(locV0, locV1, locV2);

	SupportLocalImpl<Gu::TriangleV> localTriMap(localTriangle, convexTransform, identity, identity, true);

	Vec3V patchNormal;

	generateTriangleFullContactManifold(localTriangle, triangleIndex, triFlags, polyData, &localTriMap, polyMap, manifoldContacts, numContacts, inflation, patchNormal);
	
	return true;
}

PX_FORCE_INLINE Ps::aos::Vec4V pcmDistanceSegmentSegmentSquared4(		const Ps::aos::Vec3VArg p, const Ps::aos::Vec3VArg d0, 
														const Ps::aos::Vec3VArg p02, const Ps::aos::Vec3VArg d02, 
                                                        const Ps::aos::Vec3VArg p12, const Ps::aos::Vec3VArg d12, 
														const Ps::aos::Vec3VArg p22, const Ps::aos::Vec3VArg d22,
                                                        const Ps::aos::Vec3VArg p32, const Ps::aos::Vec3VArg d32,
                                                        Ps::aos::Vec4V& s, Ps::aos::Vec4V& t)
{
      using namespace Ps::aos;
      const Vec4V zero = V4Zero();
      const Vec4V one = V4One();
      const Vec4V eps = V4Eps();
	  const Vec4V half = V4Splat(FHalf());
 
	  const Vec4V d0X = V4Splat(V3GetX(d0));
	  const Vec4V d0Y = V4Splat(V3GetY(d0));
	  const Vec4V d0Z = V4Splat(V3GetZ(d0));
	  const Vec4V pX  = V4Splat(V3GetX(p));
	  const Vec4V pY  = V4Splat(V3GetY(p));
	  const Vec4V pZ  = V4Splat(V3GetZ(p));

	  Vec4V d024 = Vec4V_From_Vec3V(d02);
	  Vec4V d124 = Vec4V_From_Vec3V(d12);
	  Vec4V d224 = Vec4V_From_Vec3V(d22);
	  Vec4V d324 = Vec4V_From_Vec3V(d32);

	  Vec4V p024 = Vec4V_From_Vec3V(p02);
	  Vec4V p124 = Vec4V_From_Vec3V(p12);
	  Vec4V p224 = Vec4V_From_Vec3V(p22);
	  Vec4V p324 = Vec4V_From_Vec3V(p32);

	  Vec4V d0123X, d0123Y, d0123Z;
	  Vec4V p0123X, p0123Y, p0123Z;

	  PX_TRANSPOSE_44_34(d024, d124, d224, d324, d0123X, d0123Y, d0123Z);
	  PX_TRANSPOSE_44_34(p024, p124, p224, p324, p0123X, p0123Y, p0123Z);

	  const Vec4V rX = V4Sub(pX, p0123X);
	  const Vec4V rY = V4Sub(pY, p0123Y);
	  const Vec4V rZ = V4Sub(pZ, p0123Z);


	  //TODO - store this in a transposed state and avoid so many dot products?

	  const FloatV dd = V3Dot(d0, d0);

	  const Vec4V e = V4MulAdd(d0123Z, d0123Z, V4MulAdd(d0123X, d0123X, V4Mul(d0123Y, d0123Y)));
	  const Vec4V b = V4MulAdd(d0Z, d0123Z, V4MulAdd(d0X, d0123X, V4Mul(d0Y, d0123Y)));
	  const Vec4V c = V4MulAdd(d0Z, rZ, V4MulAdd(d0X, rX, V4Mul(d0Y, rY)));
	  const Vec4V f = V4MulAdd(d0123Z, rZ, V4MulAdd(d0123X, rX, V4Mul(d0123Y, rY))); 
	 

	  const Vec4V a(V4Splat(dd));

      const Vec4V aRecip(V4Recip(a));
      const Vec4V eRecip(V4Recip(e));


      //if segments not parallell, compute closest point on two segments and clamp to segment1
      const Vec4V denom = V4Sub(V4Mul(a, e), V4Mul(b, b));
      const Vec4V temp = V4Sub(V4Mul(b, f), V4Mul(c, e));
      const Vec4V s0 = V4Clamp(V4Div(temp, denom), zero, one);
      
  
	  //test whether segments are parallel
	  const BoolV con2 = V4IsGrtrOrEq(eps, denom);     
	  const Vec4V sTmp = V4Sel(con2, half, s0);
      
      //compute point on segment2 closest to segment1
      const Vec4V tTmp = V4Mul(V4Add(V4Mul(b, sTmp), f), eRecip);

      //if t is in [zero, one], done. otherwise clamp t
      const Vec4V t2 = V4Clamp(tTmp, zero, one);

      //recompute s for the new value
      const Vec4V comp = V4Mul(V4Sub(V4Mul(b,t2), c), aRecip);
      const Vec4V s2 = V4Clamp(comp, zero, one);

	  s = s2;
	  t = t2;

	  const Vec4V closest1X = V4MulAdd(d0X, s2, pX);
	  const Vec4V closest1Y = V4MulAdd(d0Y, s2, pY);
	  const Vec4V closest1Z = V4MulAdd(d0Z, s2, pZ);

	  const Vec4V closest2X = V4MulAdd(d0123X, t2, p0123X);
	  const Vec4V closest2Y = V4MulAdd(d0123Y, t2, p0123Y);
	  const Vec4V closest2Z = V4MulAdd(d0123Z, t2, p0123Z);

	  const Vec4V vvX = V4Sub(closest1X, closest2X);
	  const Vec4V vvY = V4Sub(closest1Y, closest2Y);
	  const Vec4V vvZ = V4Sub(closest1Z, closest2Z);

	  const Vec4V vd = V4MulAdd(vvX, vvX, V4MulAdd(vvY, vvY, V4Mul(vvZ, vvZ)));

	  return vd;
}

static Ps::aos::FloatV pcmDistancePointTriangleSquared(	const Ps::aos::Vec3VArg p, 
														const Ps::aos::Vec3VArg a, 
														const Ps::aos::Vec3VArg b, 
														const Ps::aos::Vec3VArg c,
														const PxU8 triFlags,
														Ps::aos::Vec3V& closestP,
														bool& generateContact)
{
	using namespace Ps::aos;

	const BoolV bTrue = BTTTT();
	const FloatV zero = FZero();
	const Vec3V ab = V3Sub(b, a);
	const Vec3V ac = V3Sub(c, a);
	const Vec3V bc = V3Sub(c, b);
	const Vec3V ap = V3Sub(p, a);
	const Vec3V bp = V3Sub(p, b);
	const Vec3V cp = V3Sub(p, c);

	const FloatV d1 = V3Dot(ab, ap); //  snom
	const FloatV d2 = V3Dot(ac, ap); //  tnom
	const FloatV d3 = V3Dot(ab, bp); // -sdenom
	const FloatV d4 = V3Dot(ac, bp); //  unom = d4 - d3
	const FloatV d5 = V3Dot(ab, cp); //  udenom = d5 - d6
	const FloatV d6 = V3Dot(ac, cp); // -tdenom
	const FloatV unom = FSub(d4, d3);
	const FloatV udenom = FSub(d5, d6);
	
	//check if p in vertex region outside a
	const BoolV con00 = FIsGrtr(zero, d1); // snom <= 0
	const BoolV con01 = FIsGrtr(zero, d2); // tnom <= 0
	const BoolV con0 = BAnd(con00, con01); // vertex region a

	if(BAllEq(con0, bTrue))
	{
		//Vertex 0
		generateContact = (triFlags & Gu::ETD_CONVEX_EDGE_01) || (triFlags & Gu::ETD_CONVEX_EDGE_20);
		closestP = a;
		return V3Dot(ap, ap);
	}

	//check if p in vertex region outside b
	const BoolV con10 = FIsGrtrOrEq(d3, zero);
	const BoolV con11 = FIsGrtrOrEq(d3, d4);
	const BoolV con1 = BAnd(con10, con11); // vertex region b
	if(BAllEq(con1, bTrue))
	{
		//Vertex 1
		generateContact = (triFlags & Gu::ETD_CONVEX_EDGE_01) || (triFlags & Gu::ETD_CONVEX_EDGE_12);
		closestP = b;
		return V3Dot(bp, bp);
	}

	//check if p in vertex region outside c
	const BoolV con20 = FIsGrtrOrEq(d6, zero);
	const BoolV con21 = FIsGrtrOrEq(d6, d5); 
	const BoolV con2 = BAnd(con20, con21); // vertex region c
	if(BAllEq(con2, bTrue))
	{
		//Vertex 2
		generateContact = (triFlags & Gu::ETD_CONVEX_EDGE_12) || (triFlags & Gu::ETD_CONVEX_EDGE_20);
		closestP = c;
		return V3Dot(cp, cp);
	}

	//check if p in edge region of AB
	const FloatV vc = FSub(FMul(d1, d4), FMul(d3, d2));
	
	const BoolV con30 = FIsGrtr(zero, vc);
	const BoolV con31 = FIsGrtrOrEq(d1, zero);
	const BoolV con32 = FIsGrtr(zero, d3);
	const BoolV con3 = BAnd(con30, BAnd(con31, con32));
	if(BAllEq(con3, bTrue))
	{
		// Edge 01
		generateContact = (triFlags & Gu::ETD_CONVEX_EDGE_01) != 0;
		const FloatV sScale = FDiv(d1, FSub(d1, d3));
		const Vec3V closest3 = V3ScaleAdd(ab, sScale, a);//V3Add(a, V3Scale(ab, sScale));
		const Vec3V vv = V3Sub(p, closest3);
		closestP = closest3;
		return V3Dot(vv, vv);
	}

	//check if p in edge region of BC
	const FloatV va = FSub(FMul(d3, d6),FMul(d5, d4));
	const BoolV con40 = FIsGrtr(zero, va);
	const BoolV con41 = FIsGrtrOrEq(d4, d3);
	const BoolV con42 = FIsGrtrOrEq(d5, d6);
	const BoolV con4 = BAnd(con40, BAnd(con41, con42)); 
	if(BAllEq(con4, bTrue))
	{
		// Edge 12
		generateContact = (triFlags & Gu::ETD_CONVEX_EDGE_12) != 0;
		const FloatV uScale = FDiv(unom, FAdd(unom, udenom));
		const Vec3V closest4 = V3ScaleAdd(bc, uScale, b);//V3Add(b, V3Scale(bc, uScale));
		const Vec3V vv = V3Sub(p, closest4);
		closestP = closest4;
		return V3Dot(vv, vv);
	}

	//check if p in edge region of AC
	const FloatV vb = FSub(FMul(d5, d2), FMul(d1, d6));
	const BoolV con50 = FIsGrtr(zero, vb);
	const BoolV con51 = FIsGrtrOrEq(d2, zero);
	const BoolV con52 = FIsGrtr(zero, d6);
	const BoolV con5 = BAnd(con50, BAnd(con51, con52));
	if(BAllEq(con5, bTrue))
	{
		//Edge 20
		generateContact = (triFlags & Gu::ETD_CONVEX_EDGE_20) != 0;
		const FloatV tScale = FDiv(d2, FSub(d2, d6));
		const Vec3V closest5 = V3ScaleAdd(ac, tScale, a);//V3Add(a, V3Scale(ac, tScale));
		const Vec3V vv = V3Sub(p, closest5);
		closestP = closest5;
		return V3Dot(vv, vv);
	}

	generateContact = true;
	//P must project inside face region. Compute Q using Barycentric coordinates
	const FloatV denom = FRecip(FAdd(va, FAdd(vb, vc)));
	const FloatV t = FMul(vb, denom);
	const FloatV w = FMul(vc, denom);
	const Vec3V cCom = V3Scale(ac, w);
	const Vec3V closest6 = V3Add(a, V3ScaleAdd(ab, t, cCom));
	closestP = closest6;
	const Vec3V vv = V3Sub(p, closest6);

	return V3Dot(vv, vv);
}

bool Gu::PCMSphereVsMeshContactGeneration::processTriangle(const PxVec3* verts, PxU32 triangleIndex, PxU8 triFlags, const PxU32* vertInds)
{
	PX_UNUSED(triangleIndex);
	PX_UNUSED(vertInds);

	using namespace Ps::aos;

	const FloatV zero = FZero();
	const Vec3V zeroV = V3Zero();

	const Vec3V v0 = V3LoadU(verts[0]);
	const Vec3V v1 = V3LoadU(verts[1]);
	const Vec3V v2 = V3LoadU(verts[2]);

	const Vec3V v10 = V3Sub(v1, v0);
	const Vec3V v20 = V3Sub(v2, v0);

	const Vec3V n = V3Normalize(V3Cross(v10, v20));//(p1 - p0).cross(p2 - p0).getNormalized();
	const FloatV d = V3Dot(v0, n);//d = -p0.dot(n);

	const FloatV dist0 = FSub(V3Dot(mSphereCenter, n), d);//p.dot(n) + d;
	
	// Backface culling
	if(FAllGrtr(zero, dist0))
		return false;


	const FloatV tolerance = FLoad(0.996);//around 5 degree
	//FloatV u, v;
	Vec3V closestP;
	//mShereCenter will be in the local space of the triangle mesh
	//const FloatV sqDist = Gu::distancePointTriangleSquared(mSphereCenter, v0, v1, v2, u, v, closestP);
	bool generateContact = false;
	const FloatV sqDist = pcmDistancePointTriangleSquared(mSphereCenter, v0, v1, v2, triFlags, closestP, generateContact);

	//sphere overlap with triangles
	const Vec3V patchNormal = V3Normalize(V3Sub(mSphereCenter, closestP));
	const FloatV cosTheta = V3Dot(patchNormal, n);
	

	if(FAllGrtr(mSqInflatedSphereRadius, sqDist) && (generateContact || FAllGrtr(cosTheta, tolerance)))
	{
		const FloatV dist = FSqrt(sqDist);
	
		PX_ASSERT(mNumContactPatch <PCM_MAX_CONTACTPATCH_SIZE);

		bool foundPatch = false;
		if(mNumContactPatch > 0)
		{
			if(FAllGrtr(V3Dot(mContactPatch[mNumContactPatch-1].mPatchNormal, patchNormal), mAcceptanceEpsilon))
			{
				PCMContactPatch& patch = mContactPatch[mNumContactPatch-1];

				PX_ASSERT((patch.mEndIndex - patch.mStartIndex) == 1);
				
				if(FAllGrtr(patch.mPatchMaxPen, dist))
				{
					//overwrite the old contact
					mManifoldContacts[patch.mStartIndex].mLocalPointA = zeroV;//in sphere's space
					mManifoldContacts[patch.mStartIndex].mLocalPointB = closestP;
					mManifoldContacts[patch.mStartIndex].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(patchNormal), dist);
					mManifoldContacts[patch.mStartIndex].mFaceIndex = triangleIndex;
					patch.mPatchMaxPen = dist;
				}
				
				foundPatch = true;
			}
		}
		if(!foundPatch)
		{
			mManifoldContacts[mNumContacts].mLocalPointA = zeroV;//in sphere's space
			mManifoldContacts[mNumContacts].mLocalPointB = closestP;
			mManifoldContacts[mNumContacts].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(patchNormal), dist);
			mManifoldContacts[mNumContacts++].mFaceIndex = triangleIndex;
			
			mContactPatch[mNumContactPatch].mStartIndex = mNumContacts - 1;
			mContactPatch[mNumContactPatch].mEndIndex = mNumContacts;
			mContactPatch[mNumContactPatch].mPatchMaxPen = dist;
			mContactPatch[mNumContactPatch++].mPatchNormal = patchNormal;
		}

		PX_ASSERT(mNumContactPatch <PCM_MAX_CONTACTPATCH_SIZE);

		if(mNumContacts >= 16)
		{
			PX_ASSERT(mNumContacts <= 64);
			processContacts(GU_SPHERE_MANIFOLD_CACHE_SIZE);
		}
		
	}
	return true;
}


void Gu::PCMCapsuleVsMeshContactGeneration::generateEE(const Ps::aos::Vec3VArg p, const Ps::aos::Vec3VArg q, const Ps::aos::FloatVArg sqInflatedRadius,
													   const Ps::aos::Vec3VArg normal, const PxU32 triangleIndex, const Ps::aos::Vec3VArg a, const Ps::aos::Vec3VArg b,
														Gu::MeshPersistentContact* manifoldContacts, PxU32& numContacts)
{
	using namespace Ps::aos;
	const FloatV zero = FZero();
	const Vec3V ab = V3Sub(b, a);
	const Vec3V n = V3Cross(ab, normal);
	const FloatV d = V3Dot(n, a);
	const FloatV np = V3Dot(n, p);
	const FloatV nq = V3Dot(n,q);
	const FloatV signP = FSub(np, d);
	const FloatV signQ = FSub(nq, d);
	const FloatV temp = FMul(signP, signQ);
	if(FAllGrtr(temp, zero)) return;//If both points in the same side as the plane, no intersect points
	
	// if colliding edge (p3,p4) and plane are parallel return no collision
	const Vec3V pq = V3Sub(q, p);
	const FloatV npq= V3Dot(n, pq); 
	if(FAllEq(npq, zero))	return;

	const FloatV one = FOne();
	const BoolV bFalse = BFFFF();
	//calculate the intersect point in the segment pq
	const FloatV segTValue = FDiv(FSub(d, np), npq);
	const Vec3V localPointA = V3ScaleAdd(pq, segTValue, p);

	//calculate a normal perpendicular to ray localPointA + normal, 2D segment segment intersection
	const Vec3V perNormal = V3Cross(normal, pq);
	const Vec3V ap = V3Sub(localPointA, a);
	const FloatV nom = V3Dot(perNormal, ap);
	const FloatV denom = V3Dot(perNormal, ab);

	//const FloatV tValue = FClamp(FDiv(nom, denom), zero, FOne());
	const FloatV tValue = FDiv(nom, denom);
	const BoolV con = BAnd(FIsGrtrOrEq(one, tValue), FIsGrtrOrEq(tValue, zero));
	if(BAllEq(con, bFalse))
		return;

	//const Vec3V localPointB = V3ScaleAdd(ab, tValue, a); v = V3Sub(localPointA, localPointB); v =  V3NegScaleSub(ab, tValue, tap)
	const Vec3V v = V3NegScaleSub(ab, tValue, ap);
	const FloatV sqDist = V3Dot(v, v);
	
	if(FAllGrtr(sqInflatedRadius, sqDist))
	{
		
		const Vec3V localPointB = V3Sub(localPointA, v);
		const FloatV signedDist = V3Dot(v, normal);
	
		manifoldContacts[numContacts].mLocalPointA = localPointA;
		manifoldContacts[numContacts].mLocalPointB = localPointB;
		manifoldContacts[numContacts].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(normal), signedDist);
		manifoldContacts[numContacts++].mFaceIndex = triangleIndex;
	}
}



void Gu::PCMCapsuleVsMeshContactGeneration::generateEEContacts(const Ps::aos::Vec3VArg a, const Ps::aos::Vec3VArg b,
															   const Ps::aos::Vec3VArg c, const Ps::aos::Vec3VArg normal,
															   const PxU32 triangleIndex, const Ps::aos::Vec3VArg p, 
															   const Ps::aos::Vec3VArg q, const Ps::aos::FloatVArg sqInflatedRadius,
															   Gu::MeshPersistentContact* manifoldContacts, PxU32& numContacts)
{
	
	using namespace Ps::aos;
	generateEE(p, q, sqInflatedRadius, normal, triangleIndex, a, b, manifoldContacts, numContacts);
	generateEE(p, q, sqInflatedRadius, normal, triangleIndex, b, c, manifoldContacts, numContacts);
	generateEE(p, q, sqInflatedRadius, normal, triangleIndex, a, c, manifoldContacts, numContacts);

}


void Gu::PCMCapsuleVsMeshContactGeneration::generateEEMTD(const Ps::aos::Vec3VArg p, const Ps::aos::Vec3VArg q, const Ps::aos::FloatVArg inflatedRadius,
																const Ps::aos::Vec3VArg normal, const PxU32 triangleIndex, const Ps::aos::Vec3VArg a, const Ps::aos::Vec3VArg b,
																Gu::MeshPersistentContact* manifoldContacts, PxU32& numContacts)
{
	using namespace Ps::aos;
	const FloatV zero = FZero();
	const Vec3V ab = V3Sub(b, a);
	const Vec3V n = V3Cross(ab, normal);
	const FloatV d = V3Dot(n, a);
	const FloatV np = V3Dot(n, p);
	const FloatV nq = V3Dot(n,q);
	const FloatV signP = FSub(np, d);
	const FloatV signQ = FSub(nq, d);
	const FloatV temp = FMul(signP, signQ);
	if(FAllGrtr(temp, zero)) return;//If both points in the same side as the plane, no intersect points
	
	// if colliding edge (p3,p4) and plane are parallel return no collision
	const Vec3V pq = V3Sub(q, p);
	const FloatV npq= V3Dot(n, pq); 
	if(FAllEq(npq, zero))	return;

	//calculate the intersect point in the segment pq
	const FloatV segTValue = FDiv(FSub(d, np), npq);
	const Vec3V localPointA = V3ScaleAdd(pq, segTValue, p);

	//calculate a normal perpendicular to ray localPointA + normal, 2D segment segment intersection
	const Vec3V perNormal = V3Cross(normal, pq);
	const Vec3V ap = V3Sub(localPointA, a);
	const FloatV nom = V3Dot(perNormal, ap);
	const FloatV denom = V3Dot(perNormal, ab);

	const FloatV tValue = FClamp(FDiv(nom, denom), zero, FOne());

	const Vec3V v = V3NegScaleSub(ab, tValue, ap);
	const FloatV signedDist = V3Dot(v, normal);

	if(FAllGrtr(inflatedRadius, signedDist))
	{
		
		const Vec3V localPointB = V3Sub(localPointA, v);
		manifoldContacts[numContacts].mLocalPointA = localPointA;
		manifoldContacts[numContacts].mLocalPointB = localPointB;
		manifoldContacts[numContacts].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(normal), signedDist);
		manifoldContacts[numContacts++].mFaceIndex = triangleIndex;
	}
}


void Gu::PCMCapsuleVsMeshContactGeneration::generateEEContactsMTD(const Ps::aos::Vec3VArg a, const Ps::aos::Vec3VArg b,
															   const Ps::aos::Vec3VArg c, const Ps::aos::Vec3VArg normal,
															   const PxU32 triangleIndex, const Ps::aos::Vec3VArg p, 
															   const Ps::aos::Vec3VArg q, const Ps::aos::FloatVArg inflatedRadius,
															   Gu::MeshPersistentContact* manifoldContacts, PxU32& numContacts)
{
	
	using namespace Ps::aos;
	generateEEMTD(p, q, inflatedRadius, normal, triangleIndex, a, b, manifoldContacts, numContacts);
	generateEEMTD(p, q, inflatedRadius, normal, triangleIndex, b, c, manifoldContacts, numContacts);
	generateEEMTD(p, q, inflatedRadius, normal, triangleIndex, a, c, manifoldContacts, numContacts);

}

bool Gu::PCMCapsuleVsMeshContactGeneration::generateContacts(const Ps::aos::Vec3VArg a, const Ps::aos::Vec3VArg b,
															 const Ps::aos::Vec3VArg c, const Ps::aos::Vec3VArg planeNormal, 
															 const Ps::aos::Vec3VArg normal, const PxU32 triangleIndex,
															 const Ps::aos::Vec3VArg p, const Ps::aos::Vec3VArg q,
															 const Ps::aos::FloatVArg inflatedRadius,
															 Gu::MeshPersistentContact* manifoldContacts, PxU32& numContacts)
{

	using namespace Ps::aos;
	const BoolV bTrue = BTTTT();
	
	const Vec3V ab = V3Sub(b, a);
	const Vec3V ac = V3Sub(c, a);
	const Vec3V ap = V3Sub(p, a);
	const Vec3V aq = V3Sub(q, a);

	//This is used to calculate the barycentric coordinate
	const FloatV d00 = V3Dot(ab, ab);
	const FloatV d01 = V3Dot(ab, ac);
	const FloatV d11 = V3Dot(ac, ac);
	const FloatV bdenom = FRecip(FSub(FMul(d00,d11), FMul(d01, d01)));

	//compute the intersect point of p and triangle plane abc
	const FloatV inomp = V3Dot(planeNormal, V3Neg(ap));
	const FloatV ideom = V3Dot(planeNormal, normal);

	const FloatV ipt = FDiv(inomp, ideom);
	//compute the distance from triangle plane abc
	const FloatV dist3 = V3Dot(ap, planeNormal);
	
	const Vec3V closestP31 = V3ScaleAdd(normal, ipt, p);
	const Vec3V closestP30 = p;


	//Compute the barycentric coordinate for project point of q
	const Vec3V pV20 = V3Sub(closestP31, a);
	const FloatV pD20 = V3Dot(pV20, ab);
	const FloatV pD21 = V3Dot(pV20, ac);
	const FloatV v0 = FMul(FSub(FMul(d11, pD20), FMul(d01, pD21)), bdenom);
	const FloatV w0 = FMul(FSub(FMul(d00, pD21), FMul(d01, pD20)), bdenom);

	//check closestP3 is inside the triangle
	const BoolV con0 = isValidTriangleBarycentricCoord(v0, w0);

	
	const BoolV tempCon0 = BAnd(con0, FIsGrtr(inflatedRadius, dist3));
	if(BAllEq(tempCon0, bTrue))
	{
		manifoldContacts[numContacts].mLocalPointA = closestP30;//transform to B space, it will get convert to A space later
		manifoldContacts[numContacts].mLocalPointB = closestP31;
		manifoldContacts[numContacts].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(normal), FNeg(ipt));
		manifoldContacts[numContacts++].mFaceIndex = triangleIndex;
	}
	
	const FloatV inomq = V3Dot(planeNormal, V3Neg(aq));

	//compute the distance of q and triangle plane abc
	const FloatV dist4 = V3Dot(aq, planeNormal);

	const FloatV iqt = FDiv(inomq, ideom);

	const Vec3V closestP41 = V3ScaleAdd(normal, iqt, q);
	const Vec3V closestP40 = q;

	//Compute the barycentric coordinate for project point of q
	const Vec3V qV20 = V3Sub(closestP41, a);
	const FloatV qD20 = V3Dot(qV20, ab);
	const FloatV qD21 = V3Dot(qV20, ac);
	const FloatV v1 = FMul(FSub(FMul(d11, qD20), FMul(d01, qD21)), bdenom);
	const FloatV w1 = FMul(FSub(FMul(d00, qD21), FMul(d01, qD20)), bdenom);


	const BoolV con1 = isValidTriangleBarycentricCoord(v1, w1);
	
	const BoolV tempCon1 = BAnd(con1, FIsGrtr(inflatedRadius, dist4));
	if(BAllEq(tempCon1, bTrue))
	{
		manifoldContacts[numContacts].mLocalPointA = closestP40;
		manifoldContacts[numContacts].mLocalPointB = closestP41;
		manifoldContacts[numContacts].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(normal),FNeg(iqt));
		manifoldContacts[numContacts++].mFaceIndex = triangleIndex;

	}
	
	return false;
}


/*
	t is the barycenteric coordinate of a segment
	u is the barycenteric coordinate of a triangle
	v is the barycenteric coordinate of a triangle
*/
Ps::aos::FloatV pcmDistanceSegmentTriangleSquared(	const Ps::aos::Vec3VArg p, const Ps::aos::Vec3VArg q,
													const Ps::aos::Vec3VArg a, const Ps::aos::Vec3VArg b, const Ps::aos::Vec3VArg c,
													Ps::aos::FloatV& t, Ps::aos::FloatV& u, Ps::aos::FloatV& v)
{
	using namespace Ps::aos;
	
	const FloatV one = FOne();
	const FloatV zero = FZero();
	const BoolV bTrue = BTTTT();

	const Vec3V pq = V3Sub(q, p);
	const Vec3V ab = V3Sub(b, a);
	const Vec3V ac = V3Sub(c, a);
	const Vec3V bc = V3Sub(c, b);
	const Vec3V ap = V3Sub(p, a);
	const Vec3V aq = V3Sub(q, a);

	const Vec3V n =V3Normalize(V3Cross(ab, ac)); // normalize vector

	const Vec4V combinedDot = V3Dot4(ab, ab, ab, ac, ac, ac, ap, n);
	const FloatV d00 = V4GetX(combinedDot);
	const FloatV d01 = V4GetY(combinedDot);
	const FloatV d11 = V4GetZ(combinedDot);
	const FloatV dist3 = V4GetW(combinedDot);

	const FloatV bdenom = FRecip(FSub(FMul(d00,d11), FMul(d01, d01)));
	

	const FloatV sqDist3 = FMul(dist3, dist3);


	//compute the closest point of q and triangle plane abc
	const FloatV dist4 = V3Dot(aq, n);
	const FloatV sqDist4 = FMul(dist4, dist4);
	const FloatV dMul = FMul(dist3, dist4);
	const BoolV con = FIsGrtr(zero, dMul);

	if(BAllEq(con, bTrue))
	{
		//compute the intersect point
		const FloatV nom = FNeg(V3Dot(n, ap));
		const FloatV denom = FRecip(V3Dot(n, pq));
		const FloatV t0 = FMul(nom, denom);
		const Vec3V ip = V3ScaleAdd(pq, t0, p);//V3Add(p, V3Scale(pq, t));
		const Vec3V v2 = V3Sub(ip, a);
		const FloatV d20 = V3Dot(v2, ab);
		const FloatV d21 = V3Dot(v2, ac);

		const FloatV v0 = FMul(FNegMulSub(d01, d21, FMul(d11, d20)), bdenom);
		const FloatV w0 = FMul(FNegMulSub(d01, d20, FMul(d00, d21)), bdenom);

		const BoolV con0 = isValidTriangleBarycentricCoord(v0, w0);
		if(BAllEq(con0,bTrue))
		{
			t = t0;
			u = v0;
			v = w0;
			return zero;
		}
	}

	const Vec3V closestP31 = V3NegScaleSub(n, dist3, p);//V3Sub(p, V3Scale(n, dist3));
	const Vec3V closestP41 = V3NegScaleSub(n, dist4, q);// V3Sub(q, V3Scale(n, dist4));

	//Compute the barycentric coordinate for project point of q
	const Vec3V pV20 = V3Sub(closestP31, a);
	const Vec3V qV20 = V3Sub(closestP41, a);

	const Vec4V pD2 = V3Dot4(pV20, ab, pV20, ac, qV20, ab, qV20, ac);

	const Vec4V pD2Swizzle = V4Perm_YXWZ(pD2);

	const Vec4V d11d00 = V4UnpackXY(V4Splat(d11), V4Splat(d00));

	const Vec4V v0w0v1w1 = V4Scale(V4NegMulSub(V4Splat(d01), pD2Swizzle, V4Mul(d11d00, pD2)), bdenom);

	const FloatV v0 = V4GetX(v0w0v1w1);
	const FloatV w0 = V4GetY(v0w0v1w1);
	const FloatV v1 = V4GetZ(v0w0v1w1);
	const FloatV w1 = V4GetW(v0w0v1w1);

	const BoolV _con = isValidTriangleBarycentricCoord2(v0w0v1w1);

	const BoolV con0 = BGetX(_con);
	const BoolV con1 = BGetY(_con);


	const BoolV cond2 = BAnd(con0, con1);	

	if(BAllEq(cond2, bTrue))
	{
		/*
			both p and q project points are interior point 
		*/
		const BoolV d2 = FIsGrtr(sqDist4, sqDist3);
		t = FSel(d2, zero, one);
		u = FSel(d2, v0, v1);
		v = FSel(d2, w0, w1);
		return FSel(d2, sqDist3, sqDist4);
	}
	else
	{
		Vec4V t40, t41;
		const Vec4V sqDist44 = pcmDistanceSegmentSegmentSquared4(p,pq,a,ab, b,bc, a,ac, a,ab, t40, t41); 

		const FloatV t00 = V4GetX(t40);
		const FloatV t10 = V4GetY(t40);
		const FloatV t20 = V4GetZ(t40);

		const FloatV t01 = V4GetX(t41);
		const FloatV t11 = V4GetY(t41);
		const FloatV t21 = V4GetZ(t41);

		//edge ab
		const FloatV u01 = t01;
		const FloatV v01 = zero;

		//edge bc
		const FloatV u11 = FSub(one, t11);
		const FloatV v11 = t11;

		//edge ac
		const FloatV u21 = zero;
		const FloatV v21 = t21;

		const FloatV sqDist0(V4GetX(sqDist44));
		const FloatV sqDist1(V4GetY(sqDist44));
		const FloatV sqDist2(V4GetZ(sqDist44));

		const BoolV con2 = BAnd(FIsGrtr(sqDist1, sqDist0), FIsGrtr(sqDist2, sqDist0));
		const BoolV con3 = FIsGrtr(sqDist2, sqDist1);
		const FloatV sqDistPE = FSel(con2, sqDist0, FSel(con3, sqDist1, sqDist2));
		const FloatV uEdge = FSel(con2, u01, FSel(con3, u11, u21));
		const FloatV vEdge = FSel(con2, v01, FSel(con3, v11, v21));
		const FloatV tSeg = FSel(con2,  t00, FSel(con3, t10, t20));


		if(BAllEq(con0, bTrue))
		{
			//p's project point is an interior point
			const BoolV d2 = FIsGrtr(sqDistPE, sqDist3);
			t = FSel(d2, zero, tSeg);
			u = FSel(d2, v0, uEdge);
			v = FSel(d2, w0, vEdge);
			return FSel(d2, sqDist3, sqDistPE);
		}
		else if(BAllEq(con1, bTrue))
		{
			//q's project point is an interior point
			const BoolV d2 = FIsGrtr(sqDistPE, sqDist4);
			t = FSel(d2, one, tSeg);
			u = FSel(d2, v1, uEdge);
			v = FSel(d2, w1, vEdge);
			return FSel(d2, sqDist4, sqDistPE);
		}
		else
		{
			t = tSeg;
			u = uEdge;
			v = vEdge;
			return sqDistPE;
		}

	}

}


static bool selectNormal(const Ps::aos::FloatVArg u, Ps::aos::FloatVArg v, PxU8 data)
{
	using namespace Ps::aos;
	const FloatV zero = FZero();
	const FloatV one = FOne();
	// Analysis
	if(FAllEq(u, zero))
	{
		if(FAllEq(v,zero))
		{
			// Vertex 0
			if(!(data & (Gu::ETD_CONVEX_EDGE_01|Gu::ETD_CONVEX_EDGE_20)))
				return true;
		}
		else if(FAllEq(v,one))
		{
			// Vertex 2
			if(!(data & (Gu::ETD_CONVEX_EDGE_12|Gu::ETD_CONVEX_EDGE_20)))
				return true;
		}
		else
		{
			// Edge 0-2
			if(!(data & Gu::ETD_CONVEX_EDGE_20))
				return true;
		}
	}
	else if(FAllEq(u,one))
	{
		if(FAllEq(v,zero))
		{
			// Vertex 1
			if(!(data & (Gu::ETD_CONVEX_EDGE_01|Gu::ETD_CONVEX_EDGE_12)))
				return true;

		}
	}
	else
	{
		if(FAllEq(v,zero))
		{
			// Edge 0-1
			if(!(data & Gu::ETD_CONVEX_EDGE_01))
				return true;
		}
		else
		{
			const FloatV threshold = FLoad(0.9999f);
			const FloatV temp = FAdd(u, v);
			if(FAllGrtrOrEq(temp, threshold))
			{
				// Edge 1-2
				if(!(data & Gu::ETD_CONVEX_EDGE_12))
					return true;
			}
			else
			{
				// Face
				return true;
			}
		}
	}
	return false;
}


bool Gu::PCMCapsuleVsMeshContactGeneration::processTriangle(const PxVec3* verts, const PxU32 triangleIndex, PxU8 triFlags, const PxU32* vertInds)
{
	PX_UNUSED(triangleIndex);
	PX_UNUSED(vertInds);

	using namespace Ps::aos;

	const FloatV zero = FZero();

	const Vec3V p0 = V3LoadU(verts[0]);
	const Vec3V p1 = V3LoadU(verts[1]);
	const Vec3V p2 = V3LoadU(verts[2]);

	const Vec3V p10 = V3Sub(p1, p0);
	const Vec3V p20 = V3Sub(p2, p0);

	const Vec3V n = V3Normalize(V3Cross(p10, p20));//(p1 - p0).cross(p2 - p0).getNormalized();
	const FloatV d = V3Dot(p0, n);//d = -p0.dot(n);

	const FloatV dist = FSub(V3Dot(mCapsule.getCenter(), n), d);//p.dot(n) + d;
	
	// Backface culling
	if(FAllGrtr(zero, dist))
		return false;


	FloatV t, u, v;
	const FloatV sqDist = pcmDistanceSegmentTriangleSquared(mCapsule.p0, mCapsule.p1, p0, p1, p2, t, u, v);
	
	if(FAllGrtr(mSqInflatedRadius, sqDist))
	{

		Vec3V patchNormalInTriangle;
		if(selectNormal(u, v, triFlags))
		{
			patchNormalInTriangle = n;
		}
		else
		{
			if(FAllEq(sqDist, zero))
			{
				//segment intersect with the triangle
				patchNormalInTriangle = n;
			}
			else
			{
				const Vec3V pq = V3Sub(mCapsule.p1, mCapsule.p0);
				const Vec3V pointOnSegment = V3ScaleAdd(pq, t, mCapsule.p0);
				const FloatV w = FSub(FOne(), FAdd(u, v));
				const Vec3V pointOnTriangle = V3ScaleAdd(p0, w, V3ScaleAdd(p1, u, V3Scale(p2, v)));
				patchNormalInTriangle = V3Normalize(V3Sub(pointOnSegment, pointOnTriangle));
				
			}
		}

		const PxU32 previousNumContacts = mNumContacts;

		generateContacts(p0, p1, p2, n, patchNormalInTriangle, triangleIndex, mCapsule.p0, mCapsule.p1, mInflatedRadius, mManifoldContacts, mNumContacts);
		//ML: this need to use the sqInflatedRadius to avoid some bad contacts
		generateEEContacts(p0, p1, p2,patchNormalInTriangle,  triangleIndex, mCapsule.p0, mCapsule.p1, mSqInflatedRadius, mManifoldContacts, mNumContacts);
		

		PxU32 numContacts = mNumContacts - previousNumContacts;

		if(numContacts > 0)
		{ 
			FloatV maxPen = FMax();
			for(PxU32 i = previousNumContacts; i<mNumContacts; ++i)
			{
				const FloatV pen = V4GetW(mManifoldContacts[i].mLocalNormalPen);
				mManifoldContacts[i].mLocalPointA = mMeshToConvex.transform(mManifoldContacts[i].mLocalPointA);
				maxPen = FMin(maxPen, pen);
			}

			for(PxU32 i = previousNumContacts; i<mNumContacts; ++i)
			{
				Vec3V contact0 = mManifoldContacts[i].mLocalPointB;
				for(PxU32 j=i+1; j<mNumContacts; ++j)
				{
					Vec3V contact1 = mManifoldContacts[j].mLocalPointB;
					Vec3V dif = V3Sub(contact1, contact0);
					FloatV d1 = V3Dot(dif, dif);
					if(FAllGrtr(mSqReplaceBreakingThreshold, d1))
					{
						mManifoldContacts[j] = mManifoldContacts[mNumContacts-1];
						mNumContacts--;
						j--;
					}
				}
			}

			PX_ASSERT(mNumContactPatch <PCM_MAX_CONTACTPATCH_SIZE);

			addManifoldPointToPatch(patchNormalInTriangle, maxPen, previousNumContacts);
			
			PX_ASSERT(mNumContactPatch <PCM_MAX_CONTACTPATCH_SIZE);
			if(mNumContacts >= 16)
			{
				PX_ASSERT(mNumContacts <= 64);
				processContacts(GU_CAPSULE_MANIFOLD_CACHE_SIZE);
			}
		}
	}

	return true;
}

bool Gu::PCMCapsuleVsMeshContactGeneration::processTriangle(const TriangleV& triangleV, const PxU32 triangleIndex, const CapsuleV& capsule, const Ps::aos::FloatVArg inflatedRadius, const PxU8 trigFlag,
															Gu::MeshPersistentContact* manifoldContacts, PxU32& numContacts)
{
	using namespace Ps::aos;

	const FloatV zero = FZero();

	const Vec3V p0 = triangleV.verts[0];
	const Vec3V p1 = triangleV.verts[1];
	const Vec3V p2 = triangleV.verts[2];

	const Vec3V n = triangleV.normal();

	const FloatV sqInflatedRadius = FMul(inflatedRadius, inflatedRadius);
	

	FloatV t, u, v;
	const FloatV sqDist = pcmDistanceSegmentTriangleSquared(capsule.p0, capsule.p1, p0, p1, p2, t, u, v);
	
	if(FAllGrtr(sqInflatedRadius, sqDist))
	{

		Vec3V patchNormalInTriangle;
		if(selectNormal(u, v, trigFlag))
		{
			patchNormalInTriangle = n;
		}
		else
		{
			if(FAllEq(sqDist, zero))
			{
				//segment intersect with the triangle
				patchNormalInTriangle = n;
			}
			else
			{
				const Vec3V pq = V3Sub(capsule.p1, capsule.p0);
				const Vec3V pointOnSegment = V3ScaleAdd(pq, t, capsule.p0);
				const FloatV w = FSub(FOne(), FAdd(u, v));
				const Vec3V pointOnTriangle = V3ScaleAdd(p0, w, V3ScaleAdd(p1, u, V3Scale(p2, v)));
				patchNormalInTriangle = V3Normalize(V3Sub(pointOnSegment, pointOnTriangle));
			}
		}

		generateContacts(p0, p1, p2, n, patchNormalInTriangle, triangleIndex, capsule.p0, capsule.p1, inflatedRadius, manifoldContacts, numContacts);
	
		generateEEContactsMTD(p0, p1, p2, patchNormalInTriangle, triangleIndex, capsule.p0, capsule.p1, inflatedRadius, manifoldContacts, numContacts);
		
		
	}

	return true;

}

}
}
