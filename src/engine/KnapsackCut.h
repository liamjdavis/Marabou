/*********************                                                        */
/*! \file KnapsackCut.h
 ** \verbatim
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** KnapsackCut implements a topology-aware cutting plane for ReLU networks.
 **
 ** For a "furthest" fixed ReLU target b (no fixed downstream), pre_b is
 ** expressed as a linear function of upstream f-variables and INPUT vars:
 **
 **    pre_b = constant + sum_j w_j * x_j      (x_j upstream RELU f-var or INPUT)
 **
 ** At the UNSAT leaf the leaf forced pre_b into one half-line:
 **    ACTIVE  : pre_b >= threshold   (threshold = leaf's lb on pre_b)
 **    INACTIVE: pre_b <= threshold   (threshold = leaf's ub on pre_b)
 **
 ** The cut stores ONLY static topology data (weights w_j, constant, threshold).
 ** Bounds are queried fresh from the BoundManager at every check, making the
 ** cut globally valid (sound at any subspace). The check at a new node is:
 **
 **    ACTIVE  : (constant + sum_j w_j * (lb_j if w_j>0 else ub_j)) >= threshold
 **    INACTIVE: (constant + sum_j w_j * (ub_j if w_j>0 else lb_j)) <= threshold
 **
 ** A cut group (from one UNSAT leaf) prunes the current subproblem when every
 ** cut in the group is implied: every furthest target is forced into its leaf
 ** phase, so the subproblem is subsumed by the UNSAT leaf.
 **/

#ifndef __KnapsackCut_h__
#define __KnapsackCut_h__

#include "Map.h"
#include "Vector.h"

struct KnapsackCut
{
    // The pre-activation variable (b) of the target ReLU
    unsigned targetBVar;

    // Phase of the target at the UNSAT leaf (true = active, false = inactive)
    bool isActive;

    // NLR layer and neuron indices for identification
    unsigned reluLayerIdx;
    unsigned neuronIdx;

    // Effective network weights of pre_b expressed in terms of upstream
    // f-vars / INPUT vars. PURE WEIGHTS -- no bounds baked in. Bounds are
    // queried dynamically from the BoundManager at check time.
    Map<unsigned, double> coefficients;

    // Static part of pre_b: bias plus folded contributions from eliminated
    // (fixed-value) neurons.
    double constant;

    // The leaf's bound on pre_b that the cut must reproduce at a new node:
    //   ACTIVE  : threshold = leaf's lower bound on pre_b
    //   INACTIVE: threshold = leaf's upper bound on pre_b
    double threshold;
};

struct KnapsackCutGroup
{
    Vector<KnapsackCut> cuts;
    unsigned depth;
};

#endif // __KnapsackCut_h__
