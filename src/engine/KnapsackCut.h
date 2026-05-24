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
 ** For a ReLU neuron b with pre-activation pre_b = sum_i W_bi * post_i + bias_b,
 ** the cut encodes: if the weighted combination of upstream phase indicators
 ** exceeds a threshold, then b's phase is guaranteed.
 **
 ** A cut group (from a single UNSAT leaf) prunes a subproblem when ALL cuts
 ** in the group are satisfied, meaning all target neurons are in their
 ** UNSAT-leaf phases.
 **
 ** Only "furthest" fixes are included: neurons with no downstream fixed neurons.
 **
 ** Cut for active phase (pre_b >= 0):
 **   sum_i a_i * z_i + c >= 0
 **   where a_i = W_bi * lb_i (if W_bi > 0) or W_bi * ub_i (if W_bi < 0)
 **         c = bias_b + folded constant contributions
 **
 ** Cut for inactive phase (pre_b <= 0), negated to >= form:
 **   sum_i a_i * z_i + c >= 0
 **   where a_i = -W_bi * ub_i (if W_bi > 0) or -W_bi * lb_i (if W_bi < 0)
 **         c = -bias_b + folded constant contributions
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

    // Coefficients keyed by upstream ReLU f-variable
    // coeff[f_var] = contribution when upstream neuron is active (z=1)
    Map<unsigned, double> coefficients;

    // Constant term (folded bias + non-phase-indicator contributions)
    double constant;
};

struct KnapsackCutGroup
{
    Vector<KnapsackCut> cuts;
    unsigned depth;
};

#endif // __KnapsackCut_h__
