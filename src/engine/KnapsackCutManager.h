/*********************                                                        */
/*! \file KnapsackCutManager.h
 ** \verbatim
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** KnapsackCutManager owns the lifecycle of KnapsackCutGroups: precomputes
 ** the downstream-ReLU reachability dictionary at solve start, collects a
 ** cut group from each UNSAT leaf, and checks at every new search node
 ** whether any stored cut group is implied (which would mean the current
 ** subproblem is subsumed by a previously-explored UNSAT leaf and can be
 ** pruned).
 **/

#ifndef __KnapsackCutManager_h__
#define __KnapsackCutManager_h__

#include "KnapsackCut.h"
#include "List.h"
#include "Map.h"
#include "NeuronIndex.h"
#include "Queue.h"
#include "Set.h"
#include "Vector.h"

class IBoundManager;
class PiecewiseLinearConstraint;
class ReluConstraint;

namespace NLR {
class NetworkLevelReasoner;
}

/*
  Canonical identity of a cut, used only for end-of-solve "unique cuts"
  counting. Two cuts with the same key represent the same inequality up
  to constant.
*/
struct KnapsackCutKey
{
    unsigned reluLayerIdx;
    unsigned neuronIdx;
    bool isActive;
    Map<unsigned, double> coefficients;
    double constant;

    bool operator<( const KnapsackCutKey &other ) const
    {
        if ( reluLayerIdx != other.reluLayerIdx )
            return reluLayerIdx < other.reluLayerIdx;
        if ( neuronIdx != other.neuronIdx )
            return neuronIdx < other.neuronIdx;
        if ( isActive != other.isActive )
            return isActive < other.isActive;
        if ( coefficients != other.coefficients )
            return coefficients < other.coefficients;
        return constant < other.constant;
    }
};

class KnapsackCutManager
{
public:
    KnapsackCutManager();

    /*
      One-time setup. Builds the downstream-ReLU reachability dictionary
      from the NLR. Safe to call with a NULL NLR (manager becomes a no-op).
    */
    void initialize( NLR::NetworkLevelReasoner *nlr,
                     const List<PiecewiseLinearConstraint *> &plConstraints,
                     IBoundManager *boundManager );

    /*
      Collect a cut group from the current UNSAT leaf and append to
      _cutGroups.
    */
    void collectFromCurrentLeaf();

    /*
      Returns true if at least one stored group is fully implied at the
      current node -- caller throws InfeasibleQueryException to prune.
      Increments the prune counter on success.
    */
    bool checkPruning();

    /*
      End-of-solve summary line. Counts unique cuts on demand by scanning
      stored groups.
    */
    void printSummary() const;

    /*
      Number of cut groups stored. For statistics / debugging.
    */
    unsigned getNumCutGroups() const;

    /*
      Reset all stored cuts (e.g., between solve() calls).
    */
    void reset();

private:
    NLR::NetworkLevelReasoner *_nlr;
    const List<PiecewiseLinearConstraint *> *_plConstraints;
    IBoundManager *_boundManager;

    /*
      For each ReluConstraint, the set of ReluConstraints reachable
      forward through the NLR. Built once in initialize().
    */
    Map<ReluConstraint *, Set<ReluConstraint *>> _downstreamRelus;

    /*
      Map from (NLR layer index, neuron index) of a ReLU layer to the
      owning ReluConstraint. Built once in initialize().
    */
    Map<unsigned, Map<unsigned, ReluConstraint *>> _layerNeuronToRelu;

    /*
      f-variable -> owning ReluConstraint. Cached at initialize().
    */
    Map<unsigned, ReluConstraint *> _fVarToRelu;

    /*
      Stored cut groups, one per UNSAT leaf encountered so far.
    */
    Vector<KnapsackCutGroup> _cutGroups;

    /*
      Counters for the end-of-solve summary line.
    */
    unsigned _numCutsBuilt;
    unsigned _numPrunes;
    unsigned _numPruneChecks;

    bool _initialized;

    /*
      BFS forward from a starting neuron through the per-neuron NLR
      graph; populate downstream with every RELU-typed neuron's owning
      ReluConstraint reached.
    */
    void bfsDownstreamRelus( NLR::NeuronIndex start, Set<ReluConstraint *> &downstream ) const;

    /*
      Build a KnapsackCut for target ReluConstraint r in the given phase
      (RELU_PHASE_ACTIVE or RELU_PHASE_INACTIVE). Returns true if the cut
      was successfully built; false if the topology is not foldable (e.g.,
      target ReLU has no NLR source layer).
    */
    bool buildCut( ReluConstraint *r,
                   unsigned reluLayerIdx,
                   unsigned reluNeuronIdx,
                   bool active,
                   KnapsackCut &outCut ) const;

    /*
      Recursively fold the value of (layer, neuron) scaled by weight w
      into the cut. lower=true contributes a lower bound of (w * value);
      lower=false contributes an upper bound. Recursion terminates at
      RELU outputs (emitted as phase-indicator coefficients), INPUT
      neurons (folded to constant via bounds), or other activations
      (folded as box bounds).
    */
    void foldBackward( unsigned layerIdx,
                       unsigned neuron,
                       double w,
                       bool lower,
                       KnapsackCut &cut ) const;

    /*
      Worst-case contribution from a neuron whose value lies in [lb, ub],
      scaled by w. lower=true gives the worst-case lower bound (smallest
      w*value); lower=false gives the worst-case upper bound (largest).
    */
    static double worstCaseBoxContribution( double w, double lb, double ub, bool lower );
};

#endif // __KnapsackCutManager_h__
