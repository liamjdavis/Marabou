/*********************                                                        */
/*! \file KnapsackCutManager.cpp
 ** \verbatim
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **/

#include "KnapsackCutManager.h"

#include "FloatUtils.h"
#include "Layer.h"
#include "NetworkLevelReasoner.h"
#include "PiecewiseLinearConstraint.h"
#include "ReluConstraint.h"

KnapsackCutManager::KnapsackCutManager()
    : _nlr( nullptr )
    , _plConstraints( nullptr )
    , _boundManager( nullptr )
    , _numCutsBuilt( 0 )
    , _numPrunes( 0 )
    , _numPruneChecks( 0 )
    , _initialized( false )
{
}

void KnapsackCutManager::initialize( NLR::NetworkLevelReasoner *nlr,
                                     const List<PiecewiseLinearConstraint *> &plConstraints,
                                     IBoundManager *boundManager )
{
    _nlr = nlr;
    _plConstraints = &plConstraints;
    _boundManager = boundManager;
    _cutGroups.clear();
    _downstreamRelus.clear();
    _layerNeuronToRelu.clear();
    _fVarToRelu.clear();
    _numCutsBuilt = 0;
    _numPrunes = 0;
    _numPruneChecks = 0;
    _initialized = true;

    if ( !_nlr )
        return;

    // f-variable -> owning ReluConstraint
    for ( auto plc : *_plConstraints )
    {
        ReluConstraint *r = dynamic_cast<ReluConstraint *>( plc );
        if ( r )
            _fVarToRelu[r->getF()] = r;
    }

    // (NLR layer, neuron) -> ReluConstraint, restricted to RELU layers
    for ( unsigned L = 0; L < _nlr->getNumberOfLayers(); ++L )
    {
        const NLR::Layer *layer = _nlr->getLayer( L );
        if ( layer->getLayerType() != NLR::Layer::RELU )
            continue;
        for ( unsigned j = 0; j < layer->getSize(); ++j )
        {
            if ( !layer->neuronHasVariable( j ) )
                continue;
            unsigned fVar = layer->neuronToVariable( j );
            if ( _fVarToRelu.exists( fVar ) )
                _layerNeuronToRelu[L][j] = _fVarToRelu[fVar];
        }
    }

    // Per-ReLU forward reachability through the per-neuron NLR graph
    unsigned totalDownstreamEdges = 0;
    unsigned maxDownstream = 0;
    for ( const auto &outer : _layerNeuronToRelu )
    {
        unsigned L = outer.first;
        for ( const auto &inner : outer.second )
        {
            unsigned j = inner.first;
            ReluConstraint *r = inner.second;
            Set<ReluConstraint *> downstream;
            bfsDownstreamRelus( NLR::NeuronIndex( L, j ), downstream );
            _downstreamRelus[r] = downstream;
            totalDownstreamEdges += downstream.size();
            if ( downstream.size() > maxDownstream )
                maxDownstream = downstream.size();
        }
    }

    printf( "[knapsack] initialize: tracked_relus=%u, "
            "total_downstream_edges=%u, max_downstream_per_relu=%u\n",
            _fVarToRelu.size(),
            totalDownstreamEdges,
            maxDownstream );
}

void KnapsackCutManager::bfsDownstreamRelus( NLR::NeuronIndex start,
                                             Set<ReluConstraint *> &downstream ) const
{
    Queue<NLR::NeuronIndex> queue;
    Set<NLR::NeuronIndex> visited;
    queue.push( start );
    visited.insert( start );

    while ( !queue.empty() )
    {
        NLR::NeuronIndex cur = queue.peak();
        queue.pop();

        const NLR::Layer *layer = _nlr->getLayer( cur._layer );
        for ( unsigned S : layer->getSuccessorLayers() )
        {
            const NLR::Layer *succ = _nlr->getLayer( S );
            unsigned succSize = succ->getSize();

            if ( succ->getLayerType() == NLR::Layer::WEIGHTED_SUM )
            {
                // Per-neuron edges via nonzero weights from cur to (S, k)
                for ( unsigned k = 0; k < succSize; ++k )
                {
                    double w = succ->getWeight( cur._layer, cur._neuron, k );
                    if ( FloatUtils::isZero( w ) )
                        continue;
                    NLR::NeuronIndex next( S, k );
                    if ( visited.exists( next ) )
                        continue;
                    visited.insert( next );
                    queue.push( next );
                }
            }
            else
            {
                // Activation layer (RELU, SIGN, ABS, SIGMOID, MAX, etc.):
                // per-neuron edges via activation-source membership.
                for ( unsigned k = 0; k < succSize; ++k )
                {
                    bool isSrc = false;
                    for ( const auto &src : succ->getActivationSources( k ) )
                    {
                        if ( src._layer == cur._layer && src._neuron == cur._neuron )
                        {
                            isSrc = true;
                            break;
                        }
                    }
                    if ( !isSrc )
                        continue;
                    NLR::NeuronIndex next( S, k );
                    if ( visited.exists( next ) )
                        continue;
                    visited.insert( next );
                    queue.push( next );

                    if ( succ->getLayerType() == NLR::Layer::RELU &&
                         _layerNeuronToRelu.exists( S ) && _layerNeuronToRelu[S].exists( k ) )
                        downstream.insert( _layerNeuronToRelu[S][k] );
                }
            }
        }
    }
}

void KnapsackCutManager::collectFromCurrentLeaf()
{
    if ( !_initialized || !_nlr || !_plConstraints )
        return;

    // Collect all fixed ReLUs tracked in the NLR mapping.
    Set<ReluConstraint *> fixed;
    Map<ReluConstraint *, NLR::NeuronIndex> fixedAt;
    Map<ReluConstraint *, PhaseStatus> fixedPhase;
    for ( const auto &outer : _layerNeuronToRelu )
    {
        unsigned L = outer.first;
        for ( const auto &inner : outer.second )
        {
            unsigned j = inner.first;
            ReluConstraint *r = inner.second;
            if ( !r->phaseFixed() )
                continue;
            PhaseStatus phase = r->getPhaseStatus();
            if ( phase != RELU_PHASE_ACTIVE && phase != RELU_PHASE_INACTIVE )
                continue;
            fixed.insert( r );
            fixedAt[r] = NLR::NeuronIndex( L, j );
            fixedPhase[r] = phase;
        }
    }

    if ( fixed.empty() )
        return;

    // Filter to "furthest": no fixed ReLU is forward-reachable from r.
    KnapsackCutGroup group;
    group.depth = 0;
    for ( ReluConstraint *r : fixed )
    {
        bool hasFixedDownstream = false;
        if ( _downstreamRelus.exists( r ) )
        {
            for ( ReluConstraint *d : _downstreamRelus[r] )
            {
                if ( fixed.exists( d ) )
                {
                    hasFixedDownstream = true;
                    break;
                }
            }
        }
        if ( hasFixedDownstream )
            continue;

        const NLR::NeuronIndex &idx = fixedAt[r];
        bool active = ( fixedPhase[r] == RELU_PHASE_ACTIVE );
        KnapsackCut cut;
        if ( buildCut( r, idx._layer, idx._neuron, active, cut ) )
        {
            group.cuts.append( cut );
            ++_numCutsBuilt;
        }
    }

    if ( !group.cuts.empty() )
        _cutGroups.append( group );
}

bool KnapsackCutManager::checkPruning()
{
    if ( !_initialized || _cutGroups.empty() )
        return false;

    ++_numPruneChecks;
    for ( const KnapsackCutGroup &group : _cutGroups )
    {
        bool groupImplied = true;
        for ( const KnapsackCut &cut : group.cuts )
        {
            // Worst-case LHS = constant + Sum over upstream phase indicators:
            //   z_i = 1 (active)   -> a_i
            //   z_i = 0 (inactive) -> 0
            //   unfixed/untracked  -> min(0, a_i)
            // If worst-case LHS >= 0 then the cut is implied at this node.
            double lhs = cut.constant;
            for ( const auto &entry : cut.coefficients )
            {
                unsigned fVar = entry.first;
                double a = entry.second;
                if ( !_fVarToRelu.exists( fVar ) )
                {
                    if ( a < 0 )
                        lhs += a;
                    continue;
                }
                ReluConstraint *upstream = _fVarToRelu[fVar];
                if ( !upstream->phaseFixed() )
                {
                    if ( a < 0 )
                        lhs += a;
                    continue;
                }
                if ( upstream->getPhaseStatus() == RELU_PHASE_ACTIVE )
                    lhs += a;
                // RELU_PHASE_INACTIVE contributes 0.
            }
            if ( lhs < 0 )
            {
                groupImplied = false;
                break;
            }
        }
        if ( groupImplied )
        {
            ++_numPrunes;
            return true;
        }
    }
    return false;
}

void KnapsackCutManager::printSummary() const
{
    if ( !_initialized )
        return;

    // Count unique cuts across all stored groups (by full coef+constant identity).
    Set<KnapsackCutKey> unique;
    for ( const KnapsackCutGroup &group : _cutGroups )
    {
        for ( const KnapsackCut &cut : group.cuts )
        {
            KnapsackCutKey key;
            key.reluLayerIdx = cut.reluLayerIdx;
            key.neuronIdx = cut.neuronIdx;
            key.isActive = cut.isActive;
            key.coefficients = cut.coefficients;
            key.constant = cut.constant;
            unique.insert( key );
        }
    }

    printf( "[knapsack] summary: tracked_relus=%u groups=%u cuts_built=%u "
            "unique_cuts=%u prune_checks=%u prunes_applied=%u\n",
            _fVarToRelu.size(),
            _cutGroups.size(),
            _numCutsBuilt,
            unique.size(),
            _numPruneChecks,
            _numPrunes );
}

unsigned KnapsackCutManager::getNumCutGroups() const
{
    return _cutGroups.size();
}

void KnapsackCutManager::reset()
{
    _cutGroups.clear();
    _downstreamRelus.clear();
    _layerNeuronToRelu.clear();
    _fVarToRelu.clear();
    _numCutsBuilt = 0;
    _numPrunes = 0;
    _numPruneChecks = 0;
    _initialized = false;
}

double KnapsackCutManager::worstCaseBoxContribution( double w, double lb, double ub, bool lower )
{
    // lower=true: smallest possible w*value; lower=false: largest possible w*value.
    if ( lower )
        return ( w > 0 ) ? w * lb : w * ub;
    return ( w > 0 ) ? w * ub : w * lb;
}

void KnapsackCutManager::foldBackward( unsigned layerIdx,
                                       unsigned neuron,
                                       double w,
                                       bool lower,
                                       KnapsackCut &cut ) const
{
    const NLR::Layer *layer = _nlr->getLayer( layerIdx );

    // Eliminated neurons have fixed values -> fold to constant directly.
    if ( layer->neuronEliminated( neuron ) )
    {
        cut.constant += w * layer->getEliminatedNeuronValue( neuron );
        return;
    }

    NLR::Layer::Type type = layer->getLayerType();

    if ( type == NLR::Layer::WEIGHTED_SUM )
    {
        cut.constant += w * layer->getBias( neuron );
        for ( const auto &srcEntry : layer->getSourceLayers() )
        {
            unsigned srcLayerIdx = srcEntry.first;
            unsigned srcSize = srcEntry.second;
            for ( unsigned srcNeuron = 0; srcNeuron < srcSize; ++srcNeuron )
            {
                double wPrime = layer->getWeight( srcLayerIdx, srcNeuron, neuron );
                if ( FloatUtils::isZero( wPrime ) )
                    continue;
                foldBackward( srcLayerIdx, srcNeuron, w * wPrime, lower, cut );
            }
        }
        return;
    }

    if ( type == NLR::Layer::RELU && _layerNeuronToRelu.exists( layerIdx ) &&
         _layerNeuronToRelu[layerIdx].exists( neuron ) )
    {
        // Phase-indicator coefficient: contribution is (worst-case bound on
        // w*post) when z=1, and 0 when z=0. Coef = worst-case bound on w*post
        // restricted to post in [lb, ub] (both >= 0 by ReLU semantics).
        double lb = layer->getLb( neuron );
        double ub = layer->getUb( neuron );
        double coef = worstCaseBoxContribution( w, lb, ub, lower );
        unsigned fVar = layer->neuronToVariable( neuron );
        if ( cut.coefficients.exists( fVar ) )
            cut.coefficients[fVar] += coef;
        else
            cut.coefficients[fVar] = coef;
        return;
    }

    // INPUT, untracked RELU, or other activation: box-fold via current bounds.
    double lb = layer->getLb( neuron );
    double ub = layer->getUb( neuron );
    cut.constant += worstCaseBoxContribution( w, lb, ub, lower );
}

bool KnapsackCutManager::buildCut( ReluConstraint *r,
                                   unsigned reluLayerIdx,
                                   unsigned reluNeuronIdx,
                                   bool active,
                                   KnapsackCut &outCut ) const
{
    const NLR::Layer *reluLayer = _nlr->getLayer( reluLayerIdx );
    const auto &sources = reluLayer->getSourceLayers();
    if ( sources.size() != 1 )
        return false; // unexpected RELU topology

    unsigned preActLayerIdx = sources.begin()->first;

    outCut.targetBVar = r->getB();
    outCut.isActive = active;
    outCut.reluLayerIdx = reluLayerIdx;
    outCut.neuronIdx = reluNeuronIdx;
    outCut.coefficients.clear();
    outCut.constant = 0.0;

    // Active: want LB of pre_b such that LB >= 0 implies pre_b >= 0.
    // Inactive: want UB of pre_b such that UB <= 0 implies pre_b <= 0;
    //           store negated to keep canonical form Sum a_i z_i + c >= 0.
    bool lower = active;
    foldBackward( preActLayerIdx, reluNeuronIdx, 1.0, lower, outCut );

    if ( !active )
    {
        for ( auto &entry : outCut.coefficients )
            entry.second = -entry.second;
        outCut.constant = -outCut.constant;
    }

    return true;
}
