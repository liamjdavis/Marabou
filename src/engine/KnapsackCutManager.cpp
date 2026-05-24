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
#include "IBoundManager.h"
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
    , _debug( false )
{
}

void KnapsackCutManager::setDebug( bool debug )
{
    _debug = debug;
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
    _fVarToLayerNeuron.clear();
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
            {
                _layerNeuronToRelu[L][j] = _fVarToRelu[fVar];
                _fVarToLayerNeuron[fVar] = NLR::NeuronIndex( L, j );
            }
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

            if ( _debug )
            {
                printf( "[knapsack-dbg] BUILD cut: target b_var=%u "
                        "(L=%u, j=%u) phase=%s constant=%.17g threshold=%.17g "
                        "weights=%u\n",
                        cut.targetBVar,
                        cut.reluLayerIdx,
                        cut.neuronIdx,
                        cut.isActive ? "ACTIVE" : "INACTIVE",
                        cut.constant,
                        cut.threshold,
                        cut.coefficients.size() );
                for ( const auto &entry : cut.coefficients )
                {
                    unsigned var = entry.first;
                    double w = entry.second;
                    if ( _fVarToLayerNeuron.exists( var ) )
                    {
                        NLR::NeuronIndex upIdx = _fVarToLayerNeuron[var];
                        printf( "[knapsack-dbg]   weight: var=%u (RELU L=%u "
                                "j=%u) w=%.17g\n",
                                var,
                                upIdx._layer,
                                upIdx._neuron,
                                w );
                    }
                    else
                    {
                        printf( "[knapsack-dbg]   weight: var=%u "
                                "(INPUT/other) w=%.17g\n",
                                var,
                                w );
                    }
                }
            }
        }
    }

    if ( !group.cuts.empty() )
        _cutGroups.append( group );
}

bool KnapsackCutManager::checkPruning()
{
    if ( !_initialized || _cutGroups.empty() || !_boundManager )
        return false;

    ++_numPruneChecks;
    for ( unsigned gi = 0; gi < _cutGroups.size(); ++gi )
    {
        const KnapsackCutGroup &group = _cutGroups[gi];
        bool groupImplied = true;
        for ( const KnapsackCut &cut : group.cuts )
        {
            // Compute envelope of pre_b at this node using CURRENT BM bounds.
            //   ACTIVE  : LB(pre_b) = constant + sum_j w_j * (lb_j if w>0 else ub_j)
            //   INACTIVE: UB(pre_b) = constant + sum_j w_j * (ub_j if w>0 else lb_j)
            double envelope = cut.constant;
            for ( const auto &entry : cut.coefficients )
            {
                unsigned var = entry.first;
                double w = entry.second;
                double lb = _boundManager->getLowerBound( var );
                double ub = _boundManager->getUpperBound( var );
                if ( cut.isActive )
                    envelope += ( w > 0 ) ? w * lb : w * ub;
                else
                    envelope += ( w > 0 ) ? w * ub : w * lb;
            }
            bool implied = cut.isActive ? ( envelope >= cut.threshold )
                                        : ( envelope <= cut.threshold );
            if ( !implied )
            {
                groupImplied = false;
                break;
            }
        }
        if ( groupImplied )
        {
            ++_numPrunes;
            if ( _debug )
            {
                printf( "[knapsack-dbg] PRUNE FIRE: group=%u with %u cuts; "
                        "per-cut breakdown follows\n",
                        gi,
                        group.cuts.size() );
                for ( unsigned ci = 0; ci < group.cuts.size(); ++ci )
                {
                    const KnapsackCut &cut = group.cuts[ci];
                    double envelope = cut.constant;
                    printf( "[knapsack-dbg]   cut %u: target b_var=%u "
                            "(L=%u, j=%u) phase=%s constant=%.17g "
                            "threshold=%.17g\n",
                            ci,
                            cut.targetBVar,
                            cut.reluLayerIdx,
                            cut.neuronIdx,
                            cut.isActive ? "ACTIVE" : "INACTIVE",
                            cut.constant,
                            cut.threshold );
                    printf( "[knapsack-dbg]     target b_var bounds now: "
                            "lb=%.17g ub=%.17g\n",
                            _boundManager->getLowerBound( cut.targetBVar ),
                            _boundManager->getUpperBound( cut.targetBVar ) );
                    for ( const auto &entry : cut.coefficients )
                    {
                        unsigned var = entry.first;
                        double w = entry.second;
                        double lb = _boundManager->getLowerBound( var );
                        double ub = _boundManager->getUpperBound( var );
                        double contrib;
                        if ( cut.isActive )
                            contrib = ( w > 0 ) ? w * lb : w * ub;
                        else
                            contrib = ( w > 0 ) ? w * ub : w * lb;
                        envelope += contrib;
                        if ( _fVarToLayerNeuron.exists( var ) )
                        {
                            NLR::NeuronIndex up = _fVarToLayerNeuron[var];
                            printf( "[knapsack-dbg]     var=%u (RELU L=%u "
                                    "j=%u) w=%.17g bm_lb=%.17g bm_ub=%.17g "
                                    "contrib=%.17g\n",
                                    var,
                                    up._layer,
                                    up._neuron,
                                    w,
                                    lb,
                                    ub,
                                    contrib );
                        }
                        else
                        {
                            printf( "[knapsack-dbg]     var=%u (INPUT/other) "
                                    "w=%.17g bm_lb=%.17g bm_ub=%.17g "
                                    "contrib=%.17g\n",
                                    var,
                                    w,
                                    lb,
                                    ub,
                                    contrib );
                        }
                    }
                    printf( "[knapsack-dbg]     envelope=%.17g, "
                            "threshold=%.17g, implied=%s\n",
                            envelope,
                            cut.threshold,
                            ( cut.isActive ? envelope >= cut.threshold
                                           : envelope <= cut.threshold )
                                ? "YES"
                                : "NO" );
                }
            }
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
            key.threshold = cut.threshold;
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
    _fVarToLayerNeuron.clear();
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
                                       bool /* lower -- unused; eval uses BM */,
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
                foldBackward( srcLayerIdx, srcNeuron, w * wPrime, false, cut );
            }
        }
        return;
    }

    // Anything else (RELU, INPUT, SIGN, ABS, SIGMOID, MAX, ...) becomes a
    // bound-dependent term in pre_b. Store the pure effective weight; the
    // check phase queries the BoundManager for current bounds and computes
    // the worst-case contribution dynamically.
    if ( !layer->neuronHasVariable( neuron ) )
    {
        // Fall back to whatever the layer reports (eliminated already handled).
        cut.constant += w * layer->getLb( neuron );
        return;
    }
    unsigned var = layer->neuronToVariable( neuron );
    if ( cut.coefficients.exists( var ) )
        cut.coefficients[var] += w;
    else
        cut.coefficients[var] = w;
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

    // Fold pre_b into pure weights + static constants. The "lower" flag is
    // unused at build time -- the eval phase recomputes bounds from BM.
    foldBackward( preActLayerIdx, reluNeuronIdx, 1.0, false, outCut );

    // Threshold: leaf's bound on pre_b that the cut must reproduce at any
    // future node to imply the leaf's phase fix.
    //   ACTIVE  : leaf's lb on pre_b (typically 0; tighter if extra
    //             constraints raised it)
    //   INACTIVE: leaf's ub on pre_b (typically 0; tighter if lowered)
    if ( _boundManager )
    {
        outCut.threshold = active ? _boundManager->getLowerBound( outCut.targetBVar )
                                  : _boundManager->getUpperBound( outCut.targetBVar );
    }
    else
    {
        outCut.threshold = 0.0;
    }

    return true;
}
