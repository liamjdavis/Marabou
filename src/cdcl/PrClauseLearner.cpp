/*********************                                                        */
/*! \file PrClauseLearner.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Liam Davis
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **/

#include "PrClauseLearner.h"

PrClauseLearner::PrClauseLearner()
    : _harvesting( false )
    , _numObservedTrails( 0 )
    , _numHarvestedCandidates( 0 )
{
}

void PrClauseLearner::startHarvest()
{
    _harvesting = true;
    _pool.clear();
    _harvest.clear();
    _numObservedTrails = 0;
    _numHarvestedCandidates = 0;
}

void PrClauseLearner::stopHarvest()
{
    _harvesting = false;
}

bool PrClauseLearner::isHarvesting() const
{
    return _harvesting;
}

void PrClauseLearner::addPoolClause( const Set<int> &clause )
{
    if ( !_harvesting || clause.empty() )
        return;

    _pool.append( clause );
}

void PrClauseLearner::addPoolClauseFromCube( const Set<int> &cube )
{
    if ( !_harvesting || cube.empty() )
        return;

    Set<int> clause;
    for ( int lit : cube )
        clause.insert( -lit );

    _pool.append( clause );
}

const Vector<Set<int>> &PrClauseLearner::getPoolClauses() const
{
    return _pool;
}

void PrClauseLearner::observeTrail( const Vector<int> &trail )
{
    if ( !_harvesting || trail.empty() )
        return;

    ++_numObservedTrails;

    Set<int> assigned;
    for ( int lit : trail )
        assigned.insert( lit );

    // Lemma-first carve: trail literals falsifying some touched-but-not-
    // satisfied pool clause form the condition; the rest are the autarky
    Set<int> condition;
    for ( const Set<int> &clause : _pool )
    {
        bool touched = false;
        bool satisfied = false;
        for ( int lit : clause )
        {
            if ( assigned.exists( lit ) )
            {
                satisfied = true;
                break;
            }
            if ( assigned.exists( -lit ) )
                touched = true;
        }

        if ( satisfied || !touched )
            continue;

        for ( int lit : clause )
            if ( assigned.exists( -lit ) )
                condition.insert( -lit );
    }

    for ( int lit : trail )
    {
        if ( condition.exists( lit ) )
            continue;

        List<Set<int>> &bucket = _harvest[lit];
        if ( !bucket.exists( condition ) )
        {
            bucket.append( condition );
            ++_numHarvestedCandidates;
        }
    }
}

List<Set<int>> PrClauseLearner::finalizeHarvest()
{
    _harvesting = false;

    List<Set<int>> result;
    for ( const auto &pair : _harvest )
    {
        int autarkyLit = pair.first;
        const List<Set<int>> &bucket = pair.second;

        for ( const Set<int> &condition : bucket )
        {
            // Drop the clause if a strictly smaller condition for the same
            // autarky literal exists - the smaller clause subsumes it
            bool subsumed = false;
            for ( const Set<int> &other : bucket )
            {
                if ( other.size() < condition.size() &&
                     Set<int>::containedIn( other, condition ) )
                {
                    subsumed = true;
                    break;
                }
            }

            if ( subsumed )
                continue;

            Set<int> clause;
            for ( int conditionLit : condition )
                clause.insert( -conditionLit );
            clause.insert( -autarkyLit );
            result.append( clause );
        }
    }

    _harvest.clear();
    return result;
}

unsigned PrClauseLearner::getNumObservedTrails() const
{
    return _numObservedTrails;
}

unsigned PrClauseLearner::getNumHarvestedCandidates() const
{
    return _numHarvestedCandidates;
}
