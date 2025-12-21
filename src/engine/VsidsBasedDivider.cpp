/*********************                                                        */
/*! \file VsidsBasedDivider.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Idan Refaeli
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#include "VsidsBasedDivider.h"

#include "InfeasibleQueryException.h"

#include <utility>

VsidsBasedDivider::VsidsBasedDivider( std::shared_ptr<IEngine> engine )
    : _engine( std::move( engine ) )
{
}

void VsidsBasedDivider::createSubQueries( unsigned int numNewSubQueries,
                                          const String queryIdPrefix,
                                          const unsigned int previousDepth,
                                          const PiecewiseLinearCaseSplit &previousSplit,
                                          const unsigned int timeoutInSeconds,
                                          SubQueries &subQueries )
{
    unsigned numBisects = (unsigned)log2( numNewSubQueries );

    List<PiecewiseLinearCaseSplit *> splits;
    auto tempSplit = new PiecewiseLinearCaseSplit();
    *tempSplit = previousSplit;
    splits.append( tempSplit );

    for ( unsigned i = 0; i < numBisects; ++i )
    {
        List<PiecewiseLinearCaseSplit *> newSplits;
        for ( const auto &split : splits )
        {
            const PiecewiseLinearConstraint *pLConstraintToSplit = getPLConstraintToSplit( *split );
            if ( pLConstraintToSplit == NULL )
            {
                auto newSplit = new PiecewiseLinearCaseSplit();
                *newSplit = *split;
                newSplits.append( newSplit );
            }
            else
            {
                auto caseSplits = pLConstraintToSplit->getCaseSplits();
                for ( const auto &caseSplit : caseSplits )
                {
                    auto newSplit = new PiecewiseLinearCaseSplit();
                    *newSplit = *split;
                    for ( const auto &tightening : caseSplit.getBoundTightenings() )
                        newSplit->storeBoundTightening( tightening );
                    newSplits.append( newSplit );

                    for ( int lit : caseSplit.getCdclLiterals() )
                        newSplit->addCdclLiteral( lit );
                }
            }
            delete split;
        }
        splits = newSplits;
    }

    unsigned queryIdSuffix = 1; // For query id
    // Create a new subquery for each newly created input region
    for ( const auto &split : splits )
    {
        // Create a new query id
        String queryId;
        if ( queryIdPrefix == "" )
            queryId = queryIdPrefix + Stringf( "%u", queryIdSuffix++ );
        else
            queryId = queryIdPrefix + Stringf( "-%u", queryIdSuffix++ );

        // Construct the new subquery and add it to subqueries
        SubQuery *subQuery = new SubQuery;
        subQuery->_queryId = queryId;
        subQuery->_depth = previousDepth + 1;
        subQuery->_split.reset( split );
        subQuery->_timeoutInSeconds = timeoutInSeconds;
        subQueries.append( subQuery );
    }
}

const PiecewiseLinearConstraint *
VsidsBasedDivider::getPLConstraintToSplit( const PiecewiseLinearCaseSplit &split )
{
    try
    {
        _engine->applySnCSplit( split, "", true );
    }
    catch ( const InfeasibleQueryException & )
    {
        return NULL;
    }

    const PiecewiseLinearConstraint *constraintToSplit = NULL;
    unsigned varToSplit = _engine->getCdclCore()->decideSplitVarBasedOnPolarityAndVsids();
    constraintToSplit = _engine->getCdclCore()->getPlc( varToSplit );
    _engine->getContext().pop();
    _engine->postContextPopHook();
    return constraintToSplit;
}
