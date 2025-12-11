/*********************                                                        */
/*! \file VsidsBasedDivider.h
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

#ifndef __VsidsBasedDivider_h__
#define __VsidsBasedDivider_h__

#include "CdclCore.h"
#include "QueryDivider.h"

class VsidsBasedDivider : public QueryDivider
{
public:
    VsidsBasedDivider( std::shared_ptr<IEngine> engine );

    void createSubQueries( unsigned numNewSubQueries,
                           const String queryIdPrefix,
                           const unsigned previousDepth,
                           const PiecewiseLinearCaseSplit &previousSplit,
                           const unsigned timeoutInSeconds,
                           SubQueries &subQueries ) override;

private:
    std::shared_ptr<IEngine> _engine;

    const PiecewiseLinearConstraint *
    getPLConstraintToSplit( const PiecewiseLinearCaseSplit &split );
};


#endif // __VsidsBasedDivider_h__
