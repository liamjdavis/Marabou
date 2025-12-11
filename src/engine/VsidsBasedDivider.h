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

#include "QueryDivider.h"

#include "CdclCore.h"

class VsidsBasedDivider : public QueryDivider
{
public:
    VsidsBasedDivider( const CdclCore* cdclCore );

    void createSubQueries( unsigned numNewSubQueries,
                           const String queryIdPrefix,
                           const unsigned previousDepth,
                           const PiecewiseLinearCaseSplit &previousSplit,
                           const unsigned timeoutInSeconds,
                           SubQueries &subQueries ) override;

private:
    const CdclCore* _cdclCore;
};


#endif // __VsidsBasedDivider_h__
