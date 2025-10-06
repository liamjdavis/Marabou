/*********************                                                        */
/*! \file CuttingPlane.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Liam Davisd
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** See the description of the class in PiecewiseLinearConstraint.h.

**/

#include <utility>
#include <vector>

#ifndef __CuttingPlane_h__
#define __CuttingPlane_h__

using PhaseFix = std::pair<unsigned, bool>; // <variable index, phase>

struct CuttingPlane
{
    // Fixed neurons split into active and inactive
    std::vector<PhaseFix> inactiveNeurons;
    std::vector<PhaseFix> activeNeurons;

    // Right hand side of the cut |Z+| - 1
    int rhs;

    CuttingPlane()
        : rhs( 0 )
    {
    }
};

#endif // __CuttingPlane_h__