/*********************                                                        */
/*! \file LPSolver.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Liam Davis
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** Factory implementation for LPSolver.

 **/

#include "LPSolver.h"

#include "CuOptWrapper.h"
#include "GurobiWrapper.h"

LPSolver *createLPSolver( LPSolverType type )
{
    switch ( type )
    {
    case LPSolverType::GUROBI:
        return new GurobiWrapper();
    case LPSolverType::CUOPT:
        return new CuOptWrapper();
    case LPSolverType::NATIVE:
    default:
        return nullptr;
    }
}
