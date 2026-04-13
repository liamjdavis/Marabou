/*********************                                                        */
/*! \file Test_CuOptWrapper.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Liam Davis
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief [[ Add one-line brief description here ]]
 **
 ** [[ Add lengthier description here ]]
 **/

#include "CuOptWrapper.h"
#include "FloatUtils.h"
#include "MString.h"
#include "MockErrno.h"

#include <cxxtest/TestSuite.h>

class CuOptWrapperTestSuite : public CxxTest::TestSuite
{
public:
    MockErrno *mockErrno;

    void setUp()
    {
        TS_ASSERT( mockErrno = new MockErrno );
    }

    void tearDown()
    {
        TS_ASSERT_THROWS_NOTHING( delete mockErrno );
    }

    void test_optimize()
    {
#ifdef ENABLE_CUOPT
        CuOptWrapper cuopt;

        cuopt.addVariable( "x", 0, 3 );
        cuopt.addVariable( "y", 0, 3 );
        cuopt.addVariable( "z", 0, 3 );

        // x + y + z <= 5
        List<CuOptWrapper::Term> constraint = {
            CuOptWrapper::Term( 1, "x" ),
            CuOptWrapper::Term( 1, "y" ),
            CuOptWrapper::Term( 1, "z" ),
        };

        cuopt.addLeqConstraint( constraint, 5 );

        // Cost: -x - 2y + z
        List<CuOptWrapper::Term> cost = {
            CuOptWrapper::Term( -1, "x" ),
            CuOptWrapper::Term( -2, "y" ),
            CuOptWrapper::Term( +1, "z" ),
        };

        cuopt.setCost( cost );

        // Solve and extract
        TS_ASSERT_THROWS_NOTHING( cuopt.solve() );

        TS_ASSERT( cuopt.optimal() );

        Map<String, double> solution;
        double costValue;

        TS_ASSERT_THROWS_NOTHING( cuopt.extractSolution( solution, costValue ) );

        TS_ASSERT( FloatUtils::areEqual( solution["x"], 2, 1e-4 ) );
        TS_ASSERT( FloatUtils::areEqual( solution["y"], 3, 1e-4 ) );
        TS_ASSERT( FloatUtils::areEqual( solution["z"], 0, 1e-4 ) );

        TS_ASSERT( FloatUtils::areEqual( costValue, -8, 1e-4 ) );

#else
        TS_ASSERT( true );
#endif // ENABLE_CUOPT
    }

    void test_infeasible()
    {
#ifdef ENABLE_CUOPT
        CuOptWrapper cuopt;

        cuopt.addVariable( "x", 0, 1 );

        // x >= 2 (infeasible since x in [0,1])
        List<CuOptWrapper::Term> constraint = {
            CuOptWrapper::Term( 1, "x" ),
        };

        cuopt.addGeqConstraint( constraint, 2 );

        // Need an objective
        List<CuOptWrapper::Term> cost = {
            CuOptWrapper::Term( 1, "x" ),
        };
        cuopt.setCost( cost );

        TS_ASSERT_THROWS_NOTHING( cuopt.solve() );

        TS_ASSERT( cuopt.infeasible() );
        TS_ASSERT( !cuopt.optimal() );
        TS_ASSERT( !cuopt.haveFeasibleSolution() );

#else
        TS_ASSERT( true );
#endif // ENABLE_CUOPT
    }

    void test_geq_constraint()
    {
#ifdef ENABLE_CUOPT
        CuOptWrapper cuopt;

        // Minimize x subject to x >= 2, x in [0, 10]
        cuopt.addVariable( "x", 0, 10 );

        List<CuOptWrapper::Term> constraint = {
            CuOptWrapper::Term( 1, "x" ),
        };
        cuopt.addGeqConstraint( constraint, 2 );

        List<CuOptWrapper::Term> cost = {
            CuOptWrapper::Term( 1, "x" ),
        };
        cuopt.setCost( cost );

        TS_ASSERT_THROWS_NOTHING( cuopt.solve() );
        TS_ASSERT( cuopt.optimal() );

        Map<String, double> solution;
        double costValue;
        cuopt.extractSolution( solution, costValue );

        TS_ASSERT( FloatUtils::areEqual( solution["x"], 2, 1e-4 ) );
        TS_ASSERT( FloatUtils::areEqual( costValue, 2, 1e-4 ) );

#else
        TS_ASSERT( true );
#endif // ENABLE_CUOPT
    }

    void test_eq_constraint()
    {
#ifdef ENABLE_CUOPT
        CuOptWrapper cuopt;

        // Minimize x + y subject to x + y = 5, x,y in [0, 10]
        cuopt.addVariable( "x", 0, 10 );
        cuopt.addVariable( "y", 0, 10 );

        List<CuOptWrapper::Term> constraint = {
            CuOptWrapper::Term( 1, "x" ),
            CuOptWrapper::Term( 1, "y" ),
        };
        cuopt.addEqConstraint( constraint, 5 );

        List<CuOptWrapper::Term> cost = {
            CuOptWrapper::Term( 1, "x" ),
            CuOptWrapper::Term( 1, "y" ),
        };
        cuopt.setCost( cost );

        TS_ASSERT_THROWS_NOTHING( cuopt.solve() );
        TS_ASSERT( cuopt.optimal() );

        Map<String, double> solution;
        double costValue;
        cuopt.extractSolution( solution, costValue );

        TS_ASSERT( FloatUtils::areEqual( costValue, 5, 1e-4 ) );
        // x + y must equal 5
        TS_ASSERT( FloatUtils::areEqual( solution["x"] + solution["y"], 5, 1e-4 ) );

#else
        TS_ASSERT( true );
#endif // ENABLE_CUOPT
    }

    void test_maximize()
    {
#ifdef ENABLE_CUOPT
        CuOptWrapper cuopt;

        // Maximize x + 2y subject to x + y <= 5, x,y in [0, 10]
        cuopt.addVariable( "x", 0, 10 );
        cuopt.addVariable( "y", 0, 10 );

        List<CuOptWrapper::Term> constraint = {
            CuOptWrapper::Term( 1, "x" ),
            CuOptWrapper::Term( 1, "y" ),
        };
        cuopt.addLeqConstraint( constraint, 5 );

        List<CuOptWrapper::Term> objective = {
            CuOptWrapper::Term( 1, "x" ),
            CuOptWrapper::Term( 2, "y" ),
        };
        cuopt.setObjective( objective );

        TS_ASSERT_THROWS_NOTHING( cuopt.solve() );
        TS_ASSERT( cuopt.optimal() );

        Map<String, double> solution;
        double objValue;
        cuopt.extractSolution( solution, objValue );

        // Optimal: x=0, y=5, obj=10
        TS_ASSERT( FloatUtils::areEqual( solution["y"], 5, 1e-4 ) );
        TS_ASSERT( FloatUtils::areEqual( objValue, 10, 1e-4 ) );

#else
        TS_ASSERT( true );
#endif // ENABLE_CUOPT
    }
};
