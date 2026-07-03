/*********************                                                        */
/*! \file Test_PrClauseLearner.h
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

#include <cxxtest/TestSuite.h>

class PrClauseLearnerTestSuite : public CxxTest::TestSuite
{
public:
    void test_empty_pool_yields_unit_pr_clauses()
    {
        PrClauseLearner learner;
        learner.startHarvest();

        // With no pool clauses, no trail literal is load-bearing: the whole
        // trail is the autarky and every literal yields the unit clause (-a)
        Vector<int> trail = { 1, -2, 3 };
        learner.observeTrail( trail );

        List<Set<int>> clauses = learner.finalizeHarvest();
        TS_ASSERT_EQUALS( clauses.size(), 3U );
        TS_ASSERT( clauses.exists( Set<int>( { -1 } ) ) );
        TS_ASSERT( clauses.exists( Set<int>( { 2 } ) ) );
        TS_ASSERT( clauses.exists( Set<int>( { -3 } ) ) );
    }

    void test_open_clause_pulls_variables_into_condition()
    {
        PrClauseLearner learner;
        learner.startHarvest();

        // Pool clause ( 2 v 4 ): trail fixes 2 to false and leaves 4 free,
        // so the clause is touched but not satisfied - literal -2 is the
        // condition, and 1 and 3 form the autarky
        learner.addPoolClause( Set<int>( { 2, 4 } ) );

        Vector<int> trail = { 1, -2, 3 };
        learner.observeTrail( trail );

        List<Set<int>> clauses = learner.finalizeHarvest();
        TS_ASSERT_EQUALS( clauses.size(), 2U );
        TS_ASSERT( clauses.exists( Set<int>( { 2, -1 } ) ) );
        TS_ASSERT( clauses.exists( Set<int>( { 2, -3 } ) ) );
    }

    void test_satisfied_clause_leaves_variables_in_autarky()
    {
        PrClauseLearner learner;
        learner.startHarvest();

        // Pool clause ( -2 v 4 ) is satisfied by trail literal -2, so it
        // pulls nothing into the condition even though it is touched
        learner.addPoolClause( Set<int>( { -2, 4 } ) );

        Vector<int> trail = { 1, -2 };
        learner.observeTrail( trail );

        List<Set<int>> clauses = learner.finalizeHarvest();
        TS_ASSERT_EQUALS( clauses.size(), 2U );
        TS_ASSERT( clauses.exists( Set<int>( { -1 } ) ) );
        TS_ASSERT( clauses.exists( Set<int>( { 2 } ) ) );
    }

    void test_untouched_clause_is_ignored()
    {
        PrClauseLearner learner;
        learner.startHarvest();

        // Pool clause over variables entirely outside the trail
        learner.addPoolClause( Set<int>( { 7, 8 } ) );

        Vector<int> trail = { 1 };
        learner.observeTrail( trail );

        List<Set<int>> clauses = learner.finalizeHarvest();
        TS_ASSERT_EQUALS( clauses.size(), 1U );
        TS_ASSERT( clauses.exists( Set<int>( { -1 } ) ) );
    }

    void test_cube_mirroring_negates_literals()
    {
        PrClauseLearner learner;
        learner.startHarvest();

        // The conflicting assignment ( 1, -2 ) corresponds to the learned
        // clause ( -1 v 2 ), which trail literal 2 satisfies
        learner.addPoolClauseFromCube( Set<int>( { 1, -2 } ) );

        Vector<int> trail = { 1, 2 };
        learner.observeTrail( trail );

        List<Set<int>> clauses = learner.finalizeHarvest();
        TS_ASSERT_EQUALS( clauses.size(), 2U );
        TS_ASSERT( clauses.exists( Set<int>( { -1 } ) ) );
        TS_ASSERT( clauses.exists( Set<int>( { -2 } ) ) );
    }

    void test_subsumed_conditions_are_dropped()
    {
        PrClauseLearner learner;
        learner.startHarvest();

        // First trail: clause ( 2 ) is open, so condition is { -2 } and the
        // autarky literal 1 yields ( 2 v -1 )
        learner.addPoolClause( Set<int>( { 2 } ) );
        Vector<int> firstTrail = { 1, -2 };
        learner.observeTrail( firstTrail );

        // Second trail: literal 1 alone, empty condition - the unit ( -1 )
        // subsumes the earlier ( 2 v -1 )
        Vector<int> secondTrail = { 1 };
        learner.observeTrail( secondTrail );

        List<Set<int>> clauses = learner.finalizeHarvest();
        TS_ASSERT( clauses.exists( Set<int>( { -1 } ) ) );
        TS_ASSERT( !clauses.exists( Set<int>( { 2, -1 } ) ) );

        // The condition literal -2 never enters an autarky bucket
        TS_ASSERT_EQUALS( clauses.size(), 1U );
    }

    void test_no_harvest_when_inactive()
    {
        PrClauseLearner learner;

        learner.addPoolClause( Set<int>( { 1, 2 } ) );
        Vector<int> trail = { 1 };
        learner.observeTrail( trail );

        TS_ASSERT_EQUALS( learner.getNumObservedTrails(), 0U );
        TS_ASSERT( learner.getPoolClauses().empty() );
        TS_ASSERT( learner.finalizeHarvest().empty() );
    }
};
