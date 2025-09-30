/*********************                                                        */
/*! \file Test_PhaseFixTrie.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Liam Davis
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]
 **/

#include "../PhaseFixTrie.h"

#include <cxxtest/TestSuite.h>
#include <iostream>

class TestPhaseFixTrie : public CxxTest::TestSuite
{
public:
    PhaseFixTrie *trie;

    void setUp()
    {
        TS_ASSERT( trie = new PhaseFixTrie() );
    }

    void tearDown()
    {
        delete trie;
    }

    void test_insert()
    {
        // Create 3 different unsat prefixes
        std::vector<PhaseFix> prefix1 = { { 1, true }, { 2, false }, { 3, true } };
        std::vector<PhaseFix> prefix2 = { { 1, true }, { 2, false }, { 4, true } };
        std::vector<PhaseFix> prefix3 = { { 1, false }, { 5, true } };

        // Insert them into the trie
        TS_ASSERT( trie->insertUnsatPrefix( prefix1 ) );
        TS_ASSERT( trie->insertUnsatPrefix( prefix2 ) );
        TS_ASSERT( trie->insertUnsatPrefix( prefix3 ) );

        // Re-inserting the same prefixes should return false
        TS_ASSERT( !trie->insertUnsatPrefix( prefix1 ) );
        TS_ASSERT( !trie->insertUnsatPrefix( prefix2 ) );
        TS_ASSERT( !trie->insertUnsatPrefix( prefix3 ) );
    }

    void test_dump()
    {
        // Create test unsat prefixes
        std::vector<PhaseFix> prefix1 = { { 1, true }, { 2, false }, { 3, true } };
        std::vector<PhaseFix> prefix2 = { { 1, true }, { 2, false }, { 4, true } };
        std::vector<PhaseFix> prefix3 = { { 1, false }, { 5, true } };

        // Insert prefixes into the trie
        trie->insertUnsatPrefix( prefix1 );
        trie->insertUnsatPrefix( prefix2 );
        trie->insertUnsatPrefix( prefix3 );

        // Dump and get the unsat prefixes
        std::vector<std::vector<PhaseFix>> dumpedPrefixes = trie->dumpUnsatPrefixes();

        // Verify we got the correct number of prefixes
        TS_ASSERT_EQUALS( dumpedPrefixes.size(), 3 );

        // Helper function to check if a prefix exists in the dumped results
        auto findPrefix = [&]( const std::vector<PhaseFix> &target ) -> bool {
            for ( const auto &dumped : dumpedPrefixes )
            {
                if ( dumped.size() == target.size() )
                {
                    bool match = true;
                    for ( size_t i = 0; i < target.size(); ++i )
                    {
                        if ( dumped[i] != target[i] )
                        {
                            match = false;
                            break;
                        }
                    }
                    if ( match )
                        return true;
                }
            }
            return false;
        };

        // Verify all original prefixes are in the dumped results
        TS_ASSERT( findPrefix( prefix1 ) );
        TS_ASSERT( findPrefix( prefix2 ) );
        TS_ASSERT( findPrefix( prefix3 ) );
    }

    void test_shortest_prefix_invariance()
    {
        // 1. A short, unique UNSAT prefix
        std::vector<PhaseFix> prefix_short = { { 10, true }, { 20, false } };

        // 2. A prefix that extends the short one
        std::vector<PhaseFix> prefix_long = { { 10, true }, { 20, false }, { 30, true } };

        // 3. A completely independent prefix
        std::vector<PhaseFix> prefix_independent = { { 10, false }, { 40, true } };

        // Insert the short prefix first
        TS_ASSERT( trie->insertUnsatPrefix( prefix_short ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        // Attempt to insert the longer prefix (should be rejected/return false)
        // because its prefix (prefix_short) is already a known unsat root.
        TS_ASSERT( !trie->insertUnsatPrefix( prefix_long ) );
        TS_ASSERT_EQUALS( trie->size(), 1 ); // Size must not change

        // Insert the independent prefix (should succeed)
        TS_ASSERT( trie->insertUnsatPrefix( prefix_independent ) );
        TS_ASSERT_EQUALS( trie->size(), 2 );

        // Extract and verify only the two shortest prefixes were stored
        std::vector<std::vector<PhaseFix>> dumpedPrefixes = trie->dumpUnsatPrefixes();
        TS_ASSERT_EQUALS( dumpedPrefixes.size(), 2 );

        // Helper function
        auto findPrefix = [&]( const std::vector<PhaseFix> &target ) -> bool {
            for ( const auto &dumped : dumpedPrefixes )
            {
                if ( dumped == target )
                    return true;
            }
            return false;
        };

        TS_ASSERT( findPrefix( prefix_short ) );
        TS_ASSERT( findPrefix( prefix_independent ) );
        TS_ASSERT( !findPrefix( prefix_long ) ); // MUST NOT be present
    }

    void test_subsumption_order_dependence()
    {
        // 1. A prefix that will be subsumed
        std::vector<PhaseFix> prefix_long = { { 100, true }, { 200, false }, { 300, true } };

        // 2. The shorter prefix that subsumes it (must be inserted later)
        std::vector<PhaseFix> prefix_short = { { 100, true }, { 200, false } };

        // Insert the LONGER prefix first
        TS_ASSERT( trie->insertUnsatPrefix( prefix_long ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        // Attempt to insert the SHORTER prefix (must succeed and override the longer path)
        TS_ASSERT( trie->insertUnsatPrefix( prefix_short ) );
        TS_ASSERT_EQUALS( trie->size(), 1 ); // The shorter prefix subsumes the longer one.

        // Extract and verify only the shortest one is extracted.
        std::vector<std::vector<PhaseFix>> dumpedPrefixes = trie->dumpUnsatPrefixes();
        TS_ASSERT_EQUALS( dumpedPrefixes.size(), 1 );

        // Helper function
        auto findPrefix = [&]( const std::vector<PhaseFix> &target ) -> bool {
            for ( const auto &dumped : dumpedPrefixes )
            {
                if ( dumped == target )
                    return true;
            }
            return false;
        };

        TS_ASSERT( findPrefix( prefix_short ) );
        TS_ASSERT( !findPrefix( prefix_long ) ); // The shorter one should be returned
    }

    void test_empty_and_single_fix_prefixes()
    {
        // 1. Single fix prefix
        std::vector<PhaseFix> prefix_single = { { 50, false } };

        // 2. Empty prefix (UNSAT before any decisions)
        std::vector<PhaseFix> prefix_empty = {};

        // 3. Independent two-fix prefix
        std::vector<PhaseFix> prefix_two = { { 50, true }, { 60, true } };

        // Insert single-fix
        TS_ASSERT( trie->insertUnsatPrefix( prefix_single ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        // Insert two-fix
        TS_ASSERT( trie->insertUnsatPrefix( prefix_two ) );
        TS_ASSERT_EQUALS( trie->size(), 2 );

        // Insert empty prefix (should succeed and subsume all others)
        TS_ASSERT( trie->insertUnsatPrefix( prefix_empty ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        // Attempt to insert an extension of prefix_single (should fail as it's subsumed by empty)
        std::vector<PhaseFix> prefix_extension = { { 50, false }, { 70, false } };
        TS_ASSERT( !trie->insertUnsatPrefix( prefix_extension ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        // Extract and verify only the empty prefix is present
        std::vector<std::vector<PhaseFix>> dumpedPrefixes = trie->dumpUnsatPrefixes();
        TS_ASSERT_EQUALS( dumpedPrefixes.size(), 1 );

        // Helper function
        auto findPrefix = [&]( const std::vector<PhaseFix> &target ) -> bool {
            for ( const auto &dumped : dumpedPrefixes )
            {
                if ( dumped == target )
                    return true;
            }
            return false;
        };

        TS_ASSERT( !findPrefix( prefix_single ) );
        TS_ASSERT( !findPrefix( prefix_two ) );
        TS_ASSERT( findPrefix( prefix_empty ) );
        TS_ASSERT( !findPrefix( prefix_extension ) ); // Extension should be excluded
    }

    void test_order_invariance()
    {
        // Two prefixes with the same fixes but different order
        std::vector<PhaseFix> prefix_A = { { 1, true }, { 2, false }, { 3, true } };
        std::vector<PhaseFix> prefix_B = { { 3, true }, { 1, true }, { 2, false } };

        // Insert the first one, should succeed
        TS_ASSERT( trie->insertUnsatPrefix( prefix_A ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        // Insert the second one (same set of fixes), should return false as it's a duplicate
        TS_ASSERT( !trie->insertUnsatPrefix( prefix_B ) );
        TS_ASSERT_EQUALS( trie->size(), 1 ); // Size should not change

        // Dump prefixes
        std::vector<std::vector<PhaseFix>> dumpedPrefixes = trie->dumpUnsatPrefixes();
        TS_ASSERT_EQUALS( dumpedPrefixes.size(), 1 );

        // The dumped prefix should be the sorted version
        std::vector<PhaseFix> sorted_prefix = { { 1, true }, { 2, false }, { 3, true } };
        TS_ASSERT_EQUALS( dumpedPrefixes[0], sorted_prefix );
    }
};