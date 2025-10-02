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

    void test_subset_non_prefix_subsumption()
    {
        std::vector<PhaseFix> superset = { { 1, true }, { 2, true }, { 3, true }, { 4, true } };
        std::vector<PhaseFix> subset = { { 1, true }, { 2, true }, { 4, true } };

        TS_ASSERT( trie->insertUnsatPrefix( superset ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        // Insert subset that is not a path prefix (skips 3)
        TS_ASSERT( trie->insertUnsatPrefix( subset ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        auto dumped = trie->dumpUnsatPrefixes();
        TS_ASSERT_EQUALS( dumped.size(), 1 );
        TS_ASSERT( dumped[0] == subset );
    }

    void test_multiple_incomparable_prefixes()
    {
        // Three pairwise incomparable (no subset relation) minimal UNSAT sets
        std::vector<PhaseFix> a = { { 1, true }, { 4, false } };
        std::vector<PhaseFix> b = { { 2, false }, { 5, true } };
        std::vector<PhaseFix> c = { { 3, true }, { 6, false } };

        TS_ASSERT( trie->insertUnsatPrefix( a ) );
        TS_ASSERT( trie->insertUnsatPrefix( b ) );
        TS_ASSERT( trie->insertUnsatPrefix( c ) );

        TS_ASSERT_EQUALS( trie->size(), 3 );

        auto dumped = trie->dumpUnsatPrefixes();
        TS_ASSERT_EQUALS( dumped.size(), 3 );

        auto contains = [&]( const std::vector<PhaseFix> &t ) {
            for ( const auto &d : dumped )
                if ( d == t )
                    return true;
            return false;
        };

        TS_ASSERT( contains( a ) );
        TS_ASSERT( contains( b ) );
        TS_ASSERT( contains( c ) );
    }

    void test_subsumption_retains_other_minimals()
    {
        // Start with two incomparable larger sets
        std::vector<PhaseFix> s1 = { { 1, true }, { 2, true }, { 10, false } };
        std::vector<PhaseFix> s2 = { { 3, false }, { 4, true }, { 11, true } };

        TS_ASSERT( trie->insertUnsatPrefix( s1 ) );
        TS_ASSERT( trie->insertUnsatPrefix( s2 ) );
        TS_ASSERT_EQUALS( trie->size(), 2 );

        // Insert subset of s1 only
        std::vector<PhaseFix> s1_min = { { 1, true }, { 2, true } };
        TS_ASSERT( trie->insertUnsatPrefix( s1_min ) );

        // Now s1 should be removed, s1_min + s2 remain
        TS_ASSERT_EQUALS( trie->size(), 2 );

        auto dumped = trie->dumpUnsatPrefixes();
        TS_ASSERT_EQUALS( dumped.size(), 2 );

        auto contains = [&]( const std::vector<PhaseFix> &t ) {
            for ( const auto &d : dumped )
                if ( d == t )
                    return true;
            return false;
        };

        TS_ASSERT( contains( s1_min ) );
        TS_ASSERT( contains( s2 ) );
        TS_ASSERT( !contains( s1 ) );
    }

    void test_chain_of_subsumptions()
    {
        // Insert a long set first
        std::vector<PhaseFix> L3 = { { 1, true }, { 2, false }, { 3, true } };
        TS_ASSERT( trie->insertUnsatPrefix( L3 ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        // Insert a middle subset
        std::vector<PhaseFix> L2 = { { 1, true }, { 3, true } };
        TS_ASSERT( trie->insertUnsatPrefix( L2 ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        // Insert the smallest subset
        std::vector<PhaseFix> L1 = { { 1, true } };
        TS_ASSERT( trie->insertUnsatPrefix( L1 ) );
        TS_ASSERT_EQUALS( trie->size(), 1 );

        auto dumped = trie->dumpUnsatPrefixes();
        TS_ASSERT_EQUALS( dumped.size(), 1 );
        TS_ASSERT( dumped[0] == L1 );
    }

    void test_hasUnsatSubset_basic_same_trie()
    {
        // Insert some UNSAT prefixes
        std::vector<PhaseFix> prefix1 = { { 1, true }, { 2, false } };
        std::vector<PhaseFix> prefix2 = { { 3, true }, { 4, false } };
        std::vector<PhaseFix> prefix3 = { { 5, true } };

        trie->insertUnsatPrefix( prefix1 );
        trie->insertUnsatPrefix( prefix2 );
        trie->insertUnsatPrefix( prefix3 );

        // Test exact matches (basic subset - identical to stored prefixes)
        TS_ASSERT( trie->hasUnsatSubset( prefix1 ) );
        TS_ASSERT( trie->hasUnsatSubset( prefix2 ) );
        TS_ASSERT( trie->hasUnsatSubset( prefix3 ) );

        // Test with additional elements (supersets should still match)
        std::vector<PhaseFix> superset1 = { { 1, true }, { 2, false }, { 6, true } };
        std::vector<PhaseFix> superset2 = { { 3, true }, { 4, false }, { 7, false }, { 8, true } };
        std::vector<PhaseFix> superset3 = { { 5, true }, { 9, false } };

        TS_ASSERT( trie->hasUnsatSubset( superset1 ) );
        TS_ASSERT( trie->hasUnsatSubset( superset2 ) );
        TS_ASSERT( trie->hasUnsatSubset( superset3 ) );
    }

    void test_hasUnsatSubset_simple_sequential_subset()
    {
        // Insert UNSAT prefixes that are sequential paths in the trie
        std::vector<PhaseFix> path1 = { { 1, true }, { 2, false }, { 3, true } };
        std::vector<PhaseFix> path2 = { { 10, false }, { 11, true } };

        trie->insertUnsatPrefix( path1 );
        trie->insertUnsatPrefix( path2 );

        // Test simple sequential subsets (prefixes of stored paths)
        std::vector<PhaseFix> prefix_of_path1 = { { 1, true }, { 2, false } };
        std::vector<PhaseFix> single_from_path1 = { { 1, true } };
        std::vector<PhaseFix> prefix_of_path2 = { { 10, false } };

        // These should find the stored UNSAT paths as supersets
        std::vector<PhaseFix> test1 = { { 1, true }, { 2, false }, { 3, true }, { 4, false } };
        std::vector<PhaseFix> test2 = { { 10, false }, { 11, true }, { 12, true } };

        TS_ASSERT( trie->hasUnsatSubset( test1 ) );
        TS_ASSERT( trie->hasUnsatSubset( test2 ) );

        // Test with mixed elements
        std::vector<PhaseFix> mixed1 = { { 1, true }, { 2, false }, { 3, true }, { 10, false } };
        TS_ASSERT( trie->hasUnsatSubset( mixed1 ) ); // Contains path1
    }

    void test_hasUnsatSubset_complex_non_sequential_subset()
    {
        // Insert UNSAT prefixes
        std::vector<PhaseFix> unsat1 = { { 1, true }, { 3, false }, { 5, true } };
        std::vector<PhaseFix> unsat2 = { { 2, false }, { 4, true } };
        std::vector<PhaseFix> unsat3 = { { 7, true }, { 9, false }, { 11, true }, { 13, false } };

        trie->insertUnsatPrefix( unsat1 );
        trie->insertUnsatPrefix( unsat2 );
        trie->insertUnsatPrefix( unsat3 );

        // Test complex non-sequential subsets (scattered elements)
        std::vector<PhaseFix> scattered1 = { { 0, false }, { 1, true }, { 2, true }, { 3, false },
                                             { 4, false }, { 5, true }, { 6, true } };
        TS_ASSERT( trie->hasUnsatSubset( scattered1 ) ); // Contains unsat1: {1,true}, {3,false},
                                                         // {5,true}

        std::vector<PhaseFix> scattered2 = {
            { 1, false }, { 2, false }, { 3, true }, { 4, true }, { 6, false }
        };
        TS_ASSERT( trie->hasUnsatSubset( scattered2 ) ); // Contains unsat2: {2,false}, {4,true}

        std::vector<PhaseFix> scattered3 = { { 5, false },  { 7, true },   { 8, false },
                                             { 9, false },  { 10, true },  { 11, true },
                                             { 12, false }, { 13, false }, { 14, true } };
        TS_ASSERT( trie->hasUnsatSubset( scattered3 ) ); // Contains unsat3: {7,true}, {9,false},
                                                         // {11,true}, {13,false}

        // Test with multiple possible matches - should return true if any subset matches
        std::vector<PhaseFix> multi_match = {
            { 1, true }, { 2, false }, { 3, false }, { 4, true }, { 5, true }
        };
        TS_ASSERT( trie->hasUnsatSubset( multi_match ) ); // Contains both unsat1 and unsat2
    }

    void test_hasUnsatSubset_no_subset()
    {
        // Insert UNSAT prefixes
        std::vector<PhaseFix> unsat1 = { { 1, true }, { 2, false }, { 3, true } };
        std::vector<PhaseFix> unsat2 = { { 10, false }, { 11, true } };
        std::vector<PhaseFix> unsat3 = { { 20, true }, { 21, false }, { 22, true }, { 23, false } };

        trie->insertUnsatPrefix( unsat1 );
        trie->insertUnsatPrefix( unsat2 );
        trie->insertUnsatPrefix( unsat3 );

        // Test cases where no subset exists

        // Completely disjoint variables
        std::vector<PhaseFix> disjoint = { { 100, true }, { 101, false }, { 102, true } };
        TS_ASSERT( !trie->hasUnsatSubset( disjoint ) );

        // Partial matches but not complete subsets
        std::vector<PhaseFix> partial1 = { { 1, true }, { 2, false } }; // Missing {3,true} from
                                                                        // unsat1
        TS_ASSERT( !trie->hasUnsatSubset( partial1 ) );

        std::vector<PhaseFix> partial2 = { { 10, false } }; // Missing {11,true} from unsat2
        TS_ASSERT( !trie->hasUnsatSubset( partial2 ) );

        // Wrong phase values
        std::vector<PhaseFix> wrong_phase1 = { { 1, false }, { 2, false }, { 3, true } }; // Wrong
                                                                                          // phases
                                                                                          // for 1,2
        TS_ASSERT( !trie->hasUnsatSubset( wrong_phase1 ) );

        std::vector<PhaseFix> wrong_phase2 = { { 10, true }, { 11, true } }; // Wrong phase for 10
        TS_ASSERT( !trie->hasUnsatSubset( wrong_phase2 ) );

        // Mixed correct and incorrect elements
        std::vector<PhaseFix> mixed_wrong = {
            { 1, true }, { 2, false }, { 3, false }, { 10, false }, { 11, false }
        };
        TS_ASSERT( !trie->hasUnsatSubset( mixed_wrong ) ); // Has partial matches but no complete
                                                           // subset
    }

    void test_hasUnsatSubset_edge_cases()
    {
        // Test empty trie
        std::vector<PhaseFix> any_fixes = { { 1, true }, { 2, false } };
        TS_ASSERT( !trie->hasUnsatSubset( any_fixes ) );

        // Insert empty prefix (global UNSAT)
        std::vector<PhaseFix> empty_prefix = {};
        trie->insertUnsatPrefix( empty_prefix );

        // Any input should return true when empty set is UNSAT
        TS_ASSERT( trie->hasUnsatSubset( any_fixes ) );
        TS_ASSERT( trie->hasUnsatSubset( {} ) );

        std::vector<PhaseFix> large_input = {
            { 1, true }, { 2, false }, { 3, true }, { 4, false }, { 5, true }
        };
        TS_ASSERT( trie->hasUnsatSubset( large_input ) );

        // Reset trie for next test
        delete trie;
        trie = new PhaseFixTrie();

        // Test single element UNSAT prefix
        std::vector<PhaseFix> single = { { 42, true } };
        trie->insertUnsatPrefix( single );

        TS_ASSERT( trie->hasUnsatSubset( single ) );
        TS_ASSERT( trie->hasUnsatSubset( { { 42, true }, { 43, false } } ) );
        TS_ASSERT( !trie->hasUnsatSubset( { { 42, false } } ) );
        TS_ASSERT( !trie->hasUnsatSubset( { { 43, true } } ) );

        // Test with empty input
        TS_ASSERT( !trie->hasUnsatSubset( {} ) );

        // Test with duplicate elements in input (should be handled by canonicalization)
        std::vector<PhaseFix> with_duplicates = {
            { 42, true }, { 42, true }, { 43, false }, { 42, true }
        };
        TS_ASSERT( trie->hasUnsatSubset( with_duplicates ) );

        // Test with unsorted input (should be handled by canonicalization)
        std::vector<PhaseFix> unsorted = { { 50, false }, { 42, true }, { 45, true } };
        TS_ASSERT( trie->hasUnsatSubset( unsorted ) );
    }

    void test_hasUnsatSubset_complex_trie_structure()
    {
        // Build a more complex trie with overlapping paths
        std::vector<PhaseFix> path1 = { { 1, true }, { 2, false }, { 3, true } };
        std::vector<PhaseFix> path2 = { { 1, true }, { 2, false }, { 4, false } };
        std::vector<PhaseFix> path3 = { { 1, true }, { 5, true } };
        std::vector<PhaseFix> path4 = { { 6, false }, { 7, true }, { 8, false } };

        trie->insertUnsatPrefix( path1 );
        trie->insertUnsatPrefix( path2 );
        trie->insertUnsatPrefix( path3 );
        trie->insertUnsatPrefix( path4 );

        // Test that finds the shortest matching path (path3 is shortest for {1,true})
        std::vector<PhaseFix> test_shortest = { { 1, true },  { 2, false }, { 3, true },
                                                { 4, false }, { 5, true },  { 9, false } };
        TS_ASSERT( trie->hasUnsatSubset( test_shortest ) ); // Should match path3: {1,true},
                                                            // {5,true}

        // Test multiple potential matches
        std::vector<PhaseFix> multi_potential = { { 1, true },  { 2, false }, { 3, true },
                                                  { 6, false }, { 7, true },  { 8, false } };
        TS_ASSERT( trie->hasUnsatSubset( multi_potential ) ); // Could match path1 or path4

        // Test that requires backtracking in search
        std::vector<PhaseFix> backtrack_test = {
            { 1, true }, { 2, false }, { 4, false }, { 6, false }, { 7, true }
        };
        TS_ASSERT( trie->hasUnsatSubset( backtrack_test ) ); // Should find path2: {1,true},
                                                             // {2,false}, {4,false}

        // Test case that should fail despite having many matching elements
        std::vector<PhaseFix> almost_match = {
            { 1, true }, { 2, false }, { 3, false }, { 4, true }, { 5, false }
        };
        TS_ASSERT( !trie->hasUnsatSubset( almost_match ) ); // No complete path matches
    }
};