/*********************                                                        */
/*! \file PhaseFixTrie.h
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
#include <algorithm>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#ifndef __PhaseFixTrie_h__
#define __PhaseFixTrie_h__

using PhaseFix = std::pair<unsigned, bool>; // <variable index, phase>

struct TrieNode
{
    // Key: next phase fix in sequence, Value: pointer to child node
    std::map<PhaseFix, TrieNode *> children;

    // Flag to mark if this node is the root of an UNSAT subtree
    bool isUnsatRoot = false;

    TrieNode() = default;

    ~TrieNode()
    {
        for ( auto const &[key, val] : children )
            delete val;
    }

    // Disable copy constructor and assignment operator
    TrieNode( const TrieNode & ) = delete;
    TrieNode &operator=( const TrieNode & ) = delete;
};

class PhaseFixTrie
{
public:
    PhaseFixTrie()
        : _root( new TrieNode() )
    {
    }
    ~PhaseFixTrie()
    {
        if ( _root )
            delete _root;
    }

    TrieNode *getRoot() const
    {
        return _root;
    }

    // Disable copy constructor and assignment operator
    PhaseFixTrie( const PhaseFixTrie & ) = delete;
    PhaseFixTrie &operator=( const PhaseFixTrie & ) = delete;

    // Enable move constructor and move assignment operator
    PhaseFixTrie( PhaseFixTrie &&other ) noexcept
        : _root( other._root )
    {
        other._root = nullptr;
    }

    PhaseFixTrie &operator=( PhaseFixTrie &&other ) noexcept
    {
        if ( this != &other )
        {
            // Delete existing tree
            delete _root;

            // Transfer ownership
            _root = other._root;
            other._root = nullptr;
        }
        return *this;
    }

    /*
        Insert a new unsat prefix to the trie.
        Returns true if the prefix was new, false if it already existed.
    */
    bool insertUnsatPrefix( const std::vector<PhaseFix> &prefix )
    {
        // Canonicalize input: sort and remove duplicates
        std::vector<PhaseFix> sortedPrefix = prefix;
        std::sort( sortedPrefix.begin(), sortedPrefix.end() );
        sortedPrefix.erase( std::unique( sortedPrefix.begin(), sortedPrefix.end() ),
                            sortedPrefix.end() );

        TrieNode *curr = _root;

        // Empty set handling (global UNSAT)
        if ( sortedPrefix.empty() )
        {
            if ( !curr->isUnsatRoot )
            {
                curr->isUnsatRoot = true;
                for ( auto const &[k, v] : curr->children )
                    delete v;
                curr->children.clear();
                return true;
            }
            return false;
        }

        // If root already UNSAT, everything subsumed
        if ( _root->isUnsatRoot )
            return false;

        // Gather existing minimal UNSAT sets
        std::vector<std::vector<PhaseFix>> existing = gatherUnsatPrefixes();

        // If any existing set is subset of new → redundant
        for ( const auto &e : existing )
            if ( isSubset( e, sortedPrefix ) )
                return false;

        // Remove supersets of the new one
        std::vector<std::vector<PhaseFix>> filtered;
        filtered.reserve( existing.size() + 1 );
        for ( const auto &e : existing )
        {
            if ( !isSubset( sortedPrefix, e ) )
                filtered.push_back( e );
        }
        filtered.push_back( sortedPrefix );

        // Rebuild trie from filtered list
        rebuildFromList( filtered );

        return true;
    }

    /*
        Check if the given phase fixes are a superset of any stored UNSAT prefix.
        Returns true if phaseFixes contains all elements of at least one UNSAT prefix in the trie.
        This is the correct condition for BICCOS pruning: if current phase fixes contain
        a complete UNSAT prefix, the current subproblem is guaranteed to be UNSAT.
    */
    bool isSupersetOfUnsatPrefix( const std::vector<PhaseFix> &phaseFixes ) const
    {
        // Canonicalize input: sort and remove duplicates
        std::vector<PhaseFix> sortedFixes = phaseFixes;
        std::sort( sortedFixes.begin(), sortedFixes.end() );
        sortedFixes.erase( std::unique( sortedFixes.begin(), sortedFixes.end() ),
                           sortedFixes.end() );

        // Check if root is UNSAT
        if ( _root->isUnsatRoot )
        {
            return true;
        }

        // Start traversal from root
        std::vector<PhaseFix> currPath;
        return findUnsatSubsetRec( _root, sortedFixes, 0, currPath );
    }

    /*
        Silent collection of current minimal UNSAT sets
    */
    std::vector<std::vector<PhaseFix>> gatherUnsatPrefixes() const
    {
        std::vector<std::vector<PhaseFix>> out;
        std::vector<PhaseFix> curr;
        gatherRec( _root, curr, out );
        return out;
    }

    /*
        Dump and print the shortest unsat prefixes in the trie.
    */
    std::vector<std::vector<PhaseFix>> dumpUnsatPrefixes()
    {
        std::vector<std::vector<PhaseFix>> prefixes;
        std::vector<PhaseFix> currPrefix;

        collectUnsatPrefixes( _root, currPrefix, prefixes );

        return prefixes;
    }

    /*
        Returns the number of unsat prefixes stored in the trie.
    */
    unsigned size() const
    {
        return countUnsatRoots( _root );
    }

private:
    TrieNode *_root;

    /*
        Recursive helper function to collect unsat prefixes.
    */
    void collectUnsatPrefixes( TrieNode *node,
                               std::vector<PhaseFix> &currPrefix,
                               std::vector<std::vector<PhaseFix>> &prefixes )
    {
        if ( node->isUnsatRoot )
        {
            prefixes.push_back( currPrefix );

            // Don't traverse children of unsat roots - they are subsumed
            return;
        }

        for ( const auto &[phaseFix, child] : node->children )
        {
            currPrefix.push_back( phaseFix );
            collectUnsatPrefixes( child, currPrefix, prefixes );

            // Backtrack
            currPrefix.pop_back();
        }
    }

    /*
        Counts the number of unsat roots in the trie.
    */
    unsigned countUnsatRoots( TrieNode *node ) const
    {
        unsigned count = 0;

        if ( node->isUnsatRoot )
        {
            return 1; // This node counts as 1, don't count children
        }

        for ( const auto &[phaseFix, child] : node->children )
        {
            count += countUnsatRoots( child );
        }

        return count;
    }

    /*
        Helper to check if A is a subset of B.
    */
    static bool isSubset( const std::vector<PhaseFix> &A, const std::vector<PhaseFix> &B )
    {
        // Two-pointer walk
        size_t i = 0, j = 0;
        while ( i < A.size() && j < B.size() )
        {
            if ( A[i] == B[j] )
            {
                ++i;
                ++j;
            }
            else if ( A[i] < B[j] )
            {
                // A[i] not found in B
                return false;
            }
            else
            {
                // Skip extra element in B
                ++j;
            }
        }
        return i == A.size();
    }

    /*
        Helper method for isSupersetOfUnsatPrefix to recursively search.
        Traverses the trie following paths that exist in phaseFixes, checking if we can
        reach an isUnsatRoot node (meaning phaseFixes contains a complete UNSAT prefix).
    */
    bool findUnsatSubsetRec( TrieNode *node,
                             const std::vector<PhaseFix> &phaseFixes,
                             size_t index,
                             std::vector<PhaseFix> &currPath ) const
    {
        // If the current node is an UNSAT root, we've found a subset
        if ( node->isUnsatRoot )
        {
            return true;
        }

        // Try remaining phase fixes from the input
        for ( size_t i = index; i < phaseFixes.size(); ++i )
        {
            const PhaseFix &pf = phaseFixes[i];

            // Check if the phase fix has a corresponding child node
            auto it = node->children.find( pf );
            if ( it != node->children.end() )
            {
                // Follow this path
                currPath.push_back( pf );

                if ( findUnsatSubsetRec( it->second, phaseFixes, i + 1, currPath ) )
                {
                    return true;
                }

                // Backtrack
                currPath.pop_back();
            }
        }

        return false;
    }

    void gatherRec( TrieNode *node,
                    std::vector<PhaseFix> &curr,
                    std::vector<std::vector<PhaseFix>> &out ) const
    {
        if ( node->isUnsatRoot )
        {
            out.push_back( curr );
            return;
        }
        for ( const auto &[pf, child] : node->children )
        {
            curr.push_back( pf );
            gatherRec( child, curr, out );
            curr.pop_back();
        }
    }

    // Raw insertion, assumes sorted unique vector, no subsumption checks
    void insertPath( const std::vector<PhaseFix> &path )
    {
        if ( path.empty() )
        {
            _root->isUnsatRoot = true;

            // Clear children since empty set subsumes everything
            for ( auto const &[k, v] : _root->children )
                delete v;
            _root->children.clear();
            return;
        }

        TrieNode *curr = _root;
        for ( const auto &pf : path )
        {
            auto it = curr->children.find( pf );
            if ( it == curr->children.end() )
            {
                TrieNode *n = new TrieNode();
                curr->children.emplace( pf, n );
                curr = n;
            }
            else
                curr = it->second;
        }
        curr->isUnsatRoot = true;

        // Prune descendants
        for ( auto const &[k, v] : curr->children )
            delete v;
        curr->children.clear();
    }

    void clearTrie( TrieNode *node )
    {
        for ( auto const &[k, v] : node->children )
            delete v;
        node->children.clear();
        node->isUnsatRoot = false;
    }

    void deleteTrie( TrieNode *node )
    {
        delete node;
    }

    void rebuildFromList( const std::vector<std::vector<PhaseFix>> &sets )
    {
        deleteTrie( _root );
        _root = new TrieNode();

        for ( const auto &s : sets )
            insertPath( s );
    }
};

#endif // __PhaseFixTrie_h__