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
#include <map>
#include <utility>
#include <vector>

#ifndef __PhaseFixTrie_h__
#define __PhaseFixTrie_h__

using PhaseFix = std::pair<unsigned, bool>; // <variabl index, phase>

struct TrieNode
{
    // Key: next phase fix in sequence, value: pointer to child node
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
        delete _root;
    }

    TrieNode *getRoot() const
    {
        return _root;
    }

    // Disable copy constructor and assignment operator
    PhaseFixTrie( const PhaseFixTrie & ) = delete;
    PhaseFixTrie &operator=( const PhaseFixTrie & ) = delete;

    /*
        Insert a new unsat prefix to the trie.
        Returns true if the prefix was new, false if it already existed.
    */
    bool insertUnsatPrefix( const std::vector<PhaseFix> &prefix )
    {
        // Create a sorted copy to enforce canonical order
        std::vector<PhaseFix> sortedPrefix = prefix;
        std::sort( sortedPrefix.begin(), sortedPrefix.end() );

        TrieNode *curr = _root;

        // Check if empty prefix and root is already marked
        if ( sortedPrefix.empty() )
        {
            if ( !curr->isUnsatRoot )
            {
                curr->isUnsatRoot = true;

                // An empty prefix subsumes all others. Clear children.
                for ( auto const &[key, val] : curr->children )
                    delete val;
                curr->children.clear();

                return true;
            }
            return false;
        }

        for ( const auto &phaseFix : sortedPrefix )
        {
            if ( curr->isUnsatRoot )
            {
                // This prefix is already covered by an existing UNSAT subtree
                return false;
            }

            // Find or create the next node
            if ( curr->children.find( phaseFix ) == curr->children.end() )
            {
                curr->children[phaseFix] = new TrieNode();
            }

            curr = curr->children[phaseFix];
        }

        // Reach the end, mark this node as an UNSAT root
        if ( !curr->isUnsatRoot )
        {
            curr->isUnsatRoot = true;

            // This new prefix subsumes any of its extensions. Clear children.
            for ( auto const &[key, val] : curr->children )
                delete val;
            curr->children.clear();

            return true;
        }

        // Return false
        return false;
    }

    /*
        Dump and print the shortest unsat prefixes in the trie.
    */
    std::vector<std::vector<PhaseFix>> dumpUnsatPrefixes()
    {
        std::vector<std::vector<PhaseFix>> prefixes;
        std::vector<PhaseFix> currPrefix;

        std::cout << "Dumping UNSAT prefixes from PhaseFixTrie:" << std::endl;

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

            std::cout << "Unsat Prefix: ";
            for ( const auto &[var, phase] : currPrefix )
            {
                std::cout << "(" << var << ", " << ( phase ? "T" : "F" ) << ") ";
            }
            std::cout << std::endl;

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
};

#endif // __PhaseFixTrie_h__