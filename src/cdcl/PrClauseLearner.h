/*********************                                                        */
/*! \file PrClauseLearner.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Liam Davis
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** Harvests propagation-redundancy (PR) clause candidates from conditional
 ** autarkies of the learned clause pool, following the CAUTICAL construction
 ** (Shah et al., FMCAD 2025). A trail (the current assignment of boolean
 ** abstraction literals) is carved into a condition part - literals appearing
 ** in some pool clause that is touched but not yet satisfied - and an autarky
 ** part - the remaining trail literals, which satisfy every pool clause they
 ** touch. For each autarky literal a with condition c_1, ..., c_k, the PR
 ** clause (-c_1 v ... v -c_k v -a) preserves satisfiability of the pool.
 **
 ** Note that the pool is only the boolean clause set, not the full
 ** verification problem; the harvested clauses are injected as a search
 ** heuristic and are not justified in produced proofs.
 **/

#ifndef __PrClauseLearner_h__
#define __PrClauseLearner_h__

#include "List.h"
#include "Map.h"
#include "Set.h"
#include "Vector.h"

class PrClauseLearner
{
public:
    PrClauseLearner();

    /*
      Start/stop recording pool clauses and trail observations.
    */
    void startHarvest();
    void stopHarvest();
    bool isHarvesting() const;

    /*
      Mirror a clause already in disjunction form (e.g., an initial NAP clause).
    */
    void addPoolClause( const Set<int> &clause );

    /*
      Mirror a learned conflict recorded as a conflicting assignment (cube);
      the pooled clause is its negation.
    */
    void addPoolClauseFromCube( const Set<int> &cube );

    /*
      The mirrored clause pool, in disjunction form.
    */
    const Vector<Set<int>> &getPoolClauses() const;

    /*
      Carve the given trail into condition and autarky parts against the
      current pool, and record one PR clause candidate per autarky literal.
    */
    void observeTrail( const Vector<int> &trail );

    /*
      Stop harvesting, drop conditions subsumed within their autarky bucket,
      and return the harvested PR clauses in disjunction form.
    */
    List<Set<int>> finalizeHarvest();

    unsigned getNumObservedTrails() const;
    unsigned getNumHarvestedCandidates() const;

private:
    bool _harvesting;
    Vector<Set<int>> _pool;

    // Autarky literal -> distinct conditions it was harvested under
    Map<int, List<Set<int>>> _harvest;

    unsigned _numObservedTrails;
    unsigned _numHarvestedCandidates;
};

#endif // __PrClauseLearner_h__
