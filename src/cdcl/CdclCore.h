/*********************                                                        */
/*! \file CdclCore.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Idan Refaeli, Omri Isac
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2025 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#ifndef __CdclCore_h__
#define __CdclCore_h__

#ifdef BUILD_CADICAL

#include "CadicalWrapper.h"
#include "IEngine.h"
#include "InputQuery.h"
#include "PLConstraintScoreTracker.h"
#include "Pair.h"
#include "PiecewiseLinearConstraint.h"
#include "Statistics.h"
#include "context/cdhashmap.h"
#include "context/cdhashset.h"

#include <cadical.hpp>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#define CDCL_LOG( x, ... ) LOG( GlobalConfiguration::CDCL_LOGGING, "CDCL: %s\n", x )

using CVC4::context::Context;

typedef Set<int> Clause;

class CdclCore
    : CaDiCaL::ExternalPropagator
    , CaDiCaL::Terminator
    , CaDiCaL::FixedAssignmentListener
{
public:
    explicit CdclCore( IEngine *engine );
    ~CdclCore() override;

    /*
      Have the CDCL core start reporting statistics.
    */
    void setStatistics( Statistics *statistics );

    /*
      Initializes the boolean abstraction for a PiecewiseLinearConstraint object
    */
    void initBooleanAbstraction( PiecewiseLinearConstraint *plc );

    /*
      Install the implication skeleton's entailed clauses; they are added to
      the SAT solver when solving starts.
    */
    void setSkeletonClauses( const Vector<Set<int>> &clauses )
    {
        _skeletonClauses = clauses;

        // Index the skeleton for first-order vivification: units, and the
        // direct implication adjacency (binary {a,b} = edges -a -> b, -b -> a).
        _skeletonUnits.clear();
        _skeletonImplied.clear();
        for ( const Set<int> &clause : clauses )
        {
            if ( clause.size() == 1 )
                _skeletonUnits.insert( *clause.begin() );
            else if ( clause.size() == 2 )
            {
                auto it = clause.begin();
                int a = *it++;
                int b = *it;
                _skeletonImplied[-a].insert( b );
                _skeletonImplied[-b].insert( a );
            }
        }
    }

    /*
      Per-pin facts from the skeleton probe pass, for rung-0 numeric
      vivification: the pin's bound tightenings vs the root box, stored
      sparsely as (variable, bound) deltas. Everything here is entailed by
      the query under the pin (Q ^ pin |= bounds).
    */
    struct ProbePinFacts
    {
        std::vector<std::pair<unsigned, double>> lbDeltas;
        std::vector<std::pair<unsigned, double>> ubDeltas;
    };

    void setProbePinFacts( const Map<int, ProbePinFacts> &pinFacts,
                           const Map<unsigned, unsigned> &cdclVarToB,
                           const std::vector<double> &rootLbs,
                           const std::vector<double> &rootUbs )
    {
        _probePinFacts = pinFacts;
        _cdclVarToB = cdclVarToB;
        _probeRootLbs = rootLbs;
        _probeRootUbs = rootUbs;
        _bToCdclVar.clear();
        for ( const auto &pair : cdclVarToB )
            _bToCdclVar[pair.second] = pair.first;
    }

    /*
      While the engine runs implication-skeleton probes, facts derived under
      a probe's pin are conditional on it - literal propagations and conflict
      clauses must not reach the SAT solver as if they were root-valid.
    */
    void setProbeMode( bool probeMode )
    {
        _probeMode = probeMode;
    }

    /*
       Push _context, record statistics
     */
    void pushContext();

    /*
       Pop _context to given level, record statistics
     */
    void popContextTo( unsigned level );

    /*
      Add valid literal to clause or zero to terminate clause.
    */
    void addLiteral( int lit );

    /*
        Calls the solving method combining the SAT solver with Marabou back engine
    */
    bool solveWithCDCL( double timeoutInSeconds );

    /**********************************************************************/
    /*  IPASIR-UP functions, for integrating Marabou with the SAT solver  */
    /**********************************************************************/

    /*
      Notify Marabou about an assignment of a boolean (abstract) variable
    */
    void notify_assignment( const std::vector<int> &lits ) override;

    /*
      Notify Marabou about a new decision level
    */
    void notify_new_decision_level() override;

    /*
      Notify Marabou should backtrack to new_level decision level
    */
    void notify_backtrack( size_t new_level ) override;

    /*
      Callback from the SAT solver that calls Marabou to check a full assignment of the boolean
      (abstract) variables
    */
    bool cb_check_found_model( const std::vector<int> &model ) override;

    /*
      Callback from the SAT solver that allows Marabou to decide a boolean (abstract) variable to
      split on
    */
    int cb_decide() override;

    /*
      Callback from the SAT solver that enables Marabou propagate literals leanred based on a
      partial assignment
     */
    int cb_propagate() override;

    /*
      Callback from the SAT solver that requires Marabou to explain a propagation.
      Returns a literal in the explanation clause one at a time, including the literal to explain.
      Ends with 0.
    */
    int cb_add_reason_clause_lit( int propagated_lit ) override;

    /*
      Check if Marabou has a conflict clause to inform the SAT solver
    */
    bool cb_has_external_clause( bool &is_forgettable ) override;
    /*
      Add conflict clause from Marabou to the SAT solver, one literal at a time. Ends with 0.
    */
    int cb_add_external_clause_lit() override;

    /*
      Internally adds a conflict clause when learned, later to be informed to the SAT solver
    */
    void addExternalClause( const Set<int> &clause, bool shareClause );

    /*
       Returns the PiecewiseLinearConstraint abstraced by the literal lit
    */
    const PiecewiseLinearConstraint *getConstraintFromLit( int lit ) const;

    /*
      Internally adds a literal, when learned, later to be informed to the SAT solver
    */
    void addLiteralToPropagate( int literal );

    /*
      Adds the decision-based conflict clause (negation of all decisions), except the given literal,
      to Marabou, later to be propagated
    */
    void addDecisionBasedConflictClause();

    /*
     Remove a literal from the propagation list
    */
    void removeLiteralFromPropagations( int literal );

    /*
      Assume valid non zero literal for next call to 'solve'.
     */
    void assume( int literal );

    /*
      Check if the solver should stop due to the requested timeout by the user
     */
    bool checkIfShouldExitDueToTimeout();

    /*
      Connected terminators are checked for termination regularly.  If the
      'terminate' function of the terminator returns true the solver is
      terminated synchronously as soon it calls this function.
     */
    bool terminate() override;

    /*
     Get the index of an assigned literal in the assigned literals list
     return the size of the list if element not found
     */
    unsigned getLiteralAssignmentIndex( int literal );

    /*
      Return true iff the literal is fixed by the SAT solver
    */
    bool isLiteralFixed( int literal ) const;

    /*
      Notifying on a fixed literal assignment.
    */
    void notify_fixed_assignment( int lit ) override;

    /*
      Notify about a single assignment
     */
    void notifySingleAssignment( int lit, bool isFixed );

    /*
      Returns true if a conflict clause exists
     */
    bool hasConflictClause() const;

    /*
      Check if the given piecewise-linear constraint is currently supported by CDCL
     */
    static bool isSupported( const PiecewiseLinearConstraint *plc );

    /*
      Initialize score stracker for pseudo-impact based decisions.
     */
    void initializeScoreTracker( std::shared_ptr<PLConstraintScoreTracker> scoreTracker );

    bool isDecision( int lit );

    void reset();

    /*
     Decision heuristics
    */
    unsigned decideSplitVarBasedOnPolarityAndVsids() const;
    unsigned decideSplitVarBasedOnPseudoImpactAndVsids() const;

    const PiecewiseLinearConstraint *getPlc( unsigned var ) const;

    void connectProofWriter( const AletheProofWriter *writer ) const;
    const Vector<int> &getSncLits() const;


    static std::atomic<unsigned> numCdclCores;

    static Map<unsigned, Set<int>> sharedClauses;
    static std::mutex sharedClausesMutex;
    static std::atomic<unsigned> clauseIndex;

private:
    /*
      The engine.
    */
    IEngine *_engine;

    /*
      Context for synchronizing the search.
     */
    Context &_context;

    /*
      Collect and print various statistics.
    */
    Statistics *_statistics;

    /*
      SAT solver object
    */
    SatSolverWrapper *_satSolver;

    /*
      Boolean abstraction map, from boolean variables to the PiecewiseLinearConstraint they
      represent
    */
    Map<unsigned, PiecewiseLinearConstraint *> _satSolverVarToPlc;

    /*
      Internal data structures to keep track of literals to propagate, assigned and fixed literals;
      and reason and conflict clauses
    */
    List<Pair<int, unsigned>> _literalsToPropagate;
    CVC4::context::CDHashMap<int, unsigned> _assignedLiterals;

    Vector<int> _reasonClauseLiterals;
    bool _isReasonClauseInitialized;

    Vector<int> _externalClauseToAdd;

    Set<int> _fixedCadicalVars;

    double _timeoutInSeconds;

    unsigned _numOfClauses;
    CVC4::context::CDHashSet<unsigned, std::hash<unsigned>> _satisfiedClauses;
    Map<int, Set<unsigned>> _literalToClauses;
    unsigned _vsidsDecayThreshold;
    unsigned _vsidsDecayCounter;

    unsigned _restarts;
    unsigned _restartLimit;
    unsigned _numOfConflictClauses;
    bool _shouldRestart;

    HashMap<unsigned, bool> _largestAssignmentSoFar;

    Vector<Set<int>> _initialClauses;

    // Entailed clauses from the implication skeleton (failed-literal units
    // and binary phase implications), installed into the SAT solver at
    // solve start. Sound by construction - no accounting needed.
    Vector<Set<int>> _skeletonClauses;

    // See setProbeMode.
    bool _probeMode = false;

    // Last heartbeat progress print (cb_check_found_model).
    struct timespec _heartbeatLastPrint = { 0, 0 };


    // Skeleton index for first-order vivification of learned clauses:
    // _skeletonUnits holds root-true literals; _skeletonImplied[x] holds the
    // direct skeleton consequences of literal x being true.
    Set<int> _skeletonUnits;
    Map<int, Set<int>> _skeletonImplied;
    unsigned _numVivifiedLiterals = 0;

    // Rung-0 numeric vivification state: per-pin sparse bound deltas, the
    // root box they are relative to, and the cdcl-var -> b-variable map for
    // the forced-sign test. See setProbePinFacts.
    Map<int, ProbePinFacts> _probePinFacts;
    Map<unsigned, unsigned> _cdclVarToB;
    std::vector<double> _probeRootLbs;
    std::vector<double> _probeRootUbs;
    unsigned _numVivifiedLiteralsBounds = 0;
    unsigned long long _vivifyTimeMicro = 0;
    // Diagnostics: how often the numeric test actually ran vs was size-skipped,
    // and the largest clause seen (distinguishes weak-oracle from wrong-gate).
    unsigned _numVivifyNumericChecks = 0;
    unsigned _numVivifySizeSkips = 0;
    unsigned _maxVivifyClauseSize = 0;

    // Rung-1 (LP-grade) vivification: learned clauses queue here and are
    // descended at the next level-0 visit (restart), when the engine is at
    // its root state. See processVivifyLpQueue.
    List<Set<int>> _vivifyLpQueue;
    bool _inVivifyLpPass = false;
    unsigned _numVivifyLpDescents = 0;
    unsigned _numVivifyLpShortened = 0;
    unsigned _numVivifyLpLiteralsRemoved = 0;
    unsigned long long _vivifyLpTimeMicro = 0;
    // Diagnostics: level-0 visits that reached the pass, clauses ever
    // queued, clauses skipped for containing a fixed literal.
    unsigned _numVivifyLpVisits = 0;
    unsigned _numVivifyLpQueued = 0;
    unsigned _numVivifyLpSkippedFixed = 0;

    // Implication edges harvested from descent pin-1 fixpoints (clausalized
    // graph growth); _seenHarvestEdges dedupes across descents, keyed by the
    // normalized literal pair. _bToCdclVar maps a pre-activation variable
    // back to its boolean var for the harvest.
    Map<unsigned, unsigned> _bToCdclVar;
    std::set<std::pair<int, int>> _seenHarvestEdges;
    unsigned _numVivifyLpHarvestedEdges = 0;

    /*
      Rung-1 theory vivification: for each queued clause, one incremental
      engine descent over the pins of the negated literals
      (most-tightening-first, per the rung-0 delta counts); infeasibility
      after j pins entails the j-literal prefix subclause, which is added
      through the normal external-clause path. Runs only at boolean level 0
      with the engine at root state; probe mode is held for the whole pass.
    */
    void processVivifyLpQueue();

    /*
      Theory-grade (rung-0) vivification: drop literal l from an entailed
      clause when pinning the negations of the remaining literals is provably
      infeasible - decided purely from probe-time facts (skeleton units,
      BCP closure over skeleton edges, and the intersection of the stored
      per-pin bound vectors: box emptiness or a forced sign on l's own b).
      No LP, no propagation. Never empties a clause.
    */
    void vivifyClause( Set<int> &clause );
    bool vivifyCandidateRemovable( const Set<int> &clause, int literal, bool &byBounds );

    std::shared_ptr<PLConstraintScoreTracker> _scoreTracker;

    Map<unsigned, int> _decisionLiterals;
    unsigned _decisionIndex;
    Map<int, double> _decisionScores;

    unsigned _lastSharedClauseIndexAdded;
    Set<unsigned> _sharedClauseAdded;

    Vector<int> _sncSplitLiterals;

    /*
      Access info in the internal data structures
    */
    bool isLiteralAssigned( int literal ) const;
    bool isLiteralToBePropagated( int literal ) const;

    bool isClauseSatisfied( unsigned clause ) const;
    unsigned int getLiteralVSIDSScore( int literal ) const;
    unsigned int getVariableVSIDSScore( unsigned var ) const;

    unsigned luby( unsigned i );

    double computeDecisionScoreForLiteral( int literal ) const;
    void setInputBoundsForLiteralInNLR( int literal,
                                        const std::shared_ptr<Query> &inputQuery,
                                        NLR::NetworkLevelReasoner *networkLevelReasoner ) const;
    void runSymbolicBoundTightening( NLR::NetworkLevelReasoner *networkLevelReasoner ) const;

    double
    getUpperBoundForOutputVariableFromNLR( NLR::NetworkLevelReasoner *networkLevelReasoner ) const;

    void computeClauseScores( const Set<int> &clause, Vector<Pair<double, int>> &clauseScores );
    void reorderByDecisionLevelIfNecessary( Vector<Pair<double, int>> &clauseScores );
    void computeShortedClause( Set<int> &clause,
                               const Vector<Pair<double, int>> &clauseScores,
                               int propagated_lit ) const;
    bool checkIfShouldSkipClauseShortening( const Set<int> &clause );

    Set<int> quickXplain( const Set<int> &currentClause,
                          const Vector<Pair<double, int>> &clauseScores,
                          unsigned int startIdx,
                          unsigned int endIdx,
                          int propagated_lit ) const;

    unsigned int _index;
};

#endif
#endif // __CdclCore_h__
