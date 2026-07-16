/*********************                                                        */
/*! \file CdclCore.cpp
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

#ifdef BUILD_CADICAL
#include "CdclCore.h"

#include "InfeasibleQueryException.h"
#include "NetworkLevelReasoner.h"
#include "Options.h"
#include "Query.h"
#include "TimeUtils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <thread>
#include <utility>

std::atomic<unsigned> CdclCore::numCdclCores{ 0 };
Map<unsigned, Set<int>> CdclCore::sharedClauses{};
std::mutex CdclCore::sharedClausesMutex{};
std::atomic<unsigned> CdclCore::clauseIndex{ 0 };

CdclCore::CdclCore( IEngine *engine )
    : _engine( engine )
    , _context( _engine->getContext() )
    , _statistics( nullptr )
    , _satSolver( nullptr )
    , _satSolverVarToPlc()
    , _literalsToPropagate()
    , _assignedLiterals( &_context )
    , _reasonClauseLiterals()
    , _isReasonClauseInitialized( false )
    , _fixedCadicalVars()
    , _timeoutInSeconds( 0 )
    , _numOfClauses( 0 )
    , _satisfiedClauses( &_context )
    , _literalToClauses()
    , _vsidsDecayThreshold( 0 )
    , _vsidsDecayCounter( 0 )
    , _restarts( 1 )
    , _restartLimit( 512 * luby( 1 ) )
    , _numOfConflictClauses( 0 )
    , _shouldRestart( false )
    , _initialClauses()
    , _scoreTracker( nullptr )
    , _lastSharedClauseIndexAdded( 0 )
    , _sharedClauseAdded()
    , _sncSplitLiterals()
    , _index( CdclCore::numCdclCores.fetch_add( 1 ) )
{
    _satSolverVarToPlc.insert( 0, NULL );
}

CdclCore::~CdclCore()
{
    delete _satSolver;
}

void CdclCore::initBooleanAbstraction( PiecewiseLinearConstraint *plc )
{
    struct timespec start = TimeUtils::sampleMicro();

    plc->booleanAbstraction( _satSolverVarToPlc );

    if ( _statistics )
    {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_MAIN_LOOP_MICRO,
                                       TimeUtils::timePassed( start, end ) );
    }
}

bool CdclCore::isLiteralAssigned( int literal ) const
{
    if ( _assignedLiterals.count( literal ) > 0 )
    {
        ASSERT( _satSolverVarToPlc.at( abs( literal ) )->phaseFixed() ||
                !_satSolverVarToPlc.at( abs( literal ) )->isActive() )
        return true;
    }

    return false;
}

void CdclCore::notify_assignment( const std::vector<int> &lits )
{
    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
        return;

    if ( checkIfShouldExitDueToTimeout() )
        return;

    //    if ( !_externalClauseToAdd.empty() )
    //    {
    //        SEARCH_TREE_LOG( "Skipping notification due to conflict clause" )
    //        return;
    //    }

    struct timespec start = TimeUtils::sampleMicro();

    CDCL_LOG( Stringf( "%u l%d Notifying assignments:", _index, _satSolver->getLevel() ).ascii() )

    for ( int lit : lits )
    {
        CDCL_LOG( Stringf( "%u l%d\tNotified assignment %d; is decision: %d",
                           _index,
                           _satSolver->getLevel(),
                           lit,
                           isDecision( lit ) )
                      .ascii() )

        if ( !isLiteralAssigned( lit ) )
            notifySingleAssignment( lit, false );
    }

    if ( _statistics )
    {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CALLBACKS_MICRO,
                                       TimeUtils::timePassed( start, end ) );
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_NOTIFY_ASSIGNMENT_MICRO,
                                       TimeUtils::timePassed( start, end ) );
    }
}

void CdclCore::notify_new_decision_level()
{
    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
        return;

    if ( checkIfShouldExitDueToTimeout() )
        return;

    struct timespec start = TimeUtils::sampleMicro();
    CDCL_LOG(
        Stringf( "%u l%d Notified new decision level", _index, _satSolver->getLevel() ).ascii() )

    _engine->preContextPushHook();
    pushContext();

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_SPLITS );

        unsigned level = _satSolver->getLevel();
        _statistics->setUnsignedAttribute( Statistics::CURRENT_DECISION_LEVEL, level );
        if ( level > _statistics->getUnsignedAttribute( Statistics::MAX_DECISION_LEVEL ) )
            _statistics->setUnsignedAttribute( Statistics::MAX_DECISION_LEVEL, level );
        _statistics->incUnsignedAttribute( Statistics::NUM_DECISION_LEVELS );
        _statistics->incUnsignedAttribute( Statistics::SUM_DECISION_LEVELS, level );

        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CALLBACKS_MICRO,
                                       TimeUtils::timePassed( start, end ) );
        _statistics->incLongAttribute(
            Statistics::TOTAL_TIME_CDCL_CORE_NOTIFY_NEW_DECISION_LEVEL_MICRO,
            TimeUtils::timePassed( start, end ) );
    }
}

void CdclCore::notify_backtrack( size_t new_level )
{
    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
        return;

    if ( checkIfShouldExitDueToTimeout() )
        return;

    struct timespec start = TimeUtils::sampleMicro();
    CDCL_LOG(
        Stringf( "%u l%d Backtracking to level %d", _index, _satSolver->getLevel(), new_level )
            .ascii() )

    unsigned oldLevel = _satSolver->getLevel();

    if ( _shouldRestart )
    {
        if ( _statistics )
            _statistics->incUnsignedAttribute( Statistics::NUM_RESTARTS );

        _shouldRestart = false;
        _numOfConflictClauses = 0;
        _restartLimit = 512 * luby( ++_restarts );
        _engine->restoreInitialEngineState();
        _largestAssignmentSoFar.clear();
    }

    popContextTo( new_level );
    _engine->postContextPopHook();

    for ( unsigned l = oldLevel; l > new_level; l-- )
    {
        if ( l > _decisionIndex )
            continue;

        ASSERT( l == _decisionIndex )
        ASSERT( _decisionLiterals.exists( _decisionIndex ) );
        _decisionLiterals.erase( _decisionIndex-- );
    }

    // Maintain literals to propagate learned before the decision level
    List<Pair<int, unsigned>> currentPropagations = _literalsToPropagate;
    _literalsToPropagate.clear();

    for ( const Pair<int, unsigned> &propagation : currentPropagations )
        if ( propagation.second() <= new_level )
            _literalsToPropagate.append( propagation );

    for ( int lit : _fixedCadicalVars )
        if ( !isLiteralAssigned( lit ) )
            notifySingleAssignment( lit, true );

    struct timespec end = TimeUtils::sampleMicro();

    if ( _statistics )
    {
        unsigned jumpSize = oldLevel - new_level;

        _statistics->setUnsignedAttribute( Statistics::CURRENT_DECISION_LEVEL, new_level );
        _statistics->incUnsignedAttribute( Statistics::NUM_DECISION_LEVELS );
        _statistics->incUnsignedAttribute( Statistics::SUM_DECISION_LEVELS, new_level );

        _statistics->incUnsignedAttribute( Statistics::NUM_BACKJUMPS );
        _statistics->incUnsignedAttribute( Statistics::SUM_BACKJUMPS, jumpSize );
        if ( jumpSize > _statistics->getUnsignedAttribute( Statistics::MAX_BACKJUMP ) )
            _statistics->setUnsignedAttribute( Statistics::MAX_BACKJUMP, jumpSize );

        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CALLBACKS_MICRO,
                                       TimeUtils::timePassed( start, end ) );
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_NOTIFY_BACKTRACK_MICRO,
                                       TimeUtils::timePassed( start, end ) );
    }
}

bool CdclCore::cb_check_found_model( const std::vector<int> &model )
{
    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
        return false;

    if ( checkIfShouldExitDueToTimeout() )
        return false;

    if ( getenv( "CDCL_TRACE_CALLBACKS" ) )
    {
        static unsigned checkCalls = 0;
        ++checkCalls;
        if ( checkCalls == 1 || checkCalls % 100 == 0 )
        {
            printf( "TRACE cb_check_found_model calls: %u\n", checkCalls );
            fflush( stdout );
        }
    }

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES );

        // Heartbeat: report progress every 30s so long runs are observable
        // at verbosity 1 (stats blocks only print at solve end).
        if ( _engine->getVerbosity() > 0 )
        {
            struct timespec now = TimeUtils::sampleMicro();
            if ( _heartbeatLastPrint.tv_sec == 0 )
                _heartbeatLastPrint = now;
            else if ( TimeUtils::timePassed( _heartbeatLastPrint, now ) / 1e6 >= 30.0 )
            {
                _heartbeatLastPrint = now;
                printf( "CDCL progress: %u visited states, %u conflict clauses\n",
                        _statistics->getUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES ),
                        _numOfConflictClauses );
                fflush( stdout );
            }
        }
    }
    CDCL_LOG( Stringf( "%u l%d Checking model found by SAT solver", _index, _satSolver->getLevel() )
                  .ascii() )
    ASSERT( _externalClauseToAdd.empty() )
    notify_assignment( model );

    bool result;

    if ( _engine->getLpSolverType() == LPSolverType::NATIVE )
    {
        // Quickly try to notify constraints for bounds, which raises exception in case of
        // infeasibility
        if ( !_engine->propagateBoundManagerTightenings() )
            return false;

        // If external clause learned, no need to call solve
        if ( !_externalClauseToAdd.empty() )
            return false;

        result = _engine->solve( _timeoutInSeconds );

        // In cases where Marabou fails to provide a conflict clause, add the trivial possibility
        if ( !result && _externalClauseToAdd.empty() )
            addDecisionBasedConflictClause();

        CDCL_LOG(
            Stringf( "%u l%d\tResult is %u", _index, _satSolver->getLevel(), result ).ascii() )
        result = result && _externalClauseToAdd.empty();
    }
    else
        result = _engine->solve( _timeoutInSeconds );

    return result;
}

int CdclCore::cb_decide()
{
    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
        return 0;

    if ( checkIfShouldExitDueToTimeout() )
        return 0;

    struct timespec start = TimeUtils::sampleMicro();
    CDCL_LOG( Stringf( "%u l%d Callback for decision:", _index, _satSolver->getLevel() ).ascii() )

    if ( _shouldRestart )
    {
        CDCL_LOG( Stringf( "%u l%d Should restart. Forcing backtrack to level 0.",
                           _index,
                           _satSolver->getLevel() )
                      .ascii() );
        _satSolver->forceBacktrack( 0 );
        return 0;
    }

    // Rung-1 vivification runs at the amortized point: boolean level 0
    // (post-restart), engine restored to its root state, before the next
    // decision. Descents push/pop context around the root.
    if ( _satSolver->getLevel() == 0 )
        processVivifyLpQueue();

    unsigned decisionVariable =
        GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH && _satSolver->getLevel() > 3
            ? decideSplitVarBasedOnPseudoImpactAndVsids()
            : decideSplitVarBasedOnPolarityAndVsids();

    int decisionLiteral = 0;

    if ( decisionVariable )
        decisionLiteral = _satSolverVarToPlc[decisionVariable]->getLiteralForDecision();

    if ( decisionLiteral )
    {
        ASSERT( !isLiteralAssigned( -decisionLiteral ) && !isLiteralAssigned( decisionLiteral ) )
        ASSERT( FloatUtils::abs( decisionLiteral ) <= _satSolver->vars() )
        CDCL_LOG(
            Stringf( "%u l%d Decided literal %d", _index, _satSolver->getLevel(), decisionLiteral )
                .ascii() )

        if ( _statistics )
            _statistics->incUnsignedAttribute( Statistics::NUM_MARABOU_DECISIONS );
    }
    else
    {
        CDCL_LOG( Stringf( "%u l%d No decision made", _index, _satSolver->getLevel() ).ascii() )
        if ( _statistics )
            _statistics->incUnsignedAttribute( Statistics::NUM_SAT_SOLVER_DECISIONS );
    }

    if ( _statistics )
    {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CALLBACKS_MICRO,
                                       TimeUtils::timePassed( start, end ) );
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CB_DECIDE_MICRO,
                                       TimeUtils::timePassed( start, end ) );
    }

    return decisionLiteral;
}

int CdclCore::cb_propagate()
{
    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
        return 0;

    if ( checkIfShouldExitDueToTimeout() )
        return 0;

    struct timespec start = {};
    struct timespec end = {};
    unsigned long long total = 0;

    if ( _engine->getLpSolverType() == LPSolverType::GUROBI &&
         GlobalConfiguration::ANALYZE_PROOF_DEPENDENCIES )
    {
        if ( _engine->solve( _timeoutInSeconds ) )
        {
            if ( _statistics )
                start = TimeUtils::sampleMicro();

            bool allInitialClausesSatisfied = true;
            for ( const Set<int> &clause : _initialClauses )
                if ( !_engine->checkAssignmentComplianceWithClause( clause ) )
                {
                    allInitialClausesSatisfied = false;
                    break;
                }

            if ( _statistics )
            {
                end = TimeUtils::sampleMicro();
                total += TimeUtils::timePassed( start, end );
            }

            if ( allInitialClausesSatisfied )
            {
                ASSERT( _engine->getExitCode() == ExitCode::NOT_DONE );
                _engine->setExitCode( ExitCode::SAT );
                if ( _statistics )
                {
                    _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CALLBACKS_MICRO,
                                                   total );
                    _statistics->incLongAttribute(
                        Statistics::TOTAL_TIME_CDCL_CORE_CB_PROPAGATE_MICRO, total );
                }
                return 0;
            }
        }
    }

    //    ASSERT( _engine->getLpSolverType() == LPSolverType::NATIVE )

    if ( _literalsToPropagate.empty() )
    {
        if ( getenv( "CDCL_TRACE_CALLBACKS" ) )
        {
            static unsigned propagateSolves = 0;
            ++propagateSolves;
            if ( propagateSolves == 1 || propagateSolves % 100 == 0 )
            {
                printf( "TRACE cb_propagate theory-solves: %u\n", propagateSolves );
                fflush( stdout );
            }
        }

        if ( _statistics )
        {
            _statistics->incUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES );

            // Heartbeat: report progress every 30s so long runs are observable
            // at verbosity 1 (stats blocks only print at solve end). This is
            // the real theory-check site; cb_check_found_model never fires.
            if ( _engine->getVerbosity() > 0 )
            {
                struct timespec now = TimeUtils::sampleMicro();
                if ( _heartbeatLastPrint.tv_sec == 0 )
                    _heartbeatLastPrint = now;
                else if ( TimeUtils::timePassed( _heartbeatLastPrint, now ) / 1e6 >= 30.0 )
                {
                    _heartbeatLastPrint = now;
                    printf(
                        "CDCL progress: %u visited states, %u conflict clauses\n",
                        _statistics->getUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES ),
                        _numOfConflictClauses );
                    fflush( stdout );
                }
            }
        }

        // If no literals left to propagate, and no clause already found, attempt solving
        if ( _externalClauseToAdd.empty() )
        {
            if ( _engine->solve( _timeoutInSeconds ) )
            {
                if ( _statistics )
                    start = TimeUtils::sampleMicro();

                bool allInitialClausesSatisfied = true;
                for ( const Set<int> &clause : _initialClauses )
                    if ( !_engine->checkAssignmentComplianceWithClause( clause ) )
                    {
                        allInitialClausesSatisfied = false;
                        break;
                    }

                if ( _statistics )
                {
                    end = TimeUtils::sampleMicro();
                    total += TimeUtils::timePassed( start, end );
                }

                if ( allInitialClausesSatisfied )
                {
                    _engine->setExitCode( ExitCode::SAT );
                    if ( _statistics )
                    {
                        _statistics->incLongAttribute(
                            Statistics::TOTAL_TIME_CDCL_CORE_CALLBACKS_MICRO, total );
                        _statistics->incLongAttribute(
                            Statistics::TOTAL_TIME_CDCL_CORE_CB_PROPAGATE_MICRO, total );
                    }
                    return 0;
                }
            }
        }

        if ( _statistics )
            start = TimeUtils::sampleMicro();

        // Try learning a conflict clause if possible
        if ( _externalClauseToAdd.empty() )
        {
            if ( _engine->getLpSolverType() == LPSolverType::NATIVE )
                _engine->propagateBoundManagerTightenings();
            if ( _externalClauseToAdd.empty() )
            {
                if ( _assignedLiterals.size() + _literalsToPropagate.size() >
                     _largestAssignmentSoFar.size() )
                {
                    _largestAssignmentSoFar.clear();

                    for ( const auto &p : _assignedLiterals )
                    {
                        int lit = p.first;
                        if ( lit > 0 )
                            _largestAssignmentSoFar[lit] = true;
                        else
                            _largestAssignmentSoFar[-lit] = false;
                    }

                    for ( const auto &p : _literalsToPropagate )
                    {
                        int lit = p.first();
                        if ( lit > 0 )
                            _largestAssignmentSoFar[lit] = true;
                        else
                            _largestAssignmentSoFar[-lit] = false;
                    }
                }
            }
            else
                _literalsToPropagate.clear();
        }
        else
            _literalsToPropagate.clear();

        // Add the zero literal at the end
        _literalsToPropagate.append( Pair<int, unsigned>( 0, _satSolver->getLevel() ) );

        if ( _statistics )
        {
            end = TimeUtils::sampleMicro();
            total += TimeUtils::timePassed( start, end );
        }
    }

    if ( _statistics )
        start = TimeUtils::sampleMicro();

    int lit = _literalsToPropagate.popFront().first();

    // In case of assigned boolean variable with opposite assignment, find a conflict clause and
    // terminate propagating
    if ( lit )
        if ( isLiteralAssigned( -lit ) )
        {
            if ( _externalClauseToAdd.empty() )
            {
                if ( GlobalConfiguration::ANALYZE_PROOF_DEPENDENCIES )
                    _engine->explainSimplexFailure();
                else
                    addDecisionBasedConflictClause();
            }

            ASSERT( !_externalClauseToAdd.empty() )
            _literalsToPropagate.clear();
            _literalsToPropagate.append( Pair<int, unsigned>( 0, _satSolver->getLevel() ) );
        }

    CDCL_LOG(
        Stringf( "%u l%d Propagating literal %d", _index, _satSolver->getLevel(), lit ).ascii() )
    ASSERT( FloatUtils::abs( lit ) <= _satSolver->vars() )

    if ( _statistics )
    {
        end = TimeUtils::sampleMicro();
        total += TimeUtils::timePassed( start, end );

        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CALLBACKS_MICRO, total );
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CB_PROPAGATE_MICRO, total );
    }
    return lit;
}

int CdclCore::cb_add_reason_clause_lit( int propagated_lit )
{
    ASSERT( _engine->getLpSolverType() == LPSolverType::NATIVE )
    struct timespec start = TimeUtils::sampleMicro();
    ASSERT( propagated_lit )
    ASSERT( !isDecision( propagated_lit ) )

    if ( !_isReasonClauseInitialized )
    {
        _reasonClauseLiterals.clear();
        if ( _numOfClauses == _vsidsDecayThreshold )
        {
            _numOfClauses = 0;
            _vsidsDecayThreshold = 512 * luby( ++_vsidsDecayCounter );
            _literalToClauses.clear();
        }

        CDCL_LOG( Stringf( "%u l%d Adding reason clause for literal %d",
                           _index,
                           _satSolver->getLevel(),
                           propagated_lit )
                      .ascii() )
        Set<int> clause = {};

        if ( !isLiteralFixed( propagated_lit ) )
        {
            if ( GlobalConfiguration::ANALYZE_PROOF_DEPENDENCIES )
                clause =
                    _engine->explainPhaseWithProof( _satSolverVarToPlc[abs( propagated_lit )] );
            else
            {
                for ( int lit : _sncSplitLiterals )
                    clause.insert( lit );

                for ( unsigned level = 1; level <= _satSolver->getLevel(); ++level )
                {
                    if ( !_decisionLiterals.exists( level ) )
                    {
                        ASSERT( level == _satSolver->getLevel() )
                        continue;
                    }

                    ASSERT( _decisionLiterals.exists( level ) );
                    int lit = _decisionLiterals[level];
                    ASSERT( isDecision( lit ) && lit != propagated_lit );

                    if ( _assignedLiterals[lit] >= _assignedLiterals[propagated_lit] )
                        break;

                    if ( !isLiteralFixed( lit ) )
                        clause.insert( lit );
                }
            }

            if ( GlobalConfiguration::CDCL_SHORTEN_CLAUSES &&
                 !GlobalConfiguration::ANALYZE_PROOF_DEPENDENCIES )
            {
                std::shared_ptr<Query> inputQuery = _engine->getInputQuery();
                NLR::NetworkLevelReasoner *networkLevelReasoner =
                    _engine->getNetworkLevelReasoner();
                networkLevelReasoner->obtainCurrentBounds( *inputQuery );

                setInputBoundsForLiteralInNLR( -propagated_lit, inputQuery, networkLevelReasoner );

                if ( !checkIfShouldSkipClauseShortening( clause ) )
                {
                    Vector<Pair<double, int>> clauseScores;
                    computeClauseScores( clause, clauseScores );
                    reorderByDecisionLevelIfNecessary( clauseScores );
                    clause.clear();
                    networkLevelReasoner->obtainCurrentBounds( *inputQuery );
                    computeShortedClause( clause, clauseScores, propagated_lit );
                }
            }

            if ( GlobalConfiguration::CDCL_SHARE_CLAUSES &&
                 clause.size() <=
                     static_cast<unsigned>(
                         GlobalConfiguration::CDCL_SHARED_CLAUSES_SIZE_LIMIT_PERCENTAGE *
                         _satSolver->vars() ) -
                         1 )
            {
                unsigned newClauseIndex = CdclCore::clauseIndex.fetch_add( 1 );
                CdclCore::sharedClausesMutex.lock();
                CdclCore::sharedClauses[newClauseIndex] = clause;
                CdclCore::sharedClauses[newClauseIndex].insert( -propagated_lit );
                CdclCore::sharedClausesMutex.unlock();
                _sharedClauseAdded.insert( newClauseIndex );
            }

            for ( int lit : clause )
            {
                // Make sure all clause literals were fixed before the literal to explain
                ASSERT( isLiteralAssigned( lit ) );

                ASSERT( !GlobalConfiguration::ANALYZE_PROOF_DEPENDENCIES ||
                        _satSolverVarToPlc[abs( propagated_lit )]->getPhaseFixingEntry()->id >
                            _satSolverVarToPlc[abs( lit )]->getPhaseFixingEntry()->id )

                // Remove fixed literals from clause, as they are redundant
                if ( !isLiteralFixed( -lit ) )
                {
                    _reasonClauseLiterals.append( -lit );
                    _literalToClauses[-lit].insert( _numOfClauses );
                }
            }
        }

        ASSERT( !_reasonClauseLiterals.exists( -propagated_lit ) )
        _reasonClauseLiterals.append( propagated_lit );
        _literalToClauses[propagated_lit].insert( _numOfClauses );
        ++_numOfClauses;
        _isReasonClauseInitialized = true;

        // Unit clause fixes the propagated literal
        if ( _reasonClauseLiterals.size() == 1 )
            _fixedCadicalVars.insert( propagated_lit );
    }

    int lit = 0;
    if ( !_reasonClauseLiterals.empty() )
    {
        lit = _reasonClauseLiterals.pop();
        ASSERT( FloatUtils::abs( lit ) <= _satSolver->vars() )
        CDCL_LOG(
            Stringf(
                "%u l%d\tAdding Literal %d for Reason Clause", _index, _satSolver->getLevel(), lit )
                .ascii() )
    }
    else
        _isReasonClauseInitialized = false;

    if ( _statistics )
    {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CALLBACKS_MICRO,
                                       TimeUtils::timePassed( start, end ) );
        _statistics->incLongAttribute(
            Statistics::TOTAL_TIME_CDCL_CORE_CB_ADD_REASON_CLAUSE_LIT_MICRO,
            TimeUtils::timePassed( start, end ) );
    }

    return lit;
}

bool CdclCore::cb_has_external_clause( bool & /*is_forgettable*/ )
{
    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
        return false;

    if ( checkIfShouldExitDueToTimeout() )
        return false;

    CDCL_LOG( Stringf( "%u l%d Checking if there is a Conflict Clause to add: %d",
                       _index,
                       _satSolver->getLevel(),
                       !_externalClauseToAdd.empty() )
                  .ascii() )

    if ( !_externalClauseToAdd.empty() )
        return true;

    while ( _lastSharedClauseIndexAdded < CdclCore::sharedClauses.size() )
    {
        if ( _sharedClauseAdded.exists( _lastSharedClauseIndexAdded ) )
            ++_lastSharedClauseIndexAdded;
        else
        {
            CdclCore::sharedClausesMutex.lock();
            const auto &clause = CdclCore::sharedClauses[_lastSharedClauseIndexAdded++];
            CdclCore::sharedClausesMutex.unlock();
            bool hasIntersection = false;

            for ( int lit : _sncSplitLiterals )
                if ( clause.exists( lit ) )
                {
                    hasIntersection = true;
                    break;
                }

            if ( !hasIntersection )
                addExternalClause( clause, false );
            else
                continue;

            return true;
        }
    }

    return false;
}

int CdclCore::cb_add_external_clause_lit()
{
    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
        return 0;

    if ( checkIfShouldExitDueToTimeout() )
        return 0;

    struct timespec start = TimeUtils::sampleMicro();

    ASSERT( !_externalClauseToAdd.empty() )

    // Add literal from the last conflict clause learned
    int lit = _externalClauseToAdd.pop();
    ASSERT( FloatUtils::abs( lit ) <= _satSolver->vars() )
    CDCL_LOG(
        Stringf(
            "%u l%d\tAdding Literal %d to Conflict Clause", _index, _satSolver->getLevel(), lit )
            .ascii() )

    if ( _statistics )
    {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CALLBACKS_MICRO,
                                       TimeUtils::timePassed( start, end ) );
        _statistics->incLongAttribute(
            Statistics::TOTAL_TIME_CDCL_CORE_CB_ADD_EXTERNAL_CLAUSE_LIT_MICRO,
            TimeUtils::timePassed( start, end ) );
    }
    return lit;
}

void CdclCore::addExternalClause( const Set<int> &clause, bool shareClause )
{
    CDCL_LOG( Stringf( "%u l%d Add External Clause", _index, _satSolver->getLevel() ).ascii() )
    struct timespec start = TimeUtils::sampleMicro();

    ASSERT( !clause.exists( 0 ) )

    // Vivification against the implication skeleton: entailed clause +
    // entailed probe facts => the shortened clause is entailed too.
    if ( !( _skeletonUnits.empty() && _skeletonImplied.empty() && _probePinFacts.empty() ) &&
         clause.size() > 1 )
    {
        Set<int> vivified = clause;
        vivifyClause( vivified );
        if ( vivified.size() < clause.size() )
        {
            addExternalClause( vivified, shareClause );
            return;
        }
    }

    // Queue for rung-1 (LP-grade) vivification at the next level-0 visit.
    // Clauses produced by that pass itself must not re-queue.
    static const unsigned lpMaxSize = [] {
        const char *s = getenv( "SKELETON_VIVIFY_LP_MAX_SIZE" );
        return s ? (unsigned)atoi( s ) : 32u;
    }();
    bool queuedForVivify = false;
    if ( !_inVivifyLpPass && !_probeMode && !_cdclVarToB.empty() && clause.size() > 1 &&
         clause.size() <= lpMaxSize && !getenv( "SKELETON_NO_VIVIFY_LP" ) )
    {
        _vivifyLpQueue.append( clause );
        ++_numVivifyLpQueued;
        queuedForVivify = true;
    }
    // Queued clauses reach the mirror when dequeued (post-attempt, so a
    // clause can never refute itself); everything else mirrors now.
    if ( !queuedForVivify )
        mirrorAddClause( clause );

    if ( shareClause &&
         clause.size() <=
             static_cast<unsigned>( GlobalConfiguration::CDCL_SHARED_CLAUSES_SIZE_LIMIT_PERCENTAGE *
                                    _satSolver->vars() ) )
    {
        unsigned newClauseIndex = CdclCore::clauseIndex.fetch_add( 1 );
        CdclCore::sharedClausesMutex.lock();
        CdclCore::sharedClauses[newClauseIndex] = clause;
        CdclCore::sharedClausesMutex.unlock();
        _sharedClauseAdded.insert( newClauseIndex );
    }

    if ( _numOfClauses == _vsidsDecayThreshold )
    {
        _numOfClauses = 0;
        _vsidsDecayThreshold = 512 * luby( ++_vsidsDecayCounter );
        _literalToClauses.clear();
    }

    _externalClauseToAdd.append( 0 );

    // Remove fixed literals as they are redundant
    for ( int lit : clause )
    {
        _externalClauseToAdd.append( -lit );
        if ( !isLiteralFixed( lit ) && !isLiteralFixed( -lit ) )
            _literalToClauses[-lit].insert( _numOfClauses );
    }

    ++_numOfClauses;

    // Clauses injected by the vivification pass are not fresh conflicts;
    // keeping them out of the restart schedule keeps the cadence comparable
    // with and without vivification.
    if ( !_inVivifyLpPass )
    {
        ++_numOfConflictClauses;
        if ( _numOfConflictClauses == _restartLimit )
            _shouldRestart = true;
    }

    if ( _statistics )
    {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_MAIN_LOOP_MICRO,
                                       TimeUtils::timePassed( start, end ) );
    }
}

const PiecewiseLinearConstraint *CdclCore::getConstraintFromLit( int lit ) const
{
    if ( _satSolverVarToPlc.exists( (unsigned)FloatUtils::abs( lit ) ) )
        return _satSolverVarToPlc.at( (unsigned)FloatUtils::abs( lit ) );
    return nullptr;
}

bool CdclCore::solveWithCDCL( double timeoutInSeconds )
{
    if ( !_satSolver )
        reset();

    _timeoutInSeconds = timeoutInSeconds;

    // Maybe query detected as UNSAT in processInputQuery
    //        if ( _engine->getExitCode() == ExitCode::UNSAT )
    //            return false;

    // Add all literals initially in literalsToPropagate as snc literals
    for ( const auto &pair : _literalsToPropagate )
    {
        ASSERT( pair.first() != 0 && pair.second() == 0 )
        _sncSplitLiterals.append( pair.first() );
        _fixedCadicalVars.insert( pair.first() );
    }

    if ( Options::get()->getString( Options::NAP_EXTERNAL_CLAUSE_FILE_PATH ) == "" &&
         Options::get()->getString( Options::NAP_EXTERNAL_CLAUSE_FILE_PATH2 ) == "" )
        if ( _engine->solve( _timeoutInSeconds ) )
        {
            _engine->setExitCode( ExitCode::SAT );
            if ( GlobalConfiguration::WRITE_ALETHE_PROOF &&
                 !Options::get()->getBool( Options::DNC_MODE ) )
                _engine->deleteProofIfExists();
            return true;
        }

    // Add the zero literal at the end
    if ( !_literalsToPropagate.empty() )
        _literalsToPropagate.append( Pair<int, unsigned>( 0, _satSolver->getLevel() ) );

    if ( !_externalClauseToAdd.empty() )
    {
        ASSERT( _engine->getExitCode() == ExitCode::NOT_DONE );
        _engine->setExitCode( ExitCode::UNSAT );
        return false;
    }

    // Entailed implication-skeleton clauses (failed-literal units + binary
    // phase implications): sound by construction, so they join the formula
    // like any initial clause.
    for ( const Set<int> &clause : _skeletonClauses )
    {
        _satSolver->addClause( clause );
        _initialClauses.append( clause );
        mirrorAddClause( clause );
    }

    Set<int> externalClause;

    externalClause = _satSolver->addExternalNAPClause(
        Options::get()->getString( Options::NAP_EXTERNAL_CLAUSE_FILE_PATH ) );
    if ( !externalClause.empty() )
        _initialClauses.append( externalClause );

    externalClause = _satSolver->addExternalNAPClause(
        Options::get()->getString( Options::NAP_EXTERNAL_CLAUSE_FILE_PATH2 ) );
    if ( !externalClause.empty() )
        _initialClauses.append( externalClause );

    CDCL_LOG( Stringf( "%u l%d Start solving", _index, _satSolver->getLevel() ).ascii() )
    int result = _satSolver->solve();

    if ( _statistics && _engine->getVerbosity() )
    {
        printf( "\nCdclCore::Final statistics:\n" );
        if ( _numVivifiedLiterals > 0 || _numVivifiedLiteralsBounds > 0 || _vivifyTimeMicro > 0 )
            printf( "\tSkeleton vivification: %u edge-removed + %u bound-removed literals "
                    "(%u numeric checks, %u size-skips, max clause %u, %.2fs total)\n",
                    _numVivifiedLiterals,
                    _numVivifiedLiteralsBounds,
                    _numVivifyNumericChecks,
                    _numVivifySizeSkips,
                    _maxVivifyClauseSize,
                    _vivifyTimeMicro / 1e6 );
        if ( _numVivifyMirrorCalls > 0 )
            printf( "\tMirror oracle: %u solves -> %u clauses shortened, %u literals "
                    "removed (%u unsat / %u sat / %u unknown, %u full-core); "
                    "%u probe solves -> %u failed literals; %u units + %u edges "
                    "learned; %u unsat proofs\n",
                    _numVivifyMirrorCalls,
                    _numVivifyMirrorShortened,
                    _numVivifyMirrorLiteralsRemoved,
                    _numVivifyMirrorUnsat,
                    _numVivifyMirrorSat,
                    _numVivifyMirrorUnknown,
                    _numVivifyMirrorFullCore,
                    _numMirrorProbeSolves,
                    _numMirrorFailedLits,
                    _numMirrorUnitsLearned,
                    _numMirrorEdgesLearned,
                    _numMirrorUnsatProofs );
        if ( _numVivifyLpVisits > 0 || _numVivifyLpQueued > 0 )
            printf( "\tLP vivification (rung 1): %u descents -> %u clauses shortened, "
                    "%u literals removed, %u edges harvested (%u level-0 visits, "
                    "%u queued, %u fixed-skips, %u still queued, %.2fs total)\n",
                    _numVivifyLpDescents,
                    _numVivifyLpShortened,
                    _numVivifyLpLiteralsRemoved,
                    _numVivifyLpHarvestedEdges,
                    _numVivifyLpVisits,
                    _numVivifyLpQueued,
                    _numVivifyLpSkippedFixed,
                    _vivifyLpQueue.size(),
                    _vivifyLpTimeMicro / 1e6 );
        _statistics->print();
    }

    if ( ( result != 20 || ( _engine->getExitCode() != ExitCode::NOT_DONE &&
                             _engine->getExitCode() != ExitCode::UNSAT ) ) &&
         GlobalConfiguration::WRITE_ALETHE_PROOF && !Options::get()->getBool( Options::DNC_MODE ) )
        _engine->deleteProofIfExists();

    if ( _engine->getExitCode() == ExitCode::TIMEOUT )
        return false;

    if ( result == 0 )
    {
        if ( _engine->getExitCode() == ExitCode::SAT )
            return true;
        else if ( checkIfShouldExitDueToTimeout() )
        {
            if ( _statistics )
            {
                if ( _engine->getVerbosity() > 0 )
                {
                    printf( "\n\nCdclCore: quitting due to timeout...\n\n" );
                    printf( "Final statistics:\n" );
                    _statistics->print();
                }
                _statistics->timeout();
            }

            _engine->setExitCode( ExitCode::TIMEOUT );
            return false;
        }
    }
    else if ( result == 10 )
    {
        ASSERT( _engine->getExitCode() == ExitCode::NOT_DONE );
        _engine->setExitCode( ExitCode::SAT );
        return true;
    }
    else if ( result == 20 )
    {
        ASSERT( _engine->getExitCode() == ExitCode::NOT_DONE );
        _engine->setExitCode( ExitCode::UNSAT );
        return false;
    }
    else
    {
        ASSERT( false )
    }

    return false;
}

void CdclCore::addLiteralToPropagate( int literal )
{
    // Probe-conditional facts must not leak to the SAT solver as root facts.
    if ( _probeMode )
        return;

    // The skeleton's C2 hull fold tightens the root tableau after probe mode is
    // lifted but before solveWithCDCL constructs _satSolver, so a phase fixed by
    // that cascade lands here with no solver to propagate to. Nothing is lost by
    // dropping it: solveWithCDCL clears _literalsToPropagate right after
    // creating the solver, so a queued literal would be discarded anyway, and
    // the fold's bounds live on in the root tableau for theory propagation to
    // rediscover.
    if ( !_satSolver )
        return;

    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
        return;

    struct timespec start = TimeUtils::sampleMicro();

    ASSERT( literal )
    if ( !isLiteralAssigned( literal ) && !isLiteralToBePropagated( literal ) )
    {
        ASSERT( !isLiteralAssigned( -literal ) && !isLiteralToBePropagated( -literal ) )
        _literalsToPropagate.append( Pair<int, unsigned>( literal, _satSolver->getLevel() ) );
    }

    if ( _statistics )
    {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_MAIN_LOOP_MICRO,
                                       TimeUtils::timePassed( start, end ) );
    }
}

bool CdclCore::isLiteralToBePropagated( int literal ) const
{
    for ( const Pair<int, unsigned> &pair : _literalsToPropagate )
        if ( pair.first() == literal )
            return true;

    return false;
}

namespace {
// Harvests the oracle's learned units and binaries: learned clauses are
// resolution consequences of the clause DB alone (never of the assumptions),
// so each is an entailed fact.
class CdclMirrorLearner : public CaDiCaL::Learner
{
public:
    CdclMirrorLearner( CdclCore *core )
        : _core( core )
    {
    }
    bool learning( int size ) override
    {
        return size <= 2;
    }
    void learn( int lit ) override
    {
        if ( lit )
        {
            _current.push_back( lit );
            return;
        }
        if ( !_current.empty() )
            _core->bufferMirrorLearned( _current );
        _current.clear();
    }

private:
    CdclCore *_core;
    std::vector<int> _current;
};
} // namespace

void CdclCore::mirrorAddClause( const Set<int> &clause )
{
    if ( getenv( "SKELETON_NO_VIVIFY_MIRROR" ) )
        return;
    // The empty clause (root-conflict signal, incl. our own mirror-UNSAT
    // delivery) must not enter the mirror: it would flatten the DB to bare
    // falsum and destroy the oracle's remaining duties.
    if ( clause.empty() )
        return;
    if ( !_vivifyMirror )
    {
        _vivifyMirror = std::make_unique<CaDiCaL::Solver>();
        _mirrorLearner = std::make_unique<CdclMirrorLearner>( this );
        _vivifyMirror->connect_learner( _mirrorLearner.get() );
    }
    for ( int lit : clause )
        _vivifyMirror->add( lit );
    _vivifyMirror->add( 0 );
    ++_numMirrorClauses;

    // Streaming clause log for offline poison analysis: the solver's own
    // dump is post-simplification and hides the original clauses.
    if ( const char *logPath = getenv( "MIRROR_LOG" ) )
    {
        static std::ofstream mirrorLog( logPath );
        for ( int lit : clause )
            mirrorLog << lit << " ";
        mirrorLog << "0\n";
        mirrorLog.flush();
    }
}

void CdclCore::flushMirrorLearned()
{
    for ( const auto &lits : _mirrorLearnedBuffer )
    {
        if ( lits.size() == 1 )
        {
            int u = lits[0];
            if ( _seenMirrorUnits.exists( u ) )
                continue;
            _seenMirrorUnits.insert( u );
            _skeletonUnits.insert( u );
            Set<int> unit;
            unit.insert( u );
            ++_numMirrorUnitsLearned;
            _inVivifyLpPass = true;
            addExternalClause( unit, false );
            _inVivifyLpPass = false;
        }
        else if ( lits.size() == 2 )
        {
            int a = lits[0];
            int b = lits[1];
            std::pair<int, int> key( std::min( a, b ), std::max( a, b ) );
            if ( _seenHarvestEdges.count( key ) )
                continue;
            _seenHarvestEdges.insert( key );
            _skeletonImplied[-a].insert( b );
            _skeletonImplied[-b].insert( a );
            Set<int> binary;
            binary.insert( a );
            binary.insert( b );
            ++_numMirrorEdgesLearned;
            _inVivifyLpPass = true;
            addExternalClause( binary, false );
            _inVivifyLpPass = false;
        }
    }
    _mirrorLearnedBuffer.clear();
}

void CdclCore::mirrorProbePass()
{
    // Boolean failed-literal probing: decisions-0 solves are pure
    // propagation over the oracle's DB (including everything it has
    // learned); a conflict makes the negation a free unit. Re-runs only
    // when the DB has grown since the last pass.
    if ( !_vivifyMirror || getenv( "SKELETON_NO_MIRROR_PROBE" ) )
        return;
    if ( _numMirrorClauses < _lastMirrorProbeClauses + 32 )
        return;
    _lastMirrorProbeClauses = _numMirrorClauses;

    struct timespec start = TimeUtils::sampleMicro();
    for ( const auto &pair : _cdclVarToB )
    {
        int var = (int)pair.first;
        if ( isLiteralFixed( var ) || isLiteralFixed( -var ) )
            continue;
        for ( int lit : { var, -var } )
        {
            if ( _seenMirrorUnits.exists( lit ) || _seenMirrorUnits.exists( -lit ) )
                continue;
            if ( _vivifyMirror->fixed( lit ) != 0 )
                continue;
            _vivifyMirror->assume( lit );
            _vivifyMirror->limit( "decisions", 0 );
            int result = _vivifyMirror->solve();
            ++_numMirrorProbeSolves;
            flushMirrorLearned();
            if ( result == 20 && !_vivifyMirror->failed( lit ) )
            {
                // UNSAT below the assumption: global unsat proof.
                ++_numMirrorUnsatProofs;
                if ( _engine->getVerbosity() > 0 )
                {
                    printf( "Mirror UNSAT proof (probe pass): query is unsat\n" );
                    fflush( stdout );
                }
                Set<int> emptyClause;
                _inVivifyLpPass = true;
                addExternalClause( emptyClause, false );
                _inVivifyLpPass = false;
                return;
            }
            if ( result == 20 && !_seenMirrorUnits.exists( -lit ) )
            {
                ++_numMirrorFailedLits;
                _seenMirrorUnits.insert( -lit );
                _skeletonUnits.insert( -lit );
                Set<int> unit;
                unit.insert( -lit );
                _inVivifyLpPass = true;
                addExternalClause( unit, false );
                _inVivifyLpPass = false;
            }
        }
        if ( TimeUtils::timePassed( start, TimeUtils::sampleMicro() ) / 1e6 > 5.0 )
            break;
    }
}

void CdclCore::processVivifyLpQueue()
{
    ++_numVivifyLpVisits;
    if ( _vivifyLpQueue.empty() || _cdclVarToB.empty() || !_satSolver )
        return;

    static const unsigned clausesPerVisit = [] {
        const char *s = getenv( "SKELETON_VIVIFY_LP_CLAUSES" );
        return s ? (unsigned)atoi( s ) : 256u;
    }();
    // Global wall-clock budget: rung-1 descents must never eat the run.
    static const double totalBudgetSec = [] {
        const char *s = getenv( "SKELETON_VIVIFY_LP_BUDGET" );
        return s ? atof( s ) : 60.0;
    }();

    if ( _vivifyLpTimeMicro / 1e6 >= totalBudgetSec )
        return;

    struct timespec start = TimeUtils::sampleMicro();

    // Sync level-0 fixed literals into the mirror: theory-propagated units
    // never appear in the external clause stream, so the mirror cannot
    // re-derive them on its own.
    if ( _vivifyMirror )
        for ( int lit : _fixedCadicalVars )
            if ( !_mirroredFixed.exists( lit ) )
            {
                Set<int> unit;
                unit.insert( lit );
                mirrorAddClause( unit );
                _mirroredFixed.insert( lit );
            }

    // Facts derived under the descent pins are conditional on them: keep
    // them away from the SAT solver (same guard as the probe pass).
    setProbeMode( true );

    unsigned processed = 0;
    while ( !_vivifyLpQueue.empty() && processed < clausesPerVisit &&
            _engine->getExitCode() == ExitCode::NOT_DONE && !checkIfShouldExitDueToTimeout() &&
            ( _vivifyLpTimeMicro + TimeUtils::timePassed( start, TimeUtils::sampleMicro() ) ) /
                    1e6 <
                totalBudgetSec )
    {
        Set<int> clause = _vivifyLpQueue.back();
        _vivifyLpQueue.popBack();
        ++processed;

        // A literal fixed TRUE satisfies the clause at root: skip. A literal
        // fixed FALSE is just dead - it stays out of the pin set below (its
        // negation is a root fact, so omitting its pin only weakens the
        // check), and the descent proceeds on the live literals.
        bool skip = false;
        for ( int lit : clause )
            if ( isLiteralFixed( lit ) )
            {
                skip = true;
                break;
            }
        if ( skip )
        {
            ++_numVivifyLpSkippedFixed;
            continue;
        }

        // Full-SAT boolean vivification: refute the negated clause against
        // the mirrored clause DB (conflict-bounded); on UNSAT the
        // failed-assumption core IS a shortened clause - full conflict
        // analysis over everything learned so far, zero theory cost. The
        // clause itself is not yet in the mirror, so it cannot refute
        // itself; it enters post-attempt (shortened version wins).
        if ( _vivifyMirror )
        {
            for ( int lit : clause )
                _vivifyMirror->assume( -lit );
            _vivifyMirror->limit( "conflicts", 200 );
            int mirrorResult = _vivifyMirror->solve();
            ++_numVivifyMirrorCalls;
            flushMirrorLearned();
            if ( mirrorResult == 20 )
                ++_numVivifyMirrorUnsat;
            else if ( mirrorResult == 10 )
                ++_numVivifyMirrorSat;
            else
                ++_numVivifyMirrorUnknown;
            if ( mirrorResult == 20 )
            {
                Set<int> shortened;
                for ( int lit : clause )
                    if ( _vivifyMirror->failed( -lit ) )
                        shortened.insert( lit );
                if ( shortened.empty() )
                {
                    // UNSAT below the assumptions: the mirror holds only
                    // query-entailed clauses, so the QUERY is unsat. Deliver
                    // the root conflict; the search ends here.
                    ++_numMirrorUnsatProofs;
                    if ( _engine->getVerbosity() > 0 )
                    {
                        printf( "Mirror UNSAT proof: entailed clause set is "
                                "boolean-unsat; query is unsat\n" );
                        fflush( stdout );
                    }
                    Set<int> emptyClause;
                    _inVivifyLpPass = true;
                    addExternalClause( emptyClause, false );
                    _inVivifyLpPass = false;
                    break;
                }
                if ( shortened.size() >= clause.size() )
                    ++_numVivifyMirrorFullCore;
                if ( shortened.size() < clause.size() )
                {
                    _numVivifyMirrorLiteralsRemoved += clause.size() - shortened.size();
                    ++_numVivifyMirrorShortened;
                    _inVivifyLpPass = true;
                    addExternalClause( shortened, false ); // also mirrors it
                    _inVivifyLpPass = false;
                    continue;
                }
            }
        }
        mirrorAddClause( clause );

        // Pin order: most-tightening-first (rung-0 delta counts), so
        // infeasibility hits as early as possible in the descent.
        std::vector<std::pair<int, unsigned>> pinnable; // (literal, delta count)
        for ( int lit : clause )
        {
            if ( isLiteralFixed( -lit ) )
                continue; // dead literal: not pinned, not kept
            unsigned var = (unsigned)( lit > 0 ? lit : -lit );
            if ( !_cdclVarToB.exists( var ) )
                continue; // unpinnable: kept in the clause regardless
            unsigned deltas = 0;
            if ( _probePinFacts.exists( -lit ) )
                deltas =
                    _probePinFacts[-lit].lbDeltas.size() + _probePinFacts[-lit].ubDeltas.size();
            pinnable.emplace_back( lit, deltas );
        }
        if ( pinnable.size() < 2 )
            continue;
        std::sort( pinnable.begin(),
                   pinnable.end(),
                   []( const std::pair<int, unsigned> &a, const std::pair<int, unsigned> &b ) {
                       return a.second > b.second;
                   } );

        // Plain pins over the original literals: boolean shortening is the
        // mirror's job now (full conflict analysis beats hand-rolled BCP),
        // the LP descent handles what only the theory can refute.
        Vector<Pair<unsigned, bool>> pins;
        for ( const auto &p : pinnable )
            pins.append(
                Pair<unsigned, bool>( _cdclVarToB[(unsigned)std::abs( p.first )], p.first < 0 ) );

        Vector<Pair<unsigned, bool>> impliedPhases;
        int applied = _engine->probePinDescent(
            pins, GlobalConfiguration::SKELETON_PROBE_SIMPLEX_PIVOT_CAP, &impliedPhases );
        ++_numVivifyLpDescents;

        // Harvest the pin-1 implications as binary clauses: pin of literal
        // l1's negation fixed phase q, so {l1, q} is entailed. Each descent
        // grows the clausalized implication graph at no extra LP cost.
        if ( !impliedPhases.empty() )
        {
            int l1 = pinnable[0].first;
            _inVivifyLpPass = true;
            for ( const auto &implied : impliedPhases )
            {
                if ( !_bToCdclVar.exists( implied.first() ) )
                    continue;
                int q = implied.second() ? (int)_bToCdclVar[implied.first()]
                                         : -(int)_bToCdclVar[implied.first()];
                if ( q == l1 || q == -l1 )
                    continue;
                std::pair<int, int> key( std::min( l1, q ), std::max( l1, q ) );
                if ( _seenHarvestEdges.count( key ) )
                    continue;
                _seenHarvestEdges.insert( key );
                Set<int> edge;
                edge.insert( l1 );
                edge.insert( q );
                addExternalClause( edge, false );
                // Feed the edge into the implication map too, so BCP-extended
                // descents and rung-0 closure see the growing graph.
                _skeletonImplied[-l1].insert( q );
                _skeletonImplied[-q].insert( l1 );
                ++_numVivifyLpHarvestedEdges;
            }
            _inVivifyLpPass = false;
        }

        if ( applied < 0 )
            continue; // feasible (or numerical trouble): no shortening

        // Q ^ first `applied` pins is infeasible => the disjunction of those
        // literals is entailed on its own. Everything else drops (including
        // the unpinnable literals).
        Set<int> shortened;
        for ( int i = 0; i < applied; ++i )
            shortened.insert( pinnable[i].first );

        if ( !shortened.empty() && shortened.size() < clause.size() )
        {
            _numVivifyLpLiteralsRemoved += clause.size() - shortened.size();
            ++_numVivifyLpShortened;
            _inVivifyLpPass = true;
            addExternalClause( shortened, false );
            _inVivifyLpPass = false;
        }
    }

    // Boolean failed-literal probing over the oracle, when its DB has grown.
    mirrorProbePass();

    setProbeMode( false );

    _vivifyLpTimeMicro += TimeUtils::timePassed( start, TimeUtils::sampleMicro() );
}

void CdclCore::vivifyClause( Set<int> &clause )
{
    struct timespec start = TimeUtils::sampleMicro();

    bool changed = true;
    while ( changed && clause.size() > 1 )
    {
        changed = false;
        for ( int literal : clause )
        {
            bool byBounds = false;
            // -literal is a root fact: literal can never help satisfy.
            bool removable = _skeletonUnits.exists( -literal ) ||
                             vivifyCandidateRemovable( clause, literal, byBounds );

            if ( removable )
            {
                clause.erase( literal );
                if ( byBounds )
                    ++_numVivifiedLiteralsBounds;
                else
                    ++_numVivifiedLiterals;
                changed = true;
                break; // iterator invalidated; restart scan
            }
        }
    }

    _vivifyTimeMicro += TimeUtils::timePassed( start, TimeUtils::sampleMicro() );
}

bool CdclCore::vivifyCandidateRemovable( const Set<int> &clause, int literal, bool &byBounds )
{
    // Dropping `literal` is sound iff Q ^ (negations of the remaining
    // literals) is infeasible - then C \ {literal} is itself entailed.
    // Rung 0 decides this from probe-time facts only, no LP calls:
    //   1. BCP closure of the pinned negations over the skeleton edges
    //      (boolean contradiction / clash with a root unit / forcing
    //      -literal directly);
    //   2. intersection of the closure pins' stored bound vectors (each is
    //      entailed under its pin, so all hold under the conjunction): an
    //      empty box, or a forced sign on literal's own b, kills literal.
    byBounds = false;

    static constexpr unsigned CLOSURE_CAP = 256;
    static constexpr double VIVIFY_EPS = 1e-6;
    static const unsigned maxNumericSize = [] {
        const char *s = getenv( "SKELETON_VIVIFY_MAX_SIZE" );
        return s ? (unsigned)atoi( s ) : 64u;
    }();

    std::vector<int> worklist;
    Set<int> closure;
    for ( int other : clause )
        if ( other != literal )
            worklist.push_back( -other );

    while ( !worklist.empty() )
    {
        int p = worklist.back();
        worklist.pop_back();
        if ( closure.exists( p ) )
            continue;
        // The pins contradict each other or a root unit: infeasible outright.
        if ( closure.exists( -p ) || _skeletonUnits.exists( -p ) )
            return true;
        // The pins force -literal (multi-hop closure of the old one-edge check).
        if ( p == -literal )
            return true;
        closure.insert( p );
        if ( closure.size() >= CLOSURE_CAP )
            break;
        if ( _skeletonImplied.exists( p ) )
            for ( int q : _skeletonImplied[p] )
                if ( !closure.exists( q ) )
                    worklist.push_back( q );
    }

    if ( clause.size() > _maxVivifyClauseSize )
        _maxVivifyClauseSize = clause.size();
    if ( _probePinFacts.empty() )
        return false;
    if ( clause.size() > maxNumericSize )
    {
        ++_numVivifySizeSkips;
        return false;
    }
    ++_numVivifyNumericChecks;

    // Intersect the stored bound vectors of every closure pin with the root
    // box. Only tightened (delta) entries can participate in a crossing, so
    // the sparse lists suffice.
    std::map<unsigned, std::pair<double, double>> eff;
    auto touch = [&]( unsigned v ) -> std::pair<double, double> & {
        auto it = eff.find( v );
        if ( it == eff.end() )
            it = eff.emplace( v, std::make_pair( _probeRootLbs[v], _probeRootUbs[v] ) ).first;
        return it->second;
    };

    for ( int p : closure )
    {
        if ( !_probePinFacts.exists( p ) )
            continue;
        const ProbePinFacts &facts = _probePinFacts[p];
        for ( const auto &d : facts.lbDeltas )
        {
            auto &e = touch( d.first );
            if ( d.second > e.first )
                e.first = d.second;
        }
        for ( const auto &d : facts.ubDeltas )
        {
            auto &e = touch( d.first );
            if ( d.second < e.second )
                e.second = d.second;
        }
    }

    if ( eff.empty() )
        return false;

    // Box emptiness: some variable's intersected lower bound crosses its
    // intersected upper bound.
    for ( const auto &entry : eff )
        if ( entry.second.first > entry.second.second + VIVIFY_EPS )
        {
            byBounds = true;
            return true;
        }

    // Forced sign on literal's own pre-activation: b < 0 makes the active
    // literal false, b > 0 makes the inactive literal false.
    unsigned var = (unsigned)( literal > 0 ? literal : -literal );
    if ( _cdclVarToB.exists( var ) )
    {
        auto it = eff.find( _cdclVarToB[var] );
        if ( it != eff.end() )
        {
            if ( literal > 0 && it->second.second < -VIVIFY_EPS )
            {
                byBounds = true;
                return true;
            }
            if ( literal < 0 && it->second.first > VIVIFY_EPS )
            {
                byBounds = true;
                return true;
            }
        }
    }

    return false;
}

void CdclCore::addDecisionBasedConflictClause()
{
    // A conflict under a probe's pin refutes the pin, not the query; the
    // probe loop reads the infeasibility directly.
    if ( _probeMode )
        return;

    // Same pre-solver window as addLiteralToPropagate: the C2 hull fold and the
    // pre-search pristine root check both run after probe mode is lifted but
    // before solveWithCDCL constructs _satSolver, and a conflict there reaches
    // this callback with no solver. Bailing out is what we want regardless of
    // the crash: this clause is built from the decisions on the trail, and
    // before the search starts there are none, so it would reduce to the empty
    // clause and assert FALSE for the whole query. A genuine root conflict is
    // rediscovered by the search's own root solve.
    if ( !_satSolver )
        return;

    CDCL_LOG( Stringf( "%u l%d Add Decision Clause", _index, _satSolver->getLevel() ).ascii() )

    struct timespec start = TimeUtils::sampleMicro();

    Set<int> clause = Set<int>();

    for ( int lit : _sncSplitLiterals )
        clause.insert( lit );

    for ( unsigned l = 1; l <= _satSolver->getLevel(); ++l )
    {
        if ( !_decisionLiterals.exists( l ) )
        {
            ASSERT( l == _satSolver->getLevel() );
            continue;
        }

        ASSERT( _decisionLiterals.exists( l ) );
        int lit = _decisionLiterals[l];
        ASSERT( lit != 0 );
        ASSERT( isDecision( lit ) );
        if ( !isLiteralFixed( lit ) )
            clause.insert( lit );
    }

    if ( GlobalConfiguration::CDCL_SHORTEN_CLAUSES )
    {
        std::shared_ptr<Query> inputQuery = _engine->getInputQuery();
        NLR::NetworkLevelReasoner *networkLevelReasoner = _engine->getNetworkLevelReasoner();
        networkLevelReasoner->obtainCurrentBounds( *inputQuery );

        if ( !checkIfShouldSkipClauseShortening( clause ) )
        {
            Vector<Pair<double, int>> clauseScores;
            computeClauseScores( clause, clauseScores );
            reorderByDecisionLevelIfNecessary( clauseScores );
            clause.clear();
            networkLevelReasoner->obtainCurrentBounds( *inputQuery );
            computeShortedClause( clause, clauseScores, 0 );
        }
    }

    addExternalClause( clause, GlobalConfiguration::CDCL_SHARE_CLAUSES );

    if ( _statistics )
    {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_MAIN_LOOP_MICRO,
                                       TimeUtils::timePassed( start, end ) );
    }
}

void CdclCore::removeLiteralFromPropagations( int literal )
{
    _literalsToPropagate.erase( Pair<int, unsigned>( literal, _satSolver->getLevel() ) );
}

void CdclCore::assume( int literal )
{
    CDCL_LOG(
        Stringf( "%u l%d Assuming literal %d", _index, _satSolver->getLevel(), literal ).ascii() )
    _satSolver->assume( literal );
    _fixedCadicalVars.insert( literal );
}

bool CdclCore::checkIfShouldExitDueToTimeout()
{
    if ( _engine->shouldExitDueToTimeout( _timeoutInSeconds ) )
    {
        CDCL_LOG( Stringf( "%u l%d Timeout reached", _index, _satSolver->getLevel() ).ascii() )
        if ( _satSolver->isSolving() )
            _satSolver->terminate();
        return true;
    }

    return false;
}

bool CdclCore::terminate()
{
    CDCL_LOG( Stringf( "%u l%d Callback for terminate: %d",
                       _index,
                       _satSolver->getLevel(),
                       _engine->getExitCode() != ExitCode::NOT_DONE )
                  .ascii() )
    return _engine->getExitCode() != ExitCode::NOT_DONE;
}

unsigned CdclCore::getLiteralAssignmentIndex( int literal )
{
    struct timespec start = TimeUtils::sampleMicro();

    if ( _assignedLiterals.count( literal ) > 0 )
        return _assignedLiterals[literal].get();

    if ( _statistics )
    {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_MAIN_LOOP_MICRO,
                                       TimeUtils::timePassed( start, end ) );
    }

    return _assignedLiterals.size();
}

bool CdclCore::isLiteralFixed( int literal ) const
{
    return _fixedCadicalVars.exists( literal );
}

bool CdclCore::isClauseSatisfied( unsigned int clause ) const
{
    return _satisfiedClauses.contains( clause );
}

unsigned int CdclCore::getLiteralVSIDSScore( int literal ) const
{
    unsigned numOfClausesSatisfiedByLiteral = 0;
    if ( _literalToClauses.exists( literal ) )
        for ( unsigned clause : _literalToClauses[literal] )
            if ( !isClauseSatisfied( clause ) )
                ++numOfClausesSatisfiedByLiteral;
    return numOfClausesSatisfiedByLiteral;
}

unsigned int CdclCore::getVariableVSIDSScore( unsigned int var ) const
{
    return getLiteralVSIDSScore( (int)var ) + getLiteralVSIDSScore( -(int)var );
}

unsigned CdclCore::luby( unsigned int i )
{
    unsigned k;
    for ( k = 1; k < 32; ++k )
        if ( i == (unsigned)( ( 1 << k ) - 1 ) )
            return 1 << ( k - 1 );

    for ( k = 1;; ++k )
        if ( (unsigned)( 1 << ( k - 1 ) ) <= i && i < (unsigned)( ( 1 << k ) - 1 ) )
            return luby( i - ( 1 << ( k - 1 ) ) + 1 );
}

void CdclCore::notify_fixed_assignment( int lit )
{
    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
        return;

    if ( checkIfShouldExitDueToTimeout() )
        return;

    struct timespec start = TimeUtils::sampleMicro();

    CDCL_LOG( Stringf( "%u l%d Notified fixed assignment: %d", _index, _satSolver->getLevel(), lit )
                  .ascii() )
    if ( !isLiteralAssigned( lit ) )
        notifySingleAssignment( lit, true );
    else
        _fixedCadicalVars.insert( lit );

    if ( _statistics )
    {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_CDCL_CORE_CALLBACKS_MICRO,
                                       TimeUtils::timePassed( start, end ) );
        _statistics->incLongAttribute(
            Statistics::TOTAL_TIME_CDCL_CORE_NOTIFY_FIXED_ASSIGNMENT_MICRO,
            TimeUtils::timePassed( start, end ) );
    }
}

bool CdclCore::hasConflictClause() const
{
    return !_externalClauseToAdd.empty();
}

void CdclCore::notifySingleAssignment( int lit, bool isFixed )
{
    ASSERT( lit != 0 );
    if ( isLiteralToBePropagated( -lit ) || isLiteralAssigned( -lit ) )
        return;

    // Allow notifying a negation of assigned literal only when a conflict is already discovered
    ASSERT( !isLiteralAssigned( -lit ) || !_externalClauseToAdd.empty() )

    if ( isFixed )
        _fixedCadicalVars.insert( lit );

    if ( isDecision( lit ) )
        _decisionLiterals.insert( ++_decisionIndex, lit );

    // Pick the split to perform
    PiecewiseLinearConstraint *plc = _satSolverVarToPlc.at( (unsigned)FloatUtils::abs( lit ) );
    DEBUG( PhaseStatus originalPlcPhase = plc->getPhaseStatus() );

    plc->propagateLitAsSplit( lit );
    _engine->applySplit( plc->getValidCaseSplit() );
    plc->setActiveConstraint( false );

    ASSERT( !isLiteralAssigned( lit ) )

    _assignedLiterals.insert( lit, _assignedLiterals.size() );
    for ( unsigned clause : _literalToClauses[lit] )
        if ( !isClauseSatisfied( clause ) )
            _satisfiedClauses.insert( clause );

    ASSERT( originalPlcPhase == PHASE_NOT_FIXED || plc->getPhaseStatus() == originalPlcPhase );
}

void CdclCore::setStatistics( Statistics *statistics )
{
    _statistics = statistics;
}

void CdclCore::pushContext()
{
    struct timespec start = TimeUtils::sampleMicro();
    _context.push();
    _satSolver->push();
    struct timespec end = TimeUtils::sampleMicro();

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_CONTEXT_PUSHES );
        _statistics->incLongAttribute( Statistics::TIME_CONTEXT_PUSH,
                                       TimeUtils::timePassed( start, end ) );
    }
}

void CdclCore::popContextTo( unsigned int level )
{
    struct timespec start = TimeUtils::sampleMicro();
    unsigned int prevLevel = _satSolver->getLevel();
    _context.popto( _context.getLevel() - (int)_satSolver->getLevel() + (int)level );
    _satSolver->popto( level );
    struct timespec end = TimeUtils::sampleMicro();

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_CONTEXT_POPS, prevLevel - level );
        _statistics->incLongAttribute( Statistics::TIME_CONTEXT_POP,
                                       TimeUtils::timePassed( start, end ) );
    }
}

void CdclCore::addLiteral( int lit )
{
    _satSolver->addLiteral( lit );
}

bool CdclCore::isSupported( const PiecewiseLinearConstraint *plc )
{
    if ( plc->getType() != RELU )
        return false;

    return true;
}

unsigned CdclCore::decideSplitVarBasedOnPolarityAndVsids() const
{
    unsigned decisionVariable = 0;

    NLR::NetworkLevelReasoner *networkLevelReasoner = _engine->getNetworkLevelReasoner();
    ASSERT( networkLevelReasoner )

    List<PiecewiseLinearConstraint *> constraints =
        networkLevelReasoner->getConstraintsInTopologicalOrder();

    Map<double, PiecewiseLinearConstraint *> polarityScoreToConstraint;
    for ( auto &plConstraint : constraints )
    {
        if ( _largestAssignmentSoFar.exists( plConstraint->getVariableForDecision() ) )
            if ( plConstraint->supportPolarity() && plConstraint->isActive() &&
                 !plConstraint->phaseFixed() )
            {
                plConstraint->updateScoreBasedOnPolarity();
                polarityScoreToConstraint[plConstraint->getScore()] = plConstraint;
                if ( polarityScoreToConstraint.size() >=
                     GlobalConfiguration::POLARITY_CANDIDATES_THRESHOLD )
                    break;
            }
    }

    for ( auto &plConstraint : constraints )
    {
        if ( plConstraint->supportPolarity() && plConstraint->isActive() &&
             !plConstraint->phaseFixed() )
        {
            plConstraint->updateScoreBasedOnPolarity();
            polarityScoreToConstraint[plConstraint->getScore()] = plConstraint;
            if ( polarityScoreToConstraint.size() >=
                 GlobalConfiguration::POLARITY_CANDIDATES_THRESHOLD )
                break;
        }
    }

    if ( !polarityScoreToConstraint.empty() )
    {
        double maxScore = 0;
        for ( double polarityScore : polarityScoreToConstraint.keys() )
        {
            unsigned var = polarityScoreToConstraint[polarityScore]->getVariableForDecision();
            double score = ( getVariableVSIDSScore( var ) + 1 ) * polarityScore;
            if ( score > maxScore )
            {
                decisionVariable = var;
                maxScore = score;
            }
        }
    }

    return decisionVariable;
}

unsigned CdclCore::decideSplitVarBasedOnPseudoImpactAndVsids() const
{
    ASSERT( GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH )
    double maxScore = 0;
    unsigned variableWithMaxScore = 0;

    for ( const auto &pair : _satSolverVarToPlc )
    {
        unsigned var = pair.first;
        if ( var == 0 )
            continue;

        PiecewiseLinearConstraint *plc = pair.second;
        if ( plc->isActive() && !plc->phaseFixed() )
        {
            ASSERT( !isLiteralAssigned( (int)var ) && !isLiteralAssigned( -(int)var ) )
            double pseudoImpactScore = _scoreTracker->getScore( plc );
            double vsidsScore = getVariableVSIDSScore( var );
            double score = ( vsidsScore + 1 ) * pseudoImpactScore;
            if ( score >= maxScore )
            {
                maxScore = score;
                variableWithMaxScore = var;
            }
        }
    }
    return variableWithMaxScore;
}

void CdclCore::initializeScoreTracker( std::shared_ptr<PLConstraintScoreTracker> scoreTracker )
{
    ASSERT( GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH )
    _scoreTracker = std::move( scoreTracker );
}

bool CdclCore::isDecision( int lit )
{
    return _satSolver->isDecision( lit );
}

double CdclCore::computeDecisionScoreForLiteral( int literal ) const
{
    ASSERT( literal != 0 );
    std::shared_ptr<Query> inputQuery = _engine->getInputQuery();
    NLR::NetworkLevelReasoner *networkLevelReasoner = _engine->getNetworkLevelReasoner();
    networkLevelReasoner->obtainCurrentBounds( *inputQuery );

    setInputBoundsForLiteralInNLR( literal, inputQuery, networkLevelReasoner );
    runSymbolicBoundTightening( networkLevelReasoner );

    return getUpperBoundForOutputVariableFromNLR( networkLevelReasoner );
}

void CdclCore::setInputBoundsForLiteralInNLR(
    int literal,
    const std::shared_ptr<Query> &inputQuery,
    NLR::NetworkLevelReasoner *networkLevelReasoner ) const
{
    const auto &layers = networkLevelReasoner->getLayerIndexToLayer();
    const PiecewiseLinearConstraint *plc = _satSolverVarToPlc[abs( literal )];
    for ( unsigned variable : plc->getParticipatingVariables() )
    {
        NLR::NeuronIndex neuronIndex = networkLevelReasoner->variableToNeuron( variable );
        if ( layers[neuronIndex._layer]->getLayerType() == NLR::Layer::RELU )
        {
            if ( literal < 0 )
                networkLevelReasoner->setBounds( neuronIndex._layer, neuronIndex._neuron, 0, 0 );
            else
                networkLevelReasoner->setBounds(
                    neuronIndex._layer,
                    neuronIndex._neuron,
                    FloatUtils::max( inputQuery->getLowerBound( variable ), 0 ),
                    inputQuery->getUpperBound( variable ) );

            break;
        }
    }
}

void CdclCore::runSymbolicBoundTightening( NLR::NetworkLevelReasoner *networkLevelReasoner ) const
{
    if ( _engine->getSymbolicBoundTighteningType() ==
         SymbolicBoundTighteningType::SYMBOLIC_BOUND_TIGHTENING )
        networkLevelReasoner->symbolicBoundPropagation();
    else if ( _engine->getSymbolicBoundTighteningType() == SymbolicBoundTighteningType::DEEP_POLY )
        networkLevelReasoner->deepPolyPropagation();
}

double CdclCore::getUpperBoundForOutputVariableFromNLR(
    NLR::NetworkLevelReasoner *networkLevelReasoner ) const
{
    Map<Pair<unsigned, Tightening::BoundType>, double> outputBounds;
    networkLevelReasoner->getOutputBounds( outputBounds );

    if ( GlobalConfiguration::CONVERT_VERIFICATION_QUERY_INTO_REACHABILITY_QUERY )
    {
        List<unsigned> outputVariables = _engine->getOutputVariables();
        ASSERT( outputVariables.size() == 1 );
        unsigned outputVariable = outputVariables.front();

        if ( outputBounds.exists( Pair( outputVariable, Tightening::UB ) ) )
            return outputBounds[Pair( outputVariable, Tightening::UB )];
    }
    else
    {
        double outputUb = 0;
        std::shared_ptr<Query> inputQuery = _engine->getInputQuery();
        const NLR::Layer *outputLayer =
            networkLevelReasoner->getLayer( networkLevelReasoner->getNumberOfLayers() - 1 );

        for ( unsigned neuron = 0; neuron < outputLayer->getSize(); ++neuron )
        {
            unsigned variable = outputLayer->neuronToVariable( neuron );

            double lb;
            if ( outputBounds.exists( Pair( variable, Tightening::LB ) ) )
                lb = outputBounds[Pair( variable, Tightening::LB )];
            else
                lb = inputQuery->getLowerBound( variable );
            outputUb -= FloatUtils::max( lb, 0 );

            double ub;
            if ( outputBounds.exists( Pair( variable, Tightening::UB ) ) )
                ub = outputBounds[Pair( variable, Tightening::UB )];
            else
                ub = inputQuery->getUpperBound( variable );
            outputUb -= FloatUtils::max( -ub, 0 );
        }

        return outputUb;
    }

    return FloatUtils::infinity();
}

void CdclCore::computeClauseScores( const Set<int> &clause,
                                    Vector<Pair<double, int>> &clauseScores )
{
    for ( int literal : clause )
    {
        if ( !_decisionScores.exists( literal ) )
            _decisionScores[literal] = computeDecisionScoreForLiteral( literal );
        clauseScores.append( Pair<double, int>( _decisionScores[literal], literal ) );
    }
    clauseScores.sort();
}

void CdclCore::reorderByDecisionLevelIfNecessary( Vector<Pair<double, int>> &clauseScores )
{
    if ( !clauseScores.empty() &&
         clauseScores[0].first() == clauseScores[clauseScores.size() - 1].first() )
    {
        double score = clauseScores[0].first();
        clauseScores.clear();
        for ( unsigned level = 1; level <= _satSolver->getLevel(); ++level )
        {
            ASSERT( _decisionLiterals.exists( level ) );
            clauseScores.append( Pair<double, int>( score, _decisionLiterals[level] ) );
        }
    }
}

void CdclCore::computeShortedClause( Set<int> &clause,
                                     const Vector<Pair<double, int>> &clauseScores,
                                     int propagated_lit ) const
{
    std::shared_ptr<Query> inputQuery = _engine->getInputQuery();
    NLR::NetworkLevelReasoner *networkLevelReasoner = _engine->getNetworkLevelReasoner();

    if ( GlobalConfiguration::CDCL_SHORTEN_CLAUSES_WITH_QUICKXPLAIN )
    {
        clause = quickXplain( Set<int>(), clauseScores, 0, clauseScores.size(), propagated_lit );
    }
    else
    {
        if ( propagated_lit != 0 )
            setInputBoundsForLiteralInNLR( -propagated_lit, inputQuery, networkLevelReasoner );

        for ( const auto &pair : clauseScores )
        {
            double score = pair.first();
            int literal = pair.second();

            ASSERT( literal != 0 );

            clause.insert( literal );

            double outputUb = FloatUtils::infinity();
            if ( clause.size() == 1 )
                outputUb = score;
            else
            {
                setInputBoundsForLiteralInNLR( literal, inputQuery, networkLevelReasoner );
                runSymbolicBoundTightening( networkLevelReasoner );
                outputUb = getUpperBoundForOutputVariableFromNLR( networkLevelReasoner );
            }

            if ( GlobalConfiguration::CONVERT_VERIFICATION_QUERY_INTO_REACHABILITY_QUERY )
            {
                List<unsigned> outputVariables = _engine->getOutputVariables();
                ASSERT( outputVariables.size() == 1 );
                unsigned outputVariable = outputVariables.front();
                if ( FloatUtils::lt( outputUb, inputQuery->getLowerBound( outputVariable ) ) )
                    break;
            }
            else
            {
                if ( FloatUtils::isNegative( outputUb ) )
                    break;
            }
        }
    }
}

bool CdclCore::checkIfShouldSkipClauseShortening( const Set<int> &clause )
{
    if ( clause.empty() )
        return true;

    std::shared_ptr<Query> inputQuery = _engine->getInputQuery();
    NLR::NetworkLevelReasoner *networkLevelReasoner = _engine->getNetworkLevelReasoner();

    for ( int literal : clause )
        setInputBoundsForLiteralInNLR( literal, inputQuery, networkLevelReasoner );

    runSymbolicBoundTightening( networkLevelReasoner );
    double outputUb = getUpperBoundForOutputVariableFromNLR( networkLevelReasoner );

    if ( GlobalConfiguration::CONVERT_VERIFICATION_QUERY_INTO_REACHABILITY_QUERY )
    {
        List<unsigned> outputVariables = _engine->getOutputVariables();
        ASSERT( outputVariables.size() == 1 );
        unsigned outputVariable = outputVariables.front();
        if ( outputUb >= inputQuery->getLowerBound( outputVariable ) )
            return true;
    }
    else
    {
        if ( outputUb >= 0 )
            return true;
    }

    return false;
}

Set<int> CdclCore::quickXplain( const Set<int> &currentClause,
                                const Vector<Pair<double, int>> &clauseScores,
                                unsigned int startIdx,
                                unsigned int endIdx,
                                int propagated_lit ) const
{
    std::shared_ptr<Query> inputQuery = _engine->getInputQuery();
    NLR::NetworkLevelReasoner *networkLevelReasoner = _engine->getNetworkLevelReasoner();
    networkLevelReasoner->obtainCurrentBounds( *inputQuery );

    if ( propagated_lit != 0 )
        setInputBoundsForLiteralInNLR( -propagated_lit, inputQuery, networkLevelReasoner );

    for ( int lit : currentClause )
        setInputBoundsForLiteralInNLR( lit, inputQuery, networkLevelReasoner );

    runSymbolicBoundTightening( networkLevelReasoner );
    double outputUb = getUpperBoundForOutputVariableFromNLR( networkLevelReasoner );

    if ( !currentClause.empty() && outputUb < 0 )
        return Set<int>();

    if ( endIdx == startIdx + 1 )
        return Set<int>( { clauseScores[startIdx].second() } );

    unsigned mid = ( startIdx + endIdx ) / 2;
    Set<int> currentClause1;
    for ( unsigned i = startIdx; i < mid; ++i )
        currentClause1.insert( clauseScores[i].second() );

    Set<int> clause1 = quickXplain( currentClause1, clauseScores, mid, endIdx, propagated_lit );
    Set<int> clause2 = quickXplain( clause1, clauseScores, startIdx, mid, propagated_lit );

    return clause1 + clause2;
}

void CdclCore::reset()
{
    delete _satSolver;
    _satSolver = new CadicalWrapper( this, this, this );

    for ( unsigned var : _satSolverVarToPlc.keys() )
        if ( var != 0 )
            _satSolver->addObservedVar( (int)var );

    _fixedCadicalVars.clear();
    _literalsToPropagate.clear();
    _externalClauseToAdd.clear();
    _reasonClauseLiterals.clear();
    _isReasonClauseInitialized = false;

    _numOfClauses = 0;
    _literalToClauses.clear();
    _vsidsDecayThreshold = 0;
    _vsidsDecayCounter = 0;

    _restarts = 1;
    _restartLimit = 512 * luby( 1 );
    _numOfConflictClauses = 0;
    _shouldRestart = false;

    _largestAssignmentSoFar.clear();
    _decisionLiterals.clear();
    _decisionIndex = 0;
    _decisionScores.clear();

    _lastSharedClauseIndexAdded = 0;
    _sharedClauseAdded.clear();

    _sncSplitLiterals.clear();
}

const PiecewiseLinearConstraint *CdclCore::getPlc( unsigned int var ) const
{
    return _satSolverVarToPlc[var];
}

void CdclCore::connectProofWriter( const AletheProofWriter *writer ) const
{
    _satSolver->connectProofWriter( writer );
}

const Vector<int> &CdclCore::getSncLits() const
{
    return _sncSplitLiterals;
}

#endif