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
#include <thread>
#include <utility>
#include <vector>

std::atomic<unsigned> CdclCore::numCdclCores{ 0 };
Map<unsigned, Set<int>> CdclCore::sharedClauses{};

CdclCore::PrRebuildRole CdclCore::prRebuildRole = CdclCore::PR_REBUILD_OFF;
Vector<Set<int>> CdclCore::prSeedClauses{};
List<Set<int>> CdclCore::prHandoffSelected{};
Vector<Set<int>> CdclCore::prHandoffCarry{};
bool CdclCore::prHandoffValid = false;
unsigned CdclCore::prHandoffMaxVar = 0;
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
    , _prLearner()
    , _prStopRequested( false )
    , _prConflictLimit( 0 )
    , _prConflictCount( 0 )
    , _prSeedsAdded( false )
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

    if ( _prLearner.isHarvesting() )
    {
        Vector<int> trail;
        for ( const auto &p : _assignedLiterals )
        {
            int lit = p.first;
            if ( !isLiteralFixed( lit ) && !isLiteralFixed( -lit ) )
                trail.append( lit );
        }
        _prLearner.observeTrail( trail );
    }

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

    if ( _prStopRequested )
        return false;

    if ( _statistics )
        _statistics->incUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES );
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

    if ( _prStopRequested )
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
        if ( _statistics )
            _statistics->incUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES );

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

    if ( _prLearner.isHarvesting() )
    {
        _prLearner.addPoolClauseFromCube( clause );

        if ( ++_prConflictCount >= _prConflictLimit )
            _prStopRequested = true;
    }

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

    ++_numOfConflictClauses;
    if ( _numOfConflictClauses == _restartLimit )
        _shouldRestart = true;

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

    // PR driver: install the seed clauses (phase A's carry + PR clauses for
    // the phase B engine) into this fresh core exactly once.
    if ( prRebuildRole != PR_REBUILD_OFF && !_prSeedsAdded )
    {
        for ( const Set<int> &clause : prSeedClauses )
            _satSolver->addClause( clause );
        if ( _engine->getVerbosity() > 0 && !prSeedClauses.empty() )
        {
            printf( "PR: seeded fresh core with %u clauses\n", prSeedClauses.size() );
            fflush( stdout );
        }
        _prSeedsAdded = true;
    }

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
    {
        bool initialSolveResult = _engine->solve( _timeoutInSeconds );
        if ( getenv( "PR_DEBUG" ) )
        {
            printf( "PR-debug: initial engine solve returned %d, exit code %d, "
                    "timeout param %.1f\n",
                    (int)initialSolveResult,
                    (int)_engine->getExitCode(),
                    _timeoutInSeconds );
            fflush( stdout );
        }
        if ( initialSolveResult )
        {
            _engine->setExitCode( ExitCode::SAT );
            if ( GlobalConfiguration::WRITE_ALETHE_PROOF && !Options::get()->getBool( Options::DNC_MODE ) )
                _engine->deleteProofIfExists();
            return true;
        }
    }

    // Add the zero literal at the end
    if ( !_literalsToPropagate.empty() )
        _literalsToPropagate.append( Pair<int, unsigned>( 0, _satSolver->getLevel() ) );

    if ( !_externalClauseToAdd.empty() )
    {
        // Declaring UNSAT here is sound only for a genuine ROOT conflict:
        // every literal of the pending clause already falsified at root.
        // Otherwise it is an ordinary lemma the engine's pre-solve deposited
        // — leave it in the buffer (cb_has_external_clause delivers it to
        // the SAT solver) and keep solving. Restarted runs (PR phase B and
        // the debt recursion) reach this point with multi-literal lemmas and
        // were previously declared UNSAT without any search.
        bool rootConflict = true;
        for ( int literal : _externalClauseToAdd )
            if ( !isLiteralFixed( -literal ) )
            {
                rootConflict = false;
                break;
            }
        if ( rootConflict )
        {
            if ( getenv( "PR_DEBUG" ) )
            {
                printf( "PR-debug: pending clause (size %u) is a root conflict - UNSAT\n",
                        _externalClauseToAdd.size() );
                fflush( stdout );
            }
            ASSERT( _engine->getExitCode() == ExitCode::NOT_DONE );
            _engine->setExitCode( ExitCode::UNSAT );
            return false;
        }
        if ( getenv( "PR_DEBUG" ) )
        {
            printf( "PR-debug: pending %u-literal lemma is not a root conflict - "
                    "deferring to the SAT solver; literals:",
                    _externalClauseToAdd.size() );
            for ( int literal : _externalClauseToAdd )
                printf( " %d(fixed:%d,negFixed:%d)",
                        literal,
                        (int)isLiteralFixed( literal ),
                        (int)isLiteralFixed( -literal ) );
            printf( "\n" );
            fflush( stdout );
        }
    }

    Set<int> externalClause;

    externalClause = _satSolver->addExternalNAPClause(
        Options::get()->getString( Options::NAP_EXTERNAL_CLAUSE_FILE_PATH ) );
    if ( !externalClause.empty() )
    {
        _initialClauses.append( externalClause );
        _prLearner.addPoolClause( externalClause );
    }

    externalClause = _satSolver->addExternalNAPClause(
        Options::get()->getString( Options::NAP_EXTERNAL_CLAUSE_FILE_PATH2 ) );
    if ( !externalClause.empty() )
    {
        _initialClauses.append( externalClause );
        _prLearner.addPoolClause( externalClause );
    }

    CDCL_LOG( Stringf( "%u l%d Start solving", _index, _satSolver->getLevel() ).ascii() )
    int result = _satSolver->solve();

    if ( _statistics && _engine->getVerbosity() )
    {
        printf( "\nCdclCore::Final statistics:\n" );
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

bool CdclCore::solveWithPrPreprocessedCDCL( double timeoutInSeconds )
{
    if ( GlobalConfiguration::WRITE_ALETHE_PROOF )
        printf( "Warning: PR clause preprocessing injects clauses that are not justified in the "
                "produced proof; the proof may not check\n" );

    // SOLVE role (phase B engine): plain CDCL solve of the seeded query.
    if ( prRebuildRole == PR_REBUILD_SOLVE )
        return solveWithCDCL( timeoutInSeconds );

    // HARVEST role (phase A engine): run until the conflict budget trips,
    // then hand the harvest to Marabou::solveWithPrRebuild — phase B runs on
    // a fresh engine (the in-process engine restore is corrupt).
    _prConflictLimit =
        (unsigned)Options::get()->getInt( Options::PR_CLAUSE_PREPROCESS_CONFLICTS );
    _prConflictCount = 0;

    struct timespec prStart = TimeUtils::sampleMicro();
    if ( _engine->getVerbosity() > 0 )
    {
        printf( "PR: phase A - harvesting until %u conflicts are learned\n", _prConflictLimit );
        fflush( stdout );
    }

    _prLearner.startHarvest();
    // The node's seed clauses (debt chain + inherited lemmas) are part of
    // the formula this engine solves: the carve must respect them too.
    for ( const Set<int> &clause : prSeedClauses )
        _prLearner.addPoolClause( clause );
    bool result = solveWithCDCL( timeoutInSeconds );

    if ( _engine->getExitCode() != ExitCode::NOT_DONE )
    {
        // Concluded (or timed out) within the conflict budget
        _prLearner.stopHarvest();
        if ( _engine->getVerbosity() > 0 )
        {
            printf( "PR: phase A concluded on its own (t=%.1fs, exit code %d)\n",
                    TimeUtils::timePassed( prStart, TimeUtils::sampleMicro() ) / 1e6,
                    (int)_engine->getExitCode() );
            fflush( stdout );
        }
        return result;
    }

    ASSERT( _prStopRequested )

    List<Set<int>> prClauses = _prLearner.finalizeHarvest();
    Vector<Set<int>> carry = _prLearner.getPoolClauses();
    _prStopRequested = false;

    // top_k_strongest (alpha-beta-CROWN ranking.py): shortest condition
    // first - a tighter implication fires under more partial assignments.
    // Deterministic lexicographic tiebreak; K = 0 injects everything.
    unsigned topK = (unsigned)Options::get()->getInt( Options::PR_CLAUSE_TOP_K );
    if ( topK > 0 && prClauses.size() > topK )
    {
        std::vector<Set<int>> ranked( prClauses.begin(), prClauses.end() );
        std::stable_sort( ranked.begin(),
                          ranked.end(),
                          []( const Set<int> &a, const Set<int> &b ) {
                              if ( a.size() != b.size() )
                                  return a.size() < b.size();
                              auto ia = a.begin();
                              auto ib = b.begin();
                              while ( ia != a.end() && ib != b.end() )
                              {
                                  if ( *ia != *ib )
                                      return *ia < *ib;
                                  ++ia;
                                  ++ib;
                              }
                              return false;
                          } );
        List<Set<int>> capped;
        for ( unsigned i = 0; i < topK; ++i )
            capped.append( ranked[i] );
        prClauses = capped;
    }

    // Instrumentation: dump the injection package for offline analysis.
    if ( const char *dumpPath = getenv( "PR_DUMP" ) )
    {
        FILE *f = fopen( dumpPath, "w" );
        if ( f )
        {
            auto writeClauseArray = [f]( const char *key, const auto &clauses ) {
                fprintf( f, "\"%s\": [", key );
                bool firstClause = true;
                for ( const Set<int> &clause : clauses )
                {
                    fprintf( f, "%s[", firstClause ? "" : "," );
                    bool firstLit = true;
                    for ( int lit : clause )
                    {
                        fprintf( f, "%s%d", firstLit ? "" : ",", lit );
                        firstLit = false;
                    }
                    fprintf( f, "]" );
                    firstClause = false;
                }
                fprintf( f, "]" );
            };
            fprintf( f, "{" );
            writeClauseArray( "carry", carry );
            fprintf( f, "," );
            writeClauseArray( "pr_clauses", prClauses );
            fprintf( f, ",\"root_units\": [" );
            bool first = true;
            for ( int literal : _fixedCadicalVars )
            {
                fprintf( f, "%s%d", first ? "" : ",", literal );
                first = false;
            }
            fprintf( f, "]}\n" );
            fclose( f );
            printf( "PR: dumped %u carry + %u PR clauses to %s\n",
                    carry.size(),
                    prClauses.size(),
                    dumpPath );
        }
    }

    prHandoffSelected = prClauses;
    prHandoffCarry = carry;
    prHandoffMaxVar = 0;
    for ( unsigned var : _satSolverVarToPlc.keys() )
        if ( var > prHandoffMaxVar )
            prHandoffMaxVar = var;
    prHandoffValid = true;

    if ( _engine->getVerbosity() > 0 )
    {
        printf( "PR: phase A done (t=%.1fs) - observed %u trails, harvested %u candidates; "
                "handing off %u carry clauses and %u PR clauses\n",
                TimeUtils::timePassed( prStart, TimeUtils::sampleMicro() ) / 1e6,
                _prLearner.getNumObservedTrails(),
                _prLearner.getNumHarvestedCandidates(),
                carry.size(),
                prClauses.size() );
        fflush( stdout );
    }
    return result;
}

unsigned CdclCore::dischargeDebtCubes( const List<Set<int>> &cubes, List<Set<int>> &unpaid )
{
    // Root-entailed phases from preprocessing: a cube pinning the opposite
    // phase prunes an already-empty region. Snapshot before reset() clears
    // the list.
    Set<int> rootLiterals;
    for ( const auto &pair : _literalsToPropagate )
        if ( pair.first() != 0 )
            rootLiterals.insert( pair.first() );

    if ( !_satSolver )
        reset();

    unsigned discharged = 0;
    for ( const Set<int> &cube : cubes )
    {
        bool refuted = false;
        for ( int literal : cube )
            if ( rootLiterals.exists( -literal ) )
            {
                refuted = true;
                break;
            }

        if ( !refuted )
        {
            _engine->preContextPushHook();
            pushContext();
            try
            {
                for ( int literal : cube )
                    notifySingleAssignment( literal, false );
                refuted = !_engine->propagateBoundManagerTightenings() ||
                          !_externalClauseToAdd.empty();
            }
            catch ( const InfeasibleQueryException & )
            {
                refuted = true;
            }
            _externalClauseToAdd.clear();
            popContextTo( 0 );
            _engine->postContextPopHook();
        }

        if ( refuted )
            ++discharged;
        else
            unpaid.append( cube );
    }
    return discharged;
}

void CdclCore::addLiteralToPropagate( int literal )
{
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

void CdclCore::addDecisionBasedConflictClause()
{
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
    return _prStopRequested || _engine->getExitCode() != ExitCode::NOT_DONE;
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