/*********************                                                        */
/*! \file MarabouCore.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Christopher Lazarus, Andrew Wu, Shantanu Thakoor, Kyle Julian
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

#include "AcasParser.h"
#include "BilinearConstraint.h"
#include "CommonError.h"
#include "DisjunctionConstraint.h"
#include "DnCManager.h"
#include "Engine.h"
#include "FloatUtils.h"
#include "InputParserError.h"
#include "InputQuery.h"
#include "LeakyReluConstraint.h"
#include "MString.h"
#include "MarabouError.h"
#include "MarabouMain.h"
#include "MaxConstraint.h"
#include "NonlinearConstraint.h"
#include "Options.h"
#include "PhaseFixTrie.h"
#include "PiecewiseLinearConstraint.h"
#include "PropertyParser.h"
#include "QueryLoader.h"
#include "ReluConstraint.h"
#include "RoundConstraint.h"
#include "Set.h"
#include "SigmoidConstraint.h"
#include "SignConstraint.h"
#include "SnCDivideStrategy.h"
#include "SoftmaxConstraint.h"
#include "VnnLibParser.h"

#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <set>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>
#include <vector>

#ifdef _WIN32
#define STDOUT_FILENO 1
#else
#include <unistd.h>
#endif

namespace py = pybind11;

using PhaseFix = std::pair<unsigned, bool>; // <variable index, phase>

int maraboupyMain( std::vector<std::string> args )
{
    int argc = args.size();
    char **argv = new char *[args.size()];
    for ( int index = 0; index < args.size(); ++index )
    {
        argv[index] = (char *)args[index].c_str();
    }
    return marabouMain( argc, argv );
}

int redirectOutputToFile( std::string outputFilePath )
{
    // Redirect standard output to a file
    int outputFile = open( outputFilePath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644 );
    if ( outputFile < 0 )
    {
        printf( "Error redirecting output to file\n" );
        exit( 1 );
    }

    int outputStream = dup( STDOUT_FILENO );
    if ( outputStream < 0 )
    {
        printf( "Error duplicating standard output\n" );
        exit( 1 );
    }

    if ( dup2( outputFile, STDOUT_FILENO ) < 0 )
    {
        printf( "Error duplicating to standard output\n" );
        exit( 1 );
    }

    close( outputFile );
    return outputStream;
}

void restoreOutputStream( int outputStream )
{
    // Restore standard output
    fflush( stdout );
    if ( dup2( outputStream, STDOUT_FILENO ) < 0 )
    {
        printf( "Error restoring output stream\n" );
        exit( 1 );
    }
    close( outputStream );
}

void addClipConstraint( InputQuery &ipq,
                        unsigned var1,
                        unsigned var2,
                        double floor,
                        double ceiling )
{
    ipq.addClipConstraint( var1, var2, floor, ceiling );
}

void addLeakyReluConstraint( InputQuery &ipq, unsigned var1, unsigned var2, double slope )
{
    PiecewiseLinearConstraint *r = new LeakyReluConstraint( var1, var2, slope );
    ipq.addPiecewiseLinearConstraint( r );
}

void addReluConstraint( InputQuery &ipq, unsigned var1, unsigned var2 )
{
    PiecewiseLinearConstraint *r = new ReluConstraint( var1, var2 );
    ipq.addPiecewiseLinearConstraint( r );
}

void addRoundConstraint( InputQuery &ipq, unsigned var1, unsigned var2 )
{
    NonlinearConstraint *r = new RoundConstraint( var1, var2 );
    ipq.addNonlinearConstraint( r );
}

void addBilinearConstraint( InputQuery &ipq, unsigned var1, unsigned var2, unsigned var3 )
{
    NonlinearConstraint *r = new BilinearConstraint( var1, var2, var3 );
    ipq.addNonlinearConstraint( r );
}

void addSigmoidConstraint( InputQuery &ipq, unsigned var1, unsigned var2 )
{
    NonlinearConstraint *s = new SigmoidConstraint( var1, var2 );
    ipq.addNonlinearConstraint( s );
}

void addSignConstraint( InputQuery &ipq, unsigned var1, unsigned var2 )
{
    PiecewiseLinearConstraint *r = new SignConstraint( var1, var2 );
    ipq.addPiecewiseLinearConstraint( r );
}

void addMaxConstraint( InputQuery &ipq, std::set<unsigned> elements, unsigned v )
{
    Set<unsigned> e;
    for ( unsigned var : elements )
        e.insert( var );
    PiecewiseLinearConstraint *m = new MaxConstraint( v, e );
    ipq.addPiecewiseLinearConstraint( m );
}

void addSoftmaxConstraint( InputQuery &ipq,
                           std::list<unsigned> inputs,
                           std::list<unsigned> outputs )
{
    Vector<unsigned> inputList;
    for ( const auto &e : inputs )
        inputList.append( e );

    Vector<unsigned> outputList;
    for ( const auto &e : outputs )
        outputList.append( e );

    SoftmaxConstraint *s = new SoftmaxConstraint( inputList, outputList );
    ipq.addNonlinearConstraint( s );
}

void addAbsConstraint( InputQuery &ipq, unsigned b, unsigned f )
{
    ipq.addPiecewiseLinearConstraint( new AbsoluteValueConstraint( b, f ) );
}

void loadProperty( InputQuery &inputQuery, std::string propertyFilePath )
{
    String propertyFilePathM = String( propertyFilePath );
    if ( propertyFilePath != "" )
    {
        printf( "Property: %s\n", propertyFilePathM.ascii() );
        if ( propertyFilePathM.endsWith( ".vnnlib" ) )
        {
            VnnLibParser().parse( propertyFilePathM, inputQuery );
        }
        else
        {
            PropertyParser().parse( propertyFilePathM, inputQuery );
        }
    }
    else
        printf( "Property: None\n" );
}

bool createInputQuery( InputQuery &inputQuery,
                       std::string networkFilePath,
                       std::string propertyFilePath )
{
    try
    {
        AcasParser *acasParser = new AcasParser( String( networkFilePath ) );
        acasParser->generateQuery( inputQuery );

        String propertyFilePathM = String( propertyFilePath );
        if ( propertyFilePath != "" )
        {
            printf( "Property: %s\n", propertyFilePathM.ascii() );
            if ( propertyFilePathM.endsWith( ".vnnlib" ) )
            {
                VnnLibParser().parse( propertyFilePathM, inputQuery );
            }
            else
            {
                PropertyParser().parse( propertyFilePathM, inputQuery );
            }
        }
        else
            printf( "Property: None\n" );
    }
    catch ( const InputParserError &e )
    {
        printf( "Caught an InputParserError. Code: %u. Message: %s\n",
                e.getCode(),
                e.getUserMessage() );
        return false;
    }
    return true;
}

void addDisjunctionConstraint( InputQuery &ipq, const std::list<std::list<Equation>> &disjuncts )
{
    List<PiecewiseLinearCaseSplit> disjunctList;
    for ( const auto &disjunct : disjuncts )
    {
        PiecewiseLinearCaseSplit split;
        for ( const auto &eq : disjunct )
        {
            if ( eq._addends.size() == 1 )
            {
                // Add bounds as tightenings
                unsigned var = eq._addends.front()._variable;
                double coeff = eq._addends.front()._coefficient;
                if ( coeff == 0 )
                    throw CommonError( CommonError::DIVISION_BY_ZERO,
                                       "AddDisjunctionConstraint: zero coefficient encountered" );
                double scalar = eq._scalar / coeff;
                Equation::EquationType type = eq._type;

                if ( type == Equation::EQ )
                {
                    split.storeBoundTightening( Tightening( var, scalar, Tightening::LB ) );
                    split.storeBoundTightening( Tightening( var, scalar, Tightening::UB ) );
                }
                else if ( ( type == Equation::GE && coeff > 0 ) ||
                          ( type == Equation::LE && coeff < 0 ) )
                    split.storeBoundTightening( Tightening( var, scalar, Tightening::LB ) );
                else
                    split.storeBoundTightening( Tightening( var, scalar, Tightening::UB ) );
            }
            else
            {
                split.addEquation( eq );
            }
        }
        disjunctList.append( split );
    }
    ipq.addPiecewiseLinearConstraint( new DisjunctionConstraint( disjunctList ) );
}

struct MarabouOptions
{
    MarabouOptions()
        : _snc( Options::get()->getBool( Options::DNC_MODE ) )
        , _restoreTreeStates( Options::get()->getBool( Options::RESTORE_TREE_STATES ) )
        , _solveWithMILP( Options::get()->getBool( Options::SOLVE_WITH_MILP ) )
        , _dumpBounds( Options::get()->getBool( Options::DUMP_BOUNDS ) )
        , _numWorkers( Options::get()->getInt( Options::NUM_WORKERS ) )
        , _numBlasThreads( Options::get()->getInt( Options::NUM_BLAS_THREADS ) )
        , _initialTimeout( Options::get()->getInt( Options::INITIAL_TIMEOUT ) )
        , _initialDivides( Options::get()->getInt( Options::NUM_INITIAL_DIVIDES ) )
        , _onlineDivides( Options::get()->getInt( Options::NUM_ONLINE_DIVIDES ) )
        , _verbosity( Options::get()->getInt( Options::VERBOSITY ) )
        , _timeoutInSeconds( Options::get()->getInt( Options::TIMEOUT ) )
        , _splitThreshold( Options::get()->getInt( Options::CONSTRAINT_VIOLATION_THRESHOLD ) )
        , _numSimulations( Options::get()->getInt( Options::NUMBER_OF_SIMULATIONS ) )
        , _performLpTighteningAfterSplit(
              Options::get()->getBool( Options::PERFORM_LP_TIGHTENING_AFTER_SPLIT ) )
        , _timeoutFactor( Options::get()->getFloat( Options::TIMEOUT_FACTOR ) )
        , _preprocessorBoundTolerance(
              Options::get()->getFloat( Options::PREPROCESSOR_BOUND_TOLERANCE ) )
        , _milpSolverTimeout( Options::get()->getFloat( Options::MILP_SOLVER_TIMEOUT ) )
        , _splittingStrategyString(
              Options::get()->getString( Options::SPLITTING_STRATEGY ).ascii() )
        , _sncSplittingStrategyString(
              Options::get()->getString( Options::SNC_SPLITTING_STRATEGY ).ascii() )
        , _tighteningStrategyString(
              Options::get()->getString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE ).ascii() )
        , _milpTighteningString(
              Options::get()->getString( Options::MILP_SOLVER_BOUND_TIGHTENING_TYPE ).ascii() )
        , _lpSolverString( Options::get()->getString( Options::LP_SOLVER ).ascii() )
        , _produceProofs( Options::get()->getBool( Options::PRODUCE_PROOFS ) )
        , _coreMinimizationTimeout( 0.0 )
        , _coreMinimizationLimit( 3 )
        , _coreMinimizationSingleSolveTimeout( 1.0 ){};

    void setOptions()
    {
        // Bool options
        Options::get()->setBool( Options::DNC_MODE, _snc );
        Options::get()->setBool( Options::RESTORE_TREE_STATES, _restoreTreeStates );
        Options::get()->setBool( Options::SOLVE_WITH_MILP, _solveWithMILP );
        Options::get()->setBool( Options::DUMP_BOUNDS, _dumpBounds );
        Options::get()->setBool( Options::PERFORM_LP_TIGHTENING_AFTER_SPLIT,
                                 _performLpTighteningAfterSplit );
        Options::get()->setBool( Options::PRODUCE_PROOFS, _produceProofs );

        // int options
        Options::get()->setInt( Options::NUM_WORKERS, _numWorkers );
        Options::get()->setInt( Options::NUM_BLAS_THREADS, _numBlasThreads );
        Options::get()->setInt( Options::INITIAL_TIMEOUT, _initialTimeout );
        Options::get()->setInt( Options::NUM_INITIAL_DIVIDES, _initialDivides );
        Options::get()->setInt( Options::NUM_ONLINE_DIVIDES, _onlineDivides );
        Options::get()->setInt( Options::VERBOSITY, _verbosity );
        Options::get()->setInt( Options::TIMEOUT, _timeoutInSeconds );
        Options::get()->setInt( Options::CONSTRAINT_VIOLATION_THRESHOLD, _splitThreshold );

        // float options
        Options::get()->setFloat( Options::TIMEOUT_FACTOR, _timeoutFactor );
        Options::get()->setFloat( Options::PREPROCESSOR_BOUND_TOLERANCE,
                                  _preprocessorBoundTolerance );
        Options::get()->setFloat( Options::MILP_SOLVER_TIMEOUT, _milpSolverTimeout );

        // string options
        Options::get()->setString( Options::SPLITTING_STRATEGY, _splittingStrategyString );
        Options::get()->setString( Options::SNC_SPLITTING_STRATEGY, _sncSplittingStrategyString );
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE,
                                   _tighteningStrategyString );
        Options::get()->setString( Options::MILP_SOLVER_BOUND_TIGHTENING_TYPE,
                                   _milpTighteningString );
        Options::get()->setString( Options::LP_SOLVER, _lpSolverString );
    }

    bool _snc;
    bool _restoreTreeStates;
    bool _solveWithMILP;
    bool _dumpBounds;
    bool _performLpTighteningAfterSplit;
    bool _produceProofs;
    unsigned _numWorkers;
    unsigned _numBlasThreads;
    unsigned _initialTimeout;
    unsigned _initialDivides;
    unsigned _onlineDivides;
    unsigned _verbosity;
    unsigned _timeoutInSeconds;
    unsigned _splitThreshold;
    unsigned _numSimulations;
    float _timeoutFactor;
    float _preprocessorBoundTolerance;
    float _milpSolverTimeout;
    std::string _splittingStrategyString;
    std::string _sncSplittingStrategyString;
    std::string _tighteningStrategyString;
    std::string _milpTighteningString;
    std::string _lpSolverString;
    float _coreMinimizationTimeout;
    unsigned _coreMinimizationLimit;
    float _coreMinimizationSingleSolveTimeout;
};


std::string exitCodeToString( IEngine::ExitCode code )
{
    switch ( code )
    {
    case IEngine::UNSAT:
        return "unsat";
    case IEngine::SAT:
        return "sat";
    case IEngine::ERROR:
        return "ERROR";
    case IEngine::UNKNOWN:
        return "UNKNOWN";
    case IEngine::TIMEOUT:
        return "TIMEOUT";
    case IEngine::QUIT_REQUESTED:
        return "QUIT_REQUESTED";
    default:
        return "UNKNOWN";
    }
}

/* The default parameters here are just for readability, you should specify
 * them in the to make them work*/
std::tuple<std::string, std::map<int, double>, Statistics, PhaseFixTrie *>
solve( InputQuery &inputQuery,
       MarabouOptions &options,
       std::string redirect = "",
       PhaseFixTrie *phaseFixTrie = nullptr )
{
    // Arguments: InputQuery object, filename to redirect output
    // Returns: map from variable number to value
    std::string resultString = "";
    std::map<int, double> ret;
    Statistics retStats;
    int output = -1;
    if ( redirect.length() > 0 )
        output = redirectOutputToFile( redirect );
    try
    {
        options.setOptions();

        bool dnc = Options::get()->getBool( Options::DNC_MODE );

        Engine engine;

        if ( phaseFixTrie != nullptr )
        {
            // Move the contents of the input trie into the engine
            // The input pointer is still owned by Python and will be cleaned up by Python
            engine.setPhaseFixTrie( phaseFixTrie );
        }

        if ( !engine.processInputQuery( inputQuery ) )
        {
            // Create heap-allocated copy of the trie
            PhaseFixTrie *resultTrie = new PhaseFixTrie( std::move( *engine.getPhaseFixTrie() ) );
            return std::make_tuple( exitCodeToString( engine.getExitCode() ),
                                    ret,
                                    *( engine.getStatistics() ),
                                    resultTrie );
        }
        if ( dnc )
        {
            auto dncManager = std::unique_ptr<DnCManager>( new DnCManager( &inputQuery ) );

            dncManager->solve();
            resultString = dncManager->getResultString().ascii();
            switch ( dncManager->getExitCode() )
            {
            case DnCManager::SAT:
            {
                retStats = Statistics();
                dncManager->getSolution( ret, inputQuery );
                break;
            }
            case DnCManager::TIMEOUT:
            {
                retStats = Statistics();
                retStats.timeout();
                PhaseFixTrie *resultTrie =
                    new PhaseFixTrie( std::move( *engine.getPhaseFixTrie() ) );
                return std::make_tuple( resultString, ret, retStats, resultTrie );
            }
            default:
                PhaseFixTrie *resultTrie =
                    new PhaseFixTrie( std::move( *engine.getPhaseFixTrie() ) );
                return std::make_tuple(
                    resultString, ret, Statistics(), resultTrie ); // TODO: meaningful
                                                                   // DnCStatistics
            }
        }
        else
        {
            unsigned timeoutInSeconds = Options::get()->getInt( Options::TIMEOUT );
            engine.solve( timeoutInSeconds );

            resultString = exitCodeToString( engine.getExitCode() );

            if ( engine.getExitCode() == Engine::SAT )
            {
                engine.extractSolution( inputQuery );
                for ( unsigned int i = 0; i < inputQuery.getNumberOfVariables(); ++i )
                    ret[i] = inputQuery.getSolutionValue( i );
            }

            retStats = *( engine.getStatistics() );
        }

        // Create heap-allocated copy of the trie
        PhaseFixTrie *resultTrie = new PhaseFixTrie( std::move( *engine.getPhaseFixTrie() ) );
        if ( output != -1 )
            restoreOutputStream( output );
        return std::make_tuple( resultString, ret, retStats, resultTrie );
    }
    catch ( const MarabouError &e )
    {
        fprintf( stderr,
                 "Caught a MarabouError. Code: %u. Message: %s\n",
                 e.getCode(),
                 e.getUserMessage() );
        // Return nullptr on error, don't leak the old trie
        if ( output != -1 )
            restoreOutputStream( output );
        return std::make_tuple( "ERROR", ret, retStats, nullptr );
    }
}

// Helper to create an InputQuery from a Query object
InputQuery *inputQueryFromQuery( Query *query )
{
    InputQuery *inputQuery = new InputQuery();
    inputQuery->setNumberOfVariables( query->getNumberOfVariables() );

    for ( const auto &equation : query->getEquations() )
        inputQuery->addEquation( equation );

    for ( const auto &pair : query->getLowerBounds() )
        inputQuery->setLowerBound( pair.first, pair.second );

    for ( const auto &pair : query->getUpperBounds() )
        inputQuery->setUpperBound( pair.first, pair.second );

    for ( const auto &constraint : query->getPiecewiseLinearConstraints() )
        inputQuery->addPiecewiseLinearConstraint( constraint->duplicateConstraint() );

    for ( const auto &constraint : query->getNonlinearConstraints() )
        inputQuery->addNonlinearConstraint( constraint->duplicateConstraint() );

    for ( const auto &pair : query->_variableToInputIndex )
        inputQuery->markInputVariable( pair.first, pair.second );

    for ( const auto &pair : query->_variableToOutputIndex )
        inputQuery->markOutputVariable( pair.first, pair.second );

    return inputQuery;
}

// Helper function to check if a set of phase fixes is UNSAT
bool isUnsat( Query *baseQuery, const std::vector<PhaseFix> &fixes, MarabouOptions &options )
{
    // Create a fresh InputQuery from the base Query
    std::unique_ptr<InputQuery> inputQuery( inputQueryFromQuery( baseQuery ) );

    // Apply phase fixes as bounds
    for ( const auto &fix : fixes )
    {
        unsigned var = fix.first;
        bool active = fix.second;

        if ( active )
            inputQuery->tightenLowerBound( var, 0.0 );
        else
            inputQuery->tightenUpperBound( var, 0.0 );
    }

    Engine engine;
    options.setOptions();

    if ( !engine.processInputQuery( *inputQuery ) )
        return true; // UNSAT during preprocessing

    engine.solve( options._coreMinimizationSingleSolveTimeout );

    return engine.getExitCode() == IEngine::UNSAT;
}

// Recursive QuickXPlain algorithm
std::vector<PhaseFix> quickXPlain( Query *baseQuery,
                                   MarabouOptions &options,
                                   const std::vector<PhaseFix> &background,
                                   const std::vector<PhaseFix> &candidates,
                                   std::chrono::steady_clock::time_point globalStart,
                                   double timeoutInSeconds )
{
    // Check timeout
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - globalStart;
    if ( timeoutInSeconds > 0 && elapsed.count() > timeoutInSeconds )
        return candidates; // Timeout reached, return best effort

    // If background is already UNSAT, return empty core
    if ( candidates.empty() && isUnsat( baseQuery, background, options ) )
        return {};

    // If background + candidates is SAT, then no UNSAT core exists in candidates
    std::vector<PhaseFix> testSet = background;
    testSet.insert( testSet.end(), candidates.begin(), candidates.end() );
    if ( !isUnsat( baseQuery, testSet, options ) )
        return candidates; // Should not happen if initial call is UNSAT

    // If candidates is a singleton, it's the core
    if ( candidates.size() == 1 )
        return candidates;

    // Split candidates
    size_t k = candidates.size() / 2;
    std::vector<PhaseFix> c1( candidates.begin(), candidates.begin() + k );
    std::vector<PhaseFix> c2( candidates.begin() + k, candidates.end() );

    // Check if background + c2 is UNSAT (meaning c1 is irrelevant)
    std::vector<PhaseFix> backgroundPlusC2 = background;
    backgroundPlusC2.insert( backgroundPlusC2.end(), c2.begin(), c2.end() );

    if ( isUnsat( baseQuery, backgroundPlusC2, options ) )
    {
        // c1 is irrelevant, recurse on c2
        return quickXPlain( baseQuery, options, background, c2, globalStart, timeoutInSeconds );
    }

    // Check if background + c1 is UNSAT (meaning c2 is irrelevant)
    std::vector<PhaseFix> backgroundPlusC1 = background;
    backgroundPlusC1.insert( backgroundPlusC1.end(), c1.begin(), c1.end() );

    if ( isUnsat( baseQuery, backgroundPlusC1, options ) )
    {
        // c2 is irrelevant, recurse on c1
        return quickXPlain( baseQuery, options, background, c1, globalStart, timeoutInSeconds );
    }

    // Both parts are needed. Find minimal subset of c1 needed for c2, and vice versa.
    std::vector<PhaseFix> c2_prime =
        quickXPlain( baseQuery, options, backgroundPlusC1, c2, globalStart, timeoutInSeconds );

    std::vector<PhaseFix> backgroundPlusC2Prime = background;
    backgroundPlusC2Prime.insert( backgroundPlusC2Prime.end(), c2_prime.begin(), c2_prime.end() );

    std::vector<PhaseFix> c1_prime =
        quickXPlain( baseQuery, options, backgroundPlusC2Prime, c1, globalStart, timeoutInSeconds );

    std::vector<PhaseFix> result = c1_prime;
    result.insert( result.end(), c2_prime.begin(), c2_prime.end() );
    return result;
}

void minimizeUnsatCores( InputQuery &inputQuery,
                         MarabouOptions &options,
                         PhaseFixTrie *phaseFixTrie )
{
    // Create a base Query object for cloning
    std::unique_ptr<Query> baseQuery( inputQuery.generateQuery() );

    std::vector<std::vector<PhaseFix>> unsatPrefixes = phaseFixTrie->dumpUnsatPrefixes();
    unsigned initialCount = unsatPrefixes.size();
    unsigned minimizedCount = 0;

    printf( "Minimizing up to %u unsat cores from %u total...\n",
            options._coreMinimizationLimit,
            initialCount );

    // Sort by size to prioritize smaller cores
    std::sort( unsatPrefixes.begin(),
               unsatPrefixes.end(),
               []( const std::vector<PhaseFix> &a, const std::vector<PhaseFix> &b ) {
                   return a.size() < b.size();
               } );

    // Limit
    unsigned limit = options._coreMinimizationLimit;
    if ( unsatPrefixes.size() < limit )
        limit = unsatPrefixes.size();

    auto globalStart = std::chrono::steady_clock::now();

    for ( unsigned i = 0; i < limit; ++i )
    {
        // Check timeout before starting next core
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - globalStart;
        if ( options._coreMinimizationTimeout > 0 &&
             elapsed.count() > options._coreMinimizationTimeout )
        {
            printf( "Global minimization timeout reached.\n" );
            break;
        }

        const auto &prefix = unsatPrefixes[i];

        // Run QuickXPlain
        std::vector<PhaseFix> background;
        std::vector<PhaseFix> smallerCore = quickXPlain( baseQuery.get(),
                                                         options,
                                                         background,
                                                         prefix,
                                                         globalStart,
                                                         options._coreMinimizationTimeout );

        if ( smallerCore.size() < prefix.size() )
        {
            printf( "*** QuickXPlain minimized core from %lu to %lu  ***\n",
                    prefix.size(),
                    smallerCore.size() );
            minimizedCount++;
        }

        // Insert smaller core into the trie
        phaseFixTrie->insertUnsatPrefix( smallerCore );
    }
    printf(
        "Minimization complete. Minimized %u out of %u attempted cores.\n", minimizedCount, limit );
}
std::tuple<std::string, std::map<int, std::tuple<double, double>>, Statistics>
calculateBounds( InputQuery &inputQuery, MarabouOptions &options, std::string redirect = "" )
{
    // Arguments: InputQuery object, filename to redirect output
    // Returns: map from variable number to value
    std::string resultString = "";
    std::map<int, std::tuple<double, double>> ret;
    Statistics retStats;
    int output = -1;
    if ( redirect.length() > 0 )
        output = redirectOutputToFile( redirect );
    try
    {
        options.setOptions();

        bool dnc = Options::get()->getBool( Options::DNC_MODE );

        Engine engine;

        if ( !engine.calculateBounds( inputQuery ) )
        {
            std::string exitCode = exitCodeToString( engine.getExitCode() );
            return std::make_tuple( exitCode, ret, *( engine.getStatistics() ) );
        }

        // Extract bounds
        engine.extractBounds( inputQuery );
        for ( unsigned int i = 0; i < inputQuery.getNumberOfVariables(); ++i )
        {
            // set lower bound and upper bound in tuple
            ret[i] =
                std::make_tuple( inputQuery.getLowerBound( i ), inputQuery.getUpperBound( i ) );
        }
    }
    catch ( const MarabouError &e )
    {
        printf( "Caught a MarabouError. Code: %u. Message: %s\n", e.getCode(), e.getUserMessage() );
        return std::make_tuple( "ERROR", ret, retStats );
    }
    if ( output != -1 )
        restoreOutputStream( output );
    return std::make_tuple( resultString, ret, retStats );
}

void saveQuery( InputQuery &inputQuery, std::string filename )
{
    inputQuery.saveQuery( String( filename ) );
}

void saveQueryAsSmtLib( InputQuery &query, std::string filename )
{
    query.saveQueryAsSmtLib( String( filename ) );
}

void loadQuery( std::string filename, InputQuery &inputQuery )
{
    return QueryLoader::loadQuery( String( filename ), inputQuery );
}

// Code necessary to generate Python library
// Describes which classes and functions are exposed to API
PYBIND11_MODULE( MarabouCore, m )
{
    m.doc() = "Maraboupy bindings to the C++ Marabou via pybind11";
    py::class_<MarabouOptions>( m, "Options" )
        .def( py::init() )
        .def_readwrite( "_numWorkers", &MarabouOptions::_numWorkers )
        .def_readwrite( "_numBlasThreads", &MarabouOptions::_numBlasThreads )
        .def_readwrite( "_initialTimeout", &MarabouOptions::_initialTimeout )
        .def_readwrite( "_initialDivides", &MarabouOptions::_initialDivides )
        .def_readwrite( "_onlineDivides", &MarabouOptions::_onlineDivides )
        .def_readwrite( "_timeoutInSeconds", &MarabouOptions::_timeoutInSeconds )
        .def_readwrite( "_timeoutFactor", &MarabouOptions::_timeoutFactor )
        .def_readwrite( "_preprocessorBoundTolerance",
                        &MarabouOptions::_preprocessorBoundTolerance )
        .def_readwrite( "_milpSolverTimeout", &MarabouOptions::_milpSolverTimeout )
        .def_readwrite( "_verbosity", &MarabouOptions::_verbosity )
        .def_readwrite( "_splitThreshold", &MarabouOptions::_splitThreshold )
        .def_readwrite( "_snc", &MarabouOptions::_snc )
        .def_readwrite( "_solveWithMILP", &MarabouOptions::_solveWithMILP )
        .def_readwrite( "_dumpBounds", &MarabouOptions::_dumpBounds )
        .def_readwrite( "_restoreTreeStates", &MarabouOptions::_restoreTreeStates )
        .def_readwrite( "_splittingStrategy", &MarabouOptions::_splittingStrategyString )
        .def_readwrite( "_sncSplittingStrategy", &MarabouOptions::_sncSplittingStrategyString )
        .def_readwrite( "_tighteningStrategy", &MarabouOptions::_tighteningStrategyString )
        .def_readwrite( "_milpTightening", &MarabouOptions::_milpTighteningString )
        .def_readwrite( "_lpSolver", &MarabouOptions::_lpSolverString )
        .def_readwrite( "_numSimulations", &MarabouOptions::_numSimulations )
        .def_readwrite( "_performLpTighteningAfterSplit",
                        &MarabouOptions::_performLpTighteningAfterSplit )
        .def_readwrite( "_produceProofs", &MarabouOptions::_produceProofs )
        .def_readwrite( "_coreMinimizationTimeout", &MarabouOptions::_coreMinimizationTimeout )
        .def_readwrite( "_coreMinimizationLimit", &MarabouOptions::_coreMinimizationLimit )
        .def_readwrite( "_coreMinimizationSingleSolveTimeout",
                        &MarabouOptions::_coreMinimizationSingleSolveTimeout );
    m.def( "maraboupyMain", &maraboupyMain, "Run the Marabou command-line interface" );
    m.def( "loadProperty", &loadProperty, "Load a property file into a input query" );
    m.def( "createInputQuery",
           &createInputQuery,
           "Create input query from network and property file" );
    m.def( "solve",
           &solve,
           R"pbdoc(
        Takes in a description of the InputQuery and returns the solution

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            options (class:`~maraboupy.MarabouCore.Options`): Object defining the options used for Marabou
            redirect (str, optional): Filepath to direct standard output, defaults to ""
            phaseFixTrie (:class:`~maraboupy.MarabouCore.PhaseFixTrie`, optional): PhaseFixTrie from previous solve for incremental solving

        Returns:
            (tuple): tuple containing:
                - exitCode (str): A string representing the exit code (sat/unsat/TIMEOUT/ERROR/UNKNOWN/QUIT_REQUESTED).
                - vals (Dict[int, float]): Empty dictionary if UNSAT, otherwise a dictionary of SATisfying values for variables
                - stats (:class:`~maraboupy.MarabouCore.Statistics`): A Statistics object to how Marabou performed
                - phaseFixTrie (:class:`~maraboupy.MarabouCore.PhaseFixTrie`): Updated PhaseFixTrie to pass to next solve (Python owns this pointer)
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "options" ),
           py::arg( "redirect" ) = "",
           py::arg( "phaseFixTrie" ) = nullptr,
           py::return_value_policy::take_ownership );
    m.def( "calculateBounds",
           &calculateBounds,
           R"pbdoc(
        Takes in a description of the InputQuery and returns the bounds

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query which bounds are calculated
            options (class:`~maraboupy.MarabouCore.Options`): Object defining the options used for Marabou
            redirect (str, optional): Filepath to direct standard output, defaults to ""

        Returns:
            (tuple): tuple containing:
                - exitCode (str): A string representing the exit code. Only unsat can be returned
                - vals (Dict[int, tuple]): Empty dictionary if UNSAT, otherwise a dictionary of bounds for variables
                - stats (:class:`~maraboupy.MarabouCore.Statistics`): A Statistics object to how Marabou performed
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "options" ),
           py::arg( "redirect" ) = "" );
    m.def( "saveQuery",
           &saveQuery,
           R"pbdoc(
        Serializes the inputQuery in the given filename

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be saved
            filename (str): Name of file to save query
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "filename" ) );
    m.def( "saveQueryAsSmtLib",
           &saveQueryAsSmtLib,
           R"pbdoc(
        Serializes the inputQuery in the given filename as an SMTLIB file

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be saved
            filename (str): Name of file to save query
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "filename" ) );
    m.def( "loadQuery",
           &loadQuery,
           R"pbdoc(
        Loads and returns a serialized InputQuery from the given filename

        Args:
            filename (str): Name of file to load into an InputQuery
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): the input query to store the constraints
        )pbdoc",
           py::arg( "filename" ),
           py::arg( "inputQuery" ) );
    m.def( "addClipConstraint",
           &addClipConstraint,
           R"pbdoc(
        Add a Clip constraint to the InputQuery

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            var1 (int): Input variable to Clip constraint
            var2 (int): Output variable to Clip constraint
            lb (double): Floor
            ub (double): Ceiling
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "var1" ),
           py::arg( "var2" ),
           py::arg( "floor" ),
           py::arg( "ceiling" ) );
    m.def( "addLeakyReluConstraint",
           &addLeakyReluConstraint,
           R"pbdoc(
        Add a LeakyRelu constraint to the InputQuery

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            var1 (int): Input variable to Leaky ReLU constraint
            var2 (int): Output variable to Leaky ReLU constraint
            slope (float): Slope of the Leaky ReLU constraint
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "var1" ),
           py::arg( "var2" ),
           py::arg( "slope" ) );
    m.def( "addReluConstraint",
           &addReluConstraint,
           R"pbdoc(
        Add a Relu constraint to the InputQuery

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            var1 (int): Input variable to Relu constraint
            var2 (int): Output variable to Relu constraint
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "var1" ),
           py::arg( "var2" ) );
    m.def( "addRoundConstraint",
           &addRoundConstraint,
           R"pbdoc(
        Add a Round constraint to the InputQuery

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            var1 (int): Input variable to round constraint
            var2 (int): Output variable to round constraint
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "var1" ),
           py::arg( "var2" ) );
    m.def( "addBilinearConstraint",
           &addBilinearConstraint,
           R"pbdoc(
        Add a Bilinear constraint to the InputQuery
        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            var1 (int): Input variable to Bilinear constraint
            var2 (int): Input variable to Bilinear constraint
            var3 (int): Output variable to Bilinear constraint
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "var1" ),
           py::arg( "var2" ),
           py::arg( "var3" ) );
    m.def( "addSigmoidConstraint",
           &addSigmoidConstraint,
           R"pbdoc(
        Add a Sigmoid constraint to the InputQuery

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            var1 (int): Input variable to Sigmoid constraint
            var2 (int): Output variable to Sigmoid constraint
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "var1" ),
           py::arg( "var2" ) );
    m.def( "addSignConstraint",
           &addSignConstraint,
           R"pbdoc(
        Add a Sign constraint to the InputQuery

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            var1 (int): Input variable to Sign constraint
            var2 (int): Output variable to Sign constraint
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "var1" ),
           py::arg( "var2" ) );
    m.def( "addMaxConstraint",
           &addMaxConstraint,
           R"pbdoc(
        Add a Max constraint to the InputQuery

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            elements (set of int): Input variables to max constraint
            v (int): Output variable from max constraint
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "elements" ),
           py::arg( "v" ) );
    m.def( "addSoftmaxConstraint",
           &addSoftmaxConstraint,
           R"pbdoc(
        Add a Softmax constraint to the InputQuery
        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            inputs (list of int): Input variables to softmax constraint
            outputs (list of int): Output variables from softmax constraint
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "inputs" ),
           py::arg( "outputs" ) );
    m.def( "addAbsConstraint",
           &addAbsConstraint,
           R"pbdoc(
        Add an Abs constraint to the InputQuery

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            b (int): Input variable
            f (int): Output variable
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "b" ),
           py::arg( "f" ) );
    m.def( "addDisjunctionConstraint",
           &addDisjunctionConstraint,
           R"pbdoc(
        Add a disjunction constraint to the InputQuery

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query to be solved
            disjuncts (list of pairs): A list of disjuncts. Each disjunct is represented by a pair: a list of bounds, and a list of (in)equalities.
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "disjuncts" ) );
    py::class_<InputQuery>( m, "InputQuery" )
        .def( py::init() )
        .def( "setUpperBound", &InputQuery::setUpperBound )
        .def( "setLowerBound", &InputQuery::setLowerBound )
        .def( "getUpperBound", &InputQuery::getUpperBound )
        .def( "getLowerBound", &InputQuery::getLowerBound )
        .def( "tightenUpperBound", &InputQuery::tightenUpperBound )
        .def( "tightenLowerBound", &InputQuery::tightenLowerBound )
        .def( "pop", &InputQuery::pop )
        .def( "push", &InputQuery::push )
        .def( "getLevel", &InputQuery::getLevel )
        .def( "popTo", &InputQuery::popTo )
        .def( "dump", &InputQuery::dump )
        .def( "setNumberOfVariables", &InputQuery::setNumberOfVariables )
        .def( "addEquation", &InputQuery::addEquation )
        .def( "getSolutionValue", &InputQuery::getSolutionValue )
        .def( "getNumberOfVariables", &InputQuery::getNumberOfVariables )
        .def( "getNumInputVariables", &InputQuery::getNumInputVariables )
        .def( "getNumOutputVariables", &InputQuery::getNumOutputVariables )
        .def( "inputVariableByIndex", &InputQuery::inputVariableByIndex )
        .def( "markInputVariable", &InputQuery::markInputVariable )
        .def( "markOutputVariable", &InputQuery::markOutputVariable )
        .def( "outputVariableByIndex", &InputQuery::outputVariableByIndex );
    py::enum_<PiecewiseLinearFunctionType>( m, "PiecewiseLinearFunctionType" )
        .value( "ReLU", PiecewiseLinearFunctionType::RELU )
        .value( "AbsoluteValue", PiecewiseLinearFunctionType::ABSOLUTE_VALUE )
        .value( "Max", PiecewiseLinearFunctionType::MAX )
        .value( "Disjunction", PiecewiseLinearFunctionType::DISJUNCTION )
        .export_values();
    py::class_<Equation> eq( m, "Equation" );
    py::enum_<Equation::EquationType>( eq, "EquationType" )
        .value( "EQ", Equation::EquationType::EQ )
        .value( "GE", Equation::EquationType::GE )
        .value( "LE", Equation::EquationType::LE )
        .export_values();
    eq.def( py::init() );
    eq.def( py::init<Equation::EquationType>() );
    eq.def( "addAddend", &Equation::addAddend );
    eq.def( "setScalar", &Equation::setScalar );
    py::enum_<Statistics::StatisticsUnsignedAttribute>( m, "StatisticsUnsignedAttribute" )
        .value( "NUM_POPS", Statistics::StatisticsUnsignedAttribute::NUM_POPS )
        .value( "CURRENT_DECISION_LEVEL",
                Statistics::StatisticsUnsignedAttribute::CURRENT_DECISION_LEVEL )
        .value( "NUM_PL_SEARCH_TREE_ORIGINATED_SPLITS",
                Statistics::StatisticsUnsignedAttribute::NUM_PL_SEARCH_TREE_ORIGINATED_SPLITS )
        .value( "NUM_VISITED_TREE_STATES",
                Statistics::StatisticsUnsignedAttribute::NUM_VISITED_TREE_STATES )
        .value( "PP_NUM_TIGHTENING_ITERATIONS",
                Statistics::StatisticsUnsignedAttribute::PP_NUM_TIGHTENING_ITERATIONS )
        .value( "NUM_PL_VALID_SPLITS",
                Statistics::StatisticsUnsignedAttribute::NUM_PL_VALID_SPLITS )
        .value( "PP_NUM_ELIMINATED_VARS",
                Statistics::StatisticsUnsignedAttribute::PP_NUM_ELIMINATED_VARS )
        .value( "PP_NUM_EQUATIONS_REMOVED",
                Statistics::StatisticsUnsignedAttribute::PP_NUM_EQUATIONS_REMOVED )
        .value( "NUM_PL_CONSTRAINTS", Statistics::StatisticsUnsignedAttribute::NUM_PL_CONSTRAINTS )
        .value( "CURRENT_TABLEAU_M", Statistics::StatisticsUnsignedAttribute::CURRENT_TABLEAU_M )
        .value( "NUM_SPLITS", Statistics::StatisticsUnsignedAttribute::NUM_SPLITS )
        .value( "TOTAL_NUMBER_OF_VALID_CASE_SPLITS",
                Statistics::StatisticsUnsignedAttribute::TOTAL_NUMBER_OF_VALID_CASE_SPLITS )
        .value( "NUM_PRECISION_RESTORATIONS",
                Statistics::StatisticsUnsignedAttribute::NUM_PRECISION_RESTORATIONS )
        .value( "PP_NUM_CONSTRAINTS_REMOVED",
                Statistics::StatisticsUnsignedAttribute::PP_NUM_CONSTRAINTS_REMOVED )
        .value( "CURRENT_TABLEAU_N", Statistics::StatisticsUnsignedAttribute::CURRENT_TABLEAU_N )
        .value( "MAX_DECISION_LEVEL", Statistics::StatisticsUnsignedAttribute::MAX_DECISION_LEVEL )
        .value( "NUM_ACTIVE_PL_CONSTRAINTS",
                Statistics::StatisticsUnsignedAttribute::NUM_ACTIVE_PL_CONSTRAINTS )
        .export_values();
    py::enum_<Statistics::StatisticsLongAttribute>( m, "StatisticsLongAttribute" )
        .value( "NUM_TIGHTENINGS_FROM_EXPLICIT_BASIS",
                Statistics::StatisticsLongAttribute::NUM_TIGHTENINGS_FROM_EXPLICIT_BASIS )
        .value( "TOTAL_TIME_SEARCH_TREE_HANDLER_MICRO",
                Statistics::StatisticsLongAttribute::TOTAL_TIME_SEARCH_TREE_HANDLER_MICRO )
        .value( "TOTAL_TIME_PERFORMING_VALID_CASE_SPLITS_MICRO",
                Statistics::StatisticsLongAttribute::TOTAL_TIME_PERFORMING_VALID_CASE_SPLITS_MICRO )
        .value( "PREPROCESSING_TIME_MICRO",
                Statistics::StatisticsLongAttribute::PREPROCESSING_TIME_MICRO )
        .value( "NUM_SIMPLEX_UNSTABLE_PIVOTS",
                Statistics::StatisticsLongAttribute::NUM_SIMPLEX_UNSTABLE_PIVOTS )
        .value( "NUM_BOUND_TIGHTENINGS_ON_EXPLICIT_BASIS",
                Statistics::StatisticsLongAttribute::NUM_BOUND_TIGHTENINGS_ON_EXPLICIT_BASIS )
        .value( "NUM_BOUNDS_PROPOSED_BY_PL_CONSTRAINTS",
                Statistics::StatisticsLongAttribute::NUM_BOUNDS_PROPOSED_BY_PL_CONSTRAINTS )
        .value( "TIME_SIMPLEX_STEPS_MICRO",
                Statistics::StatisticsLongAttribute::TIME_SIMPLEX_STEPS_MICRO )
        .value( "TIME_PIVOTS_MICRO", Statistics::StatisticsLongAttribute::TIME_PIVOTS_MICRO )
        .value( "NUM_TIGHTENED_BOUNDS", Statistics::StatisticsLongAttribute::NUM_TIGHTENED_BOUNDS )
        .value( "PSE_NUM_ITERATIONS", Statistics::StatisticsLongAttribute::PSE_NUM_ITERATIONS )
        .value( "NUM_TABLEAU_PIVOTS", Statistics::StatisticsLongAttribute::NUM_TABLEAU_PIVOTS )
        .value( "NUM_TIGHTENINGS_FROM_CONSTRAINT_MATRIX",
                Statistics::StatisticsLongAttribute::NUM_TIGHTENINGS_FROM_CONSTRAINT_MATRIX )
        .value( "NUM_TABLEAU_DEGENERATE_PIVOTS_BY_REQUEST",
                Statistics::StatisticsLongAttribute::NUM_TABLEAU_DEGENERATE_PIVOTS_BY_REQUEST )
        .value( "NUM_ADDED_ROWS", Statistics::StatisticsLongAttribute::NUM_ADDED_ROWS )
        .value( "PSE_NUM_RESET_REFERENCE_SPACE",
                Statistics::StatisticsLongAttribute::PSE_NUM_RESET_REFERENCE_SPACE )
        .value( "TIME_MAIN_LOOP_MICRO", Statistics::StatisticsLongAttribute::TIME_MAIN_LOOP_MICRO )
        .value( "NUM_MERGED_COLUMNS", Statistics::StatisticsLongAttribute::NUM_MERGED_COLUMNS )
        .value( "NUM_BOUND_NOTIFICATIONS_TO_TRANSCENDENTAL_CONSTRAINTS",
                Statistics::StatisticsLongAttribute::
                    NUM_BOUND_NOTIFICATIONS_TO_TRANSCENDENTAL_CONSTRAINTS )
        .value( "NUM_SIMPLEX_PIVOT_SELECTIONS_IGNORED_FOR_STABILITY",
                Statistics::StatisticsLongAttribute::
                    NUM_SIMPLEX_PIVOT_SELECTIONS_IGNORED_FOR_STABILITY )
        .value( "TOTAL_TIME_CONSTRAINT_MATRIX_BOUND_TIGHTENING_MICRO",
                Statistics::StatisticsLongAttribute::
                    TOTAL_TIME_CONSTRAINT_MATRIX_BOUND_TIGHTENING_MICRO )
        .value( "NUM_TIGHTENINGS_FROM_ROWS",
                Statistics::StatisticsLongAttribute::NUM_TIGHTENINGS_FROM_ROWS )
        .value( "NUM_CONSTRAINT_FIXING_STEPS",
                Statistics::StatisticsLongAttribute::NUM_CONSTRAINT_FIXING_STEPS )
        .value(
            "TOTAL_TIME_PERFORMING_SYMBOLIC_BOUND_TIGHTENING",
            Statistics::StatisticsLongAttribute::TOTAL_TIME_PERFORMING_SYMBOLIC_BOUND_TIGHTENING )
        .value(
            "NUM_TIGHTENINGS_FROM_SYMBOLIC_BOUND_TIGHTENING",
            Statistics::StatisticsLongAttribute::NUM_TIGHTENINGS_FROM_SYMBOLIC_BOUND_TIGHTENING )
        .value( "NUM_ROWS_EXAMINED_BY_ROW_TIGHTENER",
                Statistics::StatisticsLongAttribute::NUM_ROWS_EXAMINED_BY_ROW_TIGHTENER )
        .value(
            "TOTAL_TIME_EXPLICIT_BASIS_BOUND_TIGHTENING_MICRO",
            Statistics::StatisticsLongAttribute::TOTAL_TIME_EXPLICIT_BASIS_BOUND_TIGHTENING_MICRO )
        .value( "NUM_BASIS_REFACTORIZATIONS",
                Statistics::StatisticsLongAttribute::NUM_BASIS_REFACTORIZATIONS )
        .value( "NUM_SIMPLEX_STEPS", Statistics::StatisticsLongAttribute::NUM_SIMPLEX_STEPS )
        .value( "NUM_BOUND_NOTIFICATIONS_TO_PL_CONSTRAINTS",
                Statistics::StatisticsLongAttribute::NUM_BOUND_NOTIFICATIONS_TO_PL_CONSTRAINTS )
        .value( "NUM_MAIN_LOOP_ITERATIONS",
                Statistics::StatisticsLongAttribute::NUM_MAIN_LOOP_ITERATIONS )
        .value( "NUM_TABLEAU_DEGENERATE_PIVOTS",
                Statistics::StatisticsLongAttribute::NUM_TABLEAU_DEGENERATE_PIVOTS )
        .value( "TOTAL_TIME_APPLYING_STORED_TIGHTENINGS_MICRO",
                Statistics::StatisticsLongAttribute::TOTAL_TIME_APPLYING_STORED_TIGHTENINGS_MICRO )
        .value( "TOTAL_TIME_HANDLING_STATISTICS_MICRO",
                Statistics::StatisticsLongAttribute::TOTAL_TIME_HANDLING_STATISTICS_MICRO )
        .value( "TOTAL_TIME_PRECISION_RESTORATION",
                Statistics::StatisticsLongAttribute::TOTAL_TIME_PRECISION_RESTORATION )
        .value( "NUM_BOUND_TIGHTENINGS_ON_CONSTRAINT_MATRIX",
                Statistics::StatisticsLongAttribute::NUM_BOUND_TIGHTENINGS_ON_CONSTRAINT_MATRIX )
        .value( "NUM_TABLEAU_BOUND_HOPPING",
                Statistics::StatisticsLongAttribute::NUM_TABLEAU_BOUND_HOPPING )
        .value( "TOTAL_TIME_DEGRADATION_CHECKING",
                Statistics::StatisticsLongAttribute::TOTAL_TIME_DEGRADATION_CHECKING )
        .value( "TIME_CONSTRAINT_FIXING_STEPS_MICRO",
                Statistics::StatisticsLongAttribute::TIME_CONSTRAINT_FIXING_STEPS_MICRO )
        .value( "TOTAL_TIME_UPDATING_SOI_PHASE_PATTERN_MICRO",
                Statistics::StatisticsLongAttribute::TOTAL_TIME_UPDATING_SOI_PHASE_PATTERN_MICRO )
        .value( "NUM_PROPOSED_PHASE_PATTERN_UPDATE",
                Statistics::StatisticsLongAttribute::NUM_PROPOSED_PHASE_PATTERN_UPDATE )
        .value( "NUM_ACCEPTED_PHASE_PATTERN_UPDATE",
                Statistics::StatisticsLongAttribute::NUM_ACCEPTED_PHASE_PATTERN_UPDATE )
        .value( "TOTAL_TIME_OBTAIN_CURRENT_ASSIGNMENT_MICRO",
                Statistics::StatisticsLongAttribute::TOTAL_TIME_OBTAIN_CURRENT_ASSIGNMENT_MICRO )
        .export_values();
    py::enum_<Statistics::StatisticsDoubleAttribute>( m, "StatisticsDoubleAttribute" )
        .value( "MAX_DEGRADATION", Statistics::StatisticsDoubleAttribute::MAX_DEGRADATION )
        .value( "CURRENT_DEGRADATION", Statistics::StatisticsDoubleAttribute::CURRENT_DEGRADATION )
        .export_values();
    py::class_<Statistics>( m, "Statistics" )
        .def( "getUnsignedAttribute", &Statistics::getUnsignedAttribute )
        .def( "getLongAttribute", &Statistics::getLongAttribute )
        .def( "getDoubleAttribute", &Statistics::getDoubleAttribute )
        .def( "getTotalTimeInMicro", &Statistics::getTotalTimeInMicro )
        .def( "hasTimedOut", &Statistics::hasTimedOut );
    py::class_<PhaseFixTrie>( m, "PhaseFixTrie" ).def( py::init<>() );
    m.def( "minimizeUnsatCores",
           &minimizeUnsatCores,
           R"pbdoc(
        Minimizes the UNSAT cores stored in the PhaseFixTrie using QuickXPlain.
        Heuristically minimizes a random subset (up to 3) of the available cores.

        Args:
            inputQuery (:class:`~maraboupy.MarabouCore.InputQuery`): Marabou input query (background constraints)
            options (class:`~maraboupy.MarabouCore.Options`): Object defining the options used for Marabou
            phaseFixTrie (:class:`~maraboupy.MarabouCore.PhaseFixTrie`): PhaseFixTrie containing UNSAT cores to minimize
        )pbdoc",
           py::arg( "inputQuery" ),
           py::arg( "options" ),
           py::arg( "phaseFixTrie" ) );
}
