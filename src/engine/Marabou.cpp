/*********************                                                        */
/*! \file Marabou.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Andrew Wu
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

#include "Marabou.h"

#include "AcasParser.h"
#include "AutoFile.h"
#include "File.h"
#include "GlobalConfiguration.h"
#include "MStringf.h"
#include "MarabouError.h"
#include "OnnxParser.h"
#include "Options.h"
#include "PropertyParser.h"
#include "QueryLoader.h"
#include "VnnLibParser.h"

#include <list>
#include <memory>

#ifdef BUILD_CADICAL
#include "CdclCore.h"
#endif

#ifdef _WIN32
#undef ERROR
#endif

Marabou::Marabou()
    : _acasParser( NULL )
    , _onnxParser( NULL )
    , _cegarSolver( NULL )
    , _engine( std::unique_ptr<Engine>( new Engine() ) )
{
}

Marabou::~Marabou()
{
    if ( _acasParser )
    {
        delete _acasParser;
        _acasParser = NULL;
    }

    if ( _onnxParser )
    {
        delete _onnxParser;
        _onnxParser = NULL;
    }
}

void Marabou::run()
{
    struct timespec start = TimeUtils::sampleMicro();

    prepareQuery();
    solveQuery();

    struct timespec end = TimeUtils::sampleMicro();

    unsigned long long totalElapsed = TimeUtils::timePassed( start, end );
    displayResults( totalElapsed );

    if ( Options::get()->getBool( Options::EXPORT_ASSIGNMENT ) )
        exportAssignment();
}

void Marabou::prepareQuery()
{
    String inputQueryFilePath = Options::get()->getString( Options::INPUT_QUERY_FILE_PATH );
    if ( inputQueryFilePath.length() > 0 )
    {
        /*
          Step 1: extract the query
        */
        if ( !File::exists( inputQueryFilePath ) )
        {
            printf( "Error: the specified inputQuery file (%s) doesn't exist!\n",
                    inputQueryFilePath.ascii() );
            throw MarabouError( MarabouError::FILE_DOESNT_EXIST, inputQueryFilePath.ascii() );
        }

        printf( "Query: %s\n", inputQueryFilePath.ascii() );
        QueryLoader::loadQuery( inputQueryFilePath, _inputQuery );
    }
    else
    {
        /*
          Step 1: extract the network
        */
        String networkFilePath = Options::get()->getString( Options::INPUT_FILE_PATH );

        if ( networkFilePath.length() == 0 )
        {
            printf( "Error: no network file provided!\n" );
            throw MarabouError( MarabouError::FILE_DOESNT_EXIST, networkFilePath.ascii() );
        }

        if ( !File::exists( networkFilePath ) )
        {
            printf( "Error: the specified network file (%s) doesn't exist!\n",
                    networkFilePath.ascii() );
            throw MarabouError( MarabouError::FILE_DOESNT_EXIST, networkFilePath.ascii() );
        }
        printf( "Network: %s\n", networkFilePath.ascii() );

        if ( ( (String)networkFilePath ).endsWith( ".onnx" ) )
        {
            InputQueryBuilder queryBuilder;
            OnnxParser::parse( queryBuilder, networkFilePath, {}, {} );
            queryBuilder.generateQuery( _inputQuery );
        }
        else
        {
            _acasParser = new AcasParser( networkFilePath );
            _acasParser->generateQuery( _inputQuery );
        }

        /*
          Step 2: extract the property in question
        */
        String propertyFilePath = Options::get()->getString( Options::PROPERTY_FILE_PATH );
        if ( propertyFilePath != "" )
        {
            printf( "Property: %s\n", propertyFilePath.ascii() );
            if ( propertyFilePath.endsWith( ".vnnlib" ) )
            {
                VnnLibParser().parse( propertyFilePath, _inputQuery );
            }
            else
            {
                PropertyParser().parse( propertyFilePath, _inputQuery );
            }
        }
        else
            printf( "Property: None\n" );

        printf( "\n" );
    }

    if ( Options::get()->getBool( Options::DEBUG_ASSIGNMENT ) )
        importDebuggingSolution();

    String queryDumpFilePath = Options::get()->getString( Options::QUERY_DUMP_FILE );
    if ( queryDumpFilePath.length() > 0 )
    {
        _inputQuery.saveQuery( queryDumpFilePath );
        printf( "\nInput query successfully dumped to file\n" );
        exit( 0 );
    }
}

void Marabou::importDebuggingSolution()
{
    String fileName = Options::get()->getString( Options::IMPORT_ASSIGNMENT_FILE_PATH );
    AutoFile input( fileName );

    if ( !IFile::exists( fileName ) )
    {
        throw MarabouError( MarabouError::FILE_DOES_NOT_EXIST,
                            Stringf( "File %s not found.\n", fileName.ascii() ).ascii() );
    }

    input->open( IFile::MODE_READ );

    unsigned numVars = atoi( input->readLine().trim().ascii() );
    ASSERT( numVars == _inputQuery.getNumberOfVariables() );

    unsigned var;
    double value;
    String line;

    // Import each assignment
    for ( unsigned i = 0; i < numVars; ++i )
    {
        line = input->readLine();
        List<String> tokens = line.tokenize( "," );
        auto it = tokens.begin();
        var = atoi( it->ascii() );
        ASSERT( var == i );
        it++;
        value = atof( it->ascii() );
        it++;
        ASSERT( it == tokens.end() );
        _inputQuery.storeDebuggingSolution( var, value );
    }

    input->close();
}

void Marabou::exportAssignment() const
{
    String assignmentFileName = "assignment.txt";
    AutoFile exportFile( assignmentFileName );
    exportFile->open( IFile::MODE_WRITE_TRUNCATE );

    unsigned numberOfVariables = _inputQuery.getNumberOfVariables();
    // Number of Variables
    exportFile->write( Stringf( "%u\n", numberOfVariables ) );

    // Export each assignment
    for ( unsigned var = 0; var < numberOfVariables; ++var )
        exportFile->write( Stringf( "%u, %f\n", var, _inputQuery.getSolutionValue( var ) ) );

    exportFile->close();
}

void Marabou::solveQuery()
{
    enum {
        MICROSECONDS_IN_SECOND = 1000000
    };

    struct timespec start = TimeUtils::sampleMicro();
    unsigned timeoutInSeconds = Options::get()->getInt( Options::TIMEOUT );
    if ( _engine->processInputQuery( _inputQuery ) )
    {
        if ( _engine->shouldSolveWithMILP() )
            _engine->solveWithMILPEncoding( timeoutInSeconds );
#ifdef BUILD_CADICAL
        else if ( _engine->shouldSolveWithCDCL() )
        {
            if ( Options::get()->getBool( Options::PR_CLAUSE_PREPROCESS ) )
                solveWithPrRebuild( timeoutInSeconds );
            else
            {
                _engine->solveWithCDCL( timeoutInSeconds );
                if ( _engine->shouldProduceProofs() &&
                     _engine->getExitCode() == ExitCode::UNSAT )
                    _engine->certifyUNSATCertificate();
            }
        }
#endif
        else
        {
            _engine->solve( timeoutInSeconds );
            if ( _engine->shouldProduceProofs() && _engine->getExitCode() == ExitCode::UNSAT )
                _engine->certifyUNSATCertificate();
        }
    }

    if ( _engine->getExitCode() == ExitCode::UNKNOWN )
    {
        struct timespec end = TimeUtils::sampleMicro();
        unsigned long long totalElapsed = TimeUtils::timePassed( start, end );
        if ( timeoutInSeconds == 0 || totalElapsed < timeoutInSeconds * MICROSECONDS_IN_SECOND )
        {
            _cegarSolver = new CEGAR::IncrementalLinearization( _inputQuery, _engine.release() );
            unsigned long long timeoutInMicroSeconds =
                ( timeoutInSeconds == 0
                      ? 0
                      : timeoutInSeconds * MICROSECONDS_IN_SECOND - totalElapsed );
            _cegarSolver->setInitialTimeoutInMicroSeconds( timeoutInMicroSeconds );
            _cegarSolver->solve();
            _engine = std::unique_ptr<Engine>( _cegarSolver->releaseEngine() );
        }
    }

    // TODO: update the variable assignment using NLR if possible and double-check that all the
    // constraints are indeed satisfied.
    bool prDriverRan = false;
#ifdef BUILD_CADICAL
    // The PR driver solves on separate phase engines and extracts the
    // solution itself; _engine never solved, so extracting from it would
    // overwrite the real counterexample.
    prDriverRan =
        _engine->shouldSolveWithCDCL() && Options::get()->getBool( Options::PR_CLAUSE_PREPROCESS );
#endif
    if ( _engine->getExitCode() == ExitCode::SAT && !prDriverRan )
        _engine->extractSolution( _inputQuery );
}

#ifdef BUILD_CADICAL
void Marabou::solveWithPrRebuild( unsigned timeoutInSeconds )
{
    struct timespec start = TimeUtils::sampleMicro();

    // Full isolation per engine: serialize the query once, reload a fresh
    // InputQuery per engine, and keep every (query, engine) pair alive until
    // the driver exits — constraints register engine-context pointers, so
    // destroying an engine while its query outlives it (or vice versa)
    // leaves dangling registrations for the next processInputQuery.
    String queryFile = "/tmp/pr_rebuild_query.ipq";
    _inputQuery.saveQuery( queryFile );
    std::list<std::unique_ptr<InputQuery>> keepAliveQueries;
    std::list<std::unique_ptr<Engine>> keepAliveEngines;
    auto freshEngineOnFreshQuery = [&]() -> Engine * {
        keepAliveQueries.push_back( std::unique_ptr<InputQuery>( new InputQuery() ) );
        QueryLoader::loadQuery( queryFile, *keepAliveQueries.back() );
        keepAliveEngines.push_back( std::unique_ptr<Engine>( new Engine() ) );
        Engine *engine = keepAliveEngines.back().get();
        if ( !engine->processInputQuery( *keepAliveQueries.back() ) )
            return NULL; // preprocessing already decided; exit code is set
        return engine;
    };

    auto remaining = [&]() -> double {
        double elapsed =
            TimeUtils::timePassed( start, TimeUtils::sampleMicro() ) / 1e6;
        return timeoutInSeconds == 0 ? 1e9 : (double)timeoutInSeconds - elapsed;
    };
    auto finish = [&]( ExitCode code, const char *msg ) {
        printf( "PR: %s\n", msg );
        fflush( stdout );
        CdclCore::prRebuildRole = CdclCore::PR_REBUILD_OFF;
        CdclCore::prSeedClauses.clear();
        // On SAT the counterexample lives in the phase engine, not _engine:
        // copy it into the query displayResults reads from.
        if ( code == ExitCode::SAT && !keepAliveEngines.empty() )
            keepAliveEngines.back()->extractSolution( _inputQuery );
        // Intentionally leak the phase engines and their queries: constraints
        // register engine-context pointers across each pair and no destruction
        // order is safe once a pair has been through processInputQuery
        // (destructor chain segfaults after the verdict). The process exits
        // right after displayResults.
        for ( auto &engine : keepAliveEngines )
            engine.release();
        for ( auto &query : keepAliveQueries )
            query.release();
        _engine->setExitCode( code );
    };

    // ---- Phase A: harvest on a virgin engine ----
    CdclCore::prRebuildRole = CdclCore::PR_REBUILD_HARVEST;
    CdclCore::prSeedClauses.clear();
    CdclCore::prHandoffValid = false;

    Engine *engineA = freshEngineOnFreshQuery();
    if ( engineA )
        engineA->solveWithCDCL( remaining() );

    ExitCode exitA = keepAliveEngines.back()->getExitCode();
    if ( exitA == ExitCode::SAT )
        return finish( ExitCode::SAT,
                       "phase A found a theory-checked counterexample - sat" );
    if ( exitA == ExitCode::UNSAT )
        return finish( ExitCode::UNSAT, "phase A concluded on its own - unsat" );
    if ( !CdclCore::prHandoffValid )
        return finish( ExitCode::TIMEOUT,
                       "phase A neither concluded nor harvested within budget" );

    // ---- Phase B: full solve on a virgin engine, ALL harvested clauses
    // injected (raw variant: the UNSAT verdict is not certified) ----
    CdclCore::prRebuildRole = CdclCore::PR_REBUILD_SOLVE;
    CdclCore::prSeedClauses = CdclCore::prHandoffCarry;
    for ( const Set<int> &clause : CdclCore::prHandoffSelected )
        CdclCore::prSeedClauses.append( clause );

    Engine *engineB = freshEngineOnFreshQuery();
    if ( engineB )
        engineB->solveWithCDCL( remaining() );

    ExitCode exitB = keepAliveEngines.back()->getExitCode();
    if ( exitB == ExitCode::SAT )
        return finish( ExitCode::SAT,
                       "phase B found a theory-checked counterexample - sat" );
    if ( exitB == ExitCode::UNSAT )
        return finish( ExitCode::UNSAT, "phase B unsat (PR clauses injected - uncertified)" );
    finish( ExitCode::TIMEOUT, "phase B inconclusive" );
}
#endif

void Marabou::displayResults( unsigned long long microSecondsElapsed ) const
{
    ExitCode result = _engine->getExitCode();
    String resultString;

    if ( result == ExitCode::UNSAT )
    {
        resultString = "unsat";
        printf( "unsat\n" );
    }
    else if ( result == ExitCode::SAT )
    {
        resultString = "sat";
        printf( "sat\n" );

        printf( "Input assignment:\n" );
        for ( unsigned i = 0; i < _inputQuery.getNumInputVariables(); ++i )
            printf( "\tx%u = %lf\n",
                    i,
                    _inputQuery.getSolutionValue( _inputQuery.inputVariableByIndex( i ) ) );

        printf( "\n" );
        printf( "Output:\n" );
        for ( unsigned i = 0; i < _inputQuery.getNumOutputVariables(); ++i )
            printf( "\ty%u = %lf\n",
                    i,
                    _inputQuery.getSolutionValue( _inputQuery.outputVariableByIndex( i ) ) );
        printf( "\n" );
    }
    else if ( result == ExitCode::TIMEOUT )
    {
        resultString = "TIMEOUT";
        printf( "Timeout\n" );
    }
    else if ( result == ExitCode::ERROR )
    {
        resultString = "ERROR";
        printf( "Error\n" );
    }
    else if ( result == ExitCode::UNKNOWN )
    {
        resultString = "UNKNOWN";
        printf( "UNKNOWN\n" );
    }
    else
    {
        resultString = "NOT_DONE";
        printf( "Unexpected exit code! (this should not happen)" );
    }

    // Create a summary file, if requested
    String summaryFilePath = Options::get()->getString( Options::SUMMARY_FILE );
    if ( summaryFilePath != "" )
    {
        File summaryFile( summaryFilePath );
        summaryFile.open( File::MODE_WRITE_TRUNCATE );

        // Field #1: result
        summaryFile.write( resultString );

        // Field #2: total elapsed time
        summaryFile.write( Stringf( " %u ", microSecondsElapsed / 1000000 ) ); // In seconds

        // Field #3: number of visited tree states
        summaryFile.write( Stringf( "%u ",
                                    _engine->getStatistics()->getUnsignedAttribute(
                                        Statistics::NUM_VISITED_TREE_STATES ) ) );

        // Field #4: average pivot time in micro seconds
        summaryFile.write(
            Stringf( "%u", _engine->getStatistics()->getAveragePivotTimeInMicro() ) );

        summaryFile.write( "\n" );
    }
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
