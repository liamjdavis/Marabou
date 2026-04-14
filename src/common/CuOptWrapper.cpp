/*********************                                                        */
/*! \file CuOptWrapper.cpp
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

#ifdef ENABLE_CUOPT

#include "CuOptWrapper.h"

#include "CommonError.h"
#include "Debug.h"
#include "GlobalConfiguration.h"
#include "MStringf.h"

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

static int g_solveCount = 0;
static double g_totalSolveMs = 0;
static double g_totalSetupMs = 0;
static double g_totalExtractMs = 0;

CuOptWrapper::CuOptWrapper()
    : _objectiveConstant( 0 )
    , _objectiveSense( CUOPT_MINIMIZE )
    , _timeoutInSeconds( 0 )
    , _verbosity( 0 )
    , _numThreads( 0 )
    , _terminationStatus( CUOPT_TERMINATION_STATUS_NO_TERMINATION )
    , _lastObjectiveValue( 0 )
    , _lastDualObjectiveValue( 0 )
    , _hasSolution( false )
    , _raftHandle( nullptr )
    , _cppProblem( nullptr )
    , _lastCppSolution( nullptr )
    , _problemInitialized( false )
{
}

CuOptWrapper::~CuOptWrapper()
{
    delete _lastCppSolution;
    delete _cppProblem;
    delete _raftHandle;
}

void CuOptWrapper::addVariable( String name, double lb, double ub, VariableType type )
{
    ASSERT( !_variableInfo.exists( name ) );

    unsigned index = _variableNames.size();

    VariableInfo info;
    info._index = index;
    info._lb = lb;
    info._ub = ub;

    switch ( type )
    {
    case CONTINUOUS:
        info._type = CUOPT_CONTINUOUS;
        break;
    case BINARY:
        info._type = CUOPT_INTEGER;
        break;
    case INTEGER:
        info._type = CUOPT_INTEGER;
        break;
    default:
        info._type = CUOPT_CONTINUOUS;
        break;
    }

    _variableInfo[name] = info;
    _variableNames.append( name );
    _objectiveCoefficients.append( 0.0 );
}

void CuOptWrapper::setLowerBound( String name, double lb )
{
    ASSERT( _variableInfo.exists( name ) );
    _variableInfo[name]._lb = lb;
}

void CuOptWrapper::setUpperBound( String name, double ub )
{
    ASSERT( _variableInfo.exists( name ) );
    _variableInfo[name]._ub = ub;
}

void CuOptWrapper::addLeqConstraint( const List<Term> &terms, double scalar )
{
    Constraint c;
    for ( const auto &term : terms )
    {
        ASSERT( _variableInfo.exists( term._variable ) );
        c._terms.append(
            std::make_pair( _variableInfo[term._variable]._index, term._coefficient ) );
    }
    c._lb = -CUOPT_INFINITY;
    c._ub = scalar;
    _constraints.append( c );
}

void CuOptWrapper::addGeqConstraint( const List<Term> &terms, double scalar )
{
    Constraint c;
    for ( const auto &term : terms )
    {
        ASSERT( _variableInfo.exists( term._variable ) );
        c._terms.append(
            std::make_pair( _variableInfo[term._variable]._index, term._coefficient ) );
    }
    c._lb = scalar;
    c._ub = CUOPT_INFINITY;
    _constraints.append( c );
}

void CuOptWrapper::addEqConstraint( const List<Term> &terms, double scalar )
{
    Constraint c;
    for ( const auto &term : terms )
    {
        ASSERT( _variableInfo.exists( term._variable ) );
        c._terms.append(
            std::make_pair( _variableInfo[term._variable]._index, term._coefficient ) );
    }
    c._lb = scalar;
    c._ub = scalar;
    _constraints.append( c );
}

void CuOptWrapper::addPiecewiseLinearConstraint( String,
                                                 String,
                                                 unsigned,
                                                 const double *,
                                                 const double * )
{
    throw CommonError( CommonError::CUOPT_EXCEPTION,
                       "CuOptWrapper does not support piecewise linear constraints" );
}

void CuOptWrapper::addLeqIndicatorConstraint( const String, const int, const List<Term> &, double )
{
    throw CommonError( CommonError::CUOPT_EXCEPTION,
                       "CuOptWrapper does not support indicator constraints" );
}

void CuOptWrapper::addGeqIndicatorConstraint( const String, const int, const List<Term> &, double )
{
    throw CommonError( CommonError::CUOPT_EXCEPTION,
                       "CuOptWrapper does not support indicator constraints" );
}

void CuOptWrapper::addEqIndicatorConstraint( const String, const int, const List<Term> &, double )
{
    throw CommonError( CommonError::CUOPT_EXCEPTION,
                       "CuOptWrapper does not support indicator constraints" );
}

void CuOptWrapper::addBilinearConstraint( const String, const String, const String )
{
    throw CommonError( CommonError::CUOPT_EXCEPTION,
                       "CuOptWrapper does not support bilinear constraints" );
}

void CuOptWrapper::setCost( const List<Term> &terms, double constant )
{
    _objectiveSense = CUOPT_MINIMIZE;

    for ( unsigned i = 0; i < _objectiveCoefficients.size(); ++i )
        _objectiveCoefficients[i] = 0.0;

    for ( const auto &term : terms )
    {
        ASSERT( _variableInfo.exists( term._variable ) );
        unsigned idx = _variableInfo[term._variable]._index;
        _objectiveCoefficients[idx] = term._coefficient;
    }

    _objectiveConstant = constant;
}

void CuOptWrapper::setObjective( const List<Term> &terms, double constant )
{
    _objectiveSense = CUOPT_MAXIMIZE;

    for ( unsigned i = 0; i < _objectiveCoefficients.size(); ++i )
        _objectiveCoefficients[i] = 0.0;

    for ( const auto &term : terms )
    {
        ASSERT( _variableInfo.exists( term._variable ) );
        unsigned idx = _variableInfo[term._variable]._index;
        _objectiveCoefficients[idx] = term._coefficient;
    }

    _objectiveConstant = constant;
}

void CuOptWrapper::setCutoff( double )
{
}

void CuOptWrapper::setTimeLimit( double seconds )
{
    _timeoutInSeconds = seconds;
}

void CuOptWrapper::solve()
{
    auto t0 = std::chrono::high_resolution_clock::now();

    _hasSolution = false;
    _terminationStatus = CUOPT_TERMINATION_STATUS_NO_TERMINATION;

    int numVariables = (int)_variableNames.size();
    int numConstraints = (int)_constraints.size();

    // Build variable bounds
    std::vector<double> varLB( numVariables );
    std::vector<double> varUB( numVariables );

    for ( const auto &entry : _variableInfo )
    {
        unsigned idx = entry.second._index;
        varLB[idx] = entry.second._lb;
        varUB[idx] = entry.second._ub;
    }

    // Build objective coefficients
    std::vector<double> objCoeffs( numVariables );
    for ( int i = 0; i < numVariables; ++i )
        objCoeffs[i] = _objectiveCoefficients[i];

    if ( !_problemInitialized )
    {
        auto tInit0 = std::chrono::high_resolution_clock::now();
        // First solve: create GPU handle and problem, build + cache CSR
        _raftHandle = new raft::handle_t();
        auto tRaft = std::chrono::high_resolution_clock::now();
        double raftMs = std::chrono::duration<double, std::milli>( tRaft - tInit0 ).count();
        fprintf( stderr, "[cuopt] raft::handle_t init: %.1f ms\n", raftMs );

        _cppProblem = new CppProblem( _raftHandle );

        _cachedRowOffsets.clear();
        _cachedColIndices.clear();
        _cachedValues.clear();
        _cachedConstraintLB.clear();
        _cachedConstraintUB.clear();

        _cachedRowOffsets.reserve( numConstraints + 1 );
        _cachedRowOffsets.push_back( 0 );

        for ( const auto &constraint : _constraints )
        {
            for ( const auto &term : constraint._terms )
            {
                _cachedColIndices.push_back( (int)term.first );
                _cachedValues.push_back( term.second );
            }
            _cachedRowOffsets.push_back( (int)_cachedColIndices.size() );
            _cachedConstraintLB.push_back( constraint._lb );
            _cachedConstraintUB.push_back( constraint._ub );
        }

        int nnz = (int)_cachedValues.size();

        _cppProblem->set_csr_constraint_matrix( _cachedValues.data(),
                                                nnz,
                                                _cachedColIndices.data(),
                                                nnz,
                                                _cachedRowOffsets.data(),
                                                numConstraints + 1 );

        _cppProblem->set_constraint_lower_bounds( _cachedConstraintLB.data(), numConstraints );
        _cppProblem->set_constraint_upper_bounds( _cachedConstraintUB.data(), numConstraints );

        // Set variable types
        std::vector<cuopt::linear_programming::var_t> varTypes( numVariables );
        for ( const auto &entry : _variableInfo )
        {
            unsigned idx = entry.second._index;
            varTypes[idx] = ( entry.second._type == CUOPT_INTEGER )
                              ? cuopt::linear_programming::var_t::INTEGER
                              : cuopt::linear_programming::var_t::CONTINUOUS;
        }
        _cppProblem->set_variable_types( varTypes.data(), numVariables );

        _problemInitialized = true;
        auto tInit1 = std::chrono::high_resolution_clock::now();
        double initMs = std::chrono::duration<double, std::milli>( tInit1 - tInit0 ).count();
        fprintf( stderr,
                 "[cuopt] First-solve init: %.1f ms (vars=%d, cons=%d, nnz=%d)\n",
                 initMs,
                 numVariables,
                 numConstraints,
                 nnz );
    }

    // Every solve: update variable bounds + objective
    _cppProblem->set_variable_lower_bounds( varLB.data(), numVariables );
    _cppProblem->set_variable_upper_bounds( varUB.data(), numVariables );
    _cppProblem->set_objective_coefficients( objCoeffs.data(), numVariables );
    _cppProblem->set_objective_offset( _objectiveConstant );
    _cppProblem->set_maximize( _objectiveSense == CUOPT_MAXIMIZE );

    auto tSetup = std::chrono::high_resolution_clock::now();
    double setupMs = std::chrono::duration<double, std::milli>( tSetup - t0 ).count();

    // DualSimplex only — wins Concurrent race anyway, uses far less GPU memory
    CppSettings settings;
    settings.method = cuopt::linear_programming::method_t::DualSimplex;
    settings.tolerances.absolute_primal_tolerance = 1e-6;
    settings.tolerances.relative_primal_tolerance = 1e-6;
    settings.tolerances.absolute_dual_tolerance = 1e-6;
    settings.tolerances.relative_dual_tolerance = 1e-6;
    settings.tolerances.absolute_gap_tolerance = 1e-6;
    settings.tolerances.relative_gap_tolerance = 1e-6;
    settings.detect_infeasibility = true;
    settings.log_to_console = ( _verbosity > 0 );

    // Solve
    auto tSolve0 = std::chrono::high_resolution_clock::now();
    auto solution =
        cuopt::linear_programming::solve_lp( *_cppProblem, settings, /*problem_checking=*/false );
    auto tSolve1 = std::chrono::high_resolution_clock::now();
    double solveMs = std::chrono::duration<double, std::milli>( tSolve1 - tSolve0 ).count();

    _lastCppSolution = new CppSolution( std::move( solution ) );

    auto status = _lastCppSolution->get_termination_status();
    _terminationStatus = static_cast<int>( status );

    // Extract solution if feasible
    if ( status == cuopt::linear_programming::pdlp_termination_status_t::Optimal ||
         status == cuopt::linear_programming::pdlp_termination_status_t::PrimalFeasible )
    {
        auto &primal = _lastCppSolution->get_primal_solution();

        _lastSolutionValues.clear();
        _lastSolutionValues = Vector<double>( numVariables, 0.0 );

        std::vector<double> hostVals( numVariables );
        cudaMemcpy( hostVals.data(),
                    primal.data(),
                    numVariables * sizeof( double ),
                    cudaMemcpyDeviceToHost );

        for ( int i = 0; i < numVariables; ++i )
            _lastSolutionValues[i] = hostVals[i];

        _lastObjectiveValue = _lastCppSolution->get_objective_value();
        _lastDualObjectiveValue = _lastCppSolution->get_dual_objective_value();
        _hasSolution = true;
    }

    // Extract stats before freeing GPU memory
    auto tEnd = std::chrono::high_resolution_clock::now();
    double extractMs = std::chrono::duration<double, std::milli>( tEnd - tSolve1 ).count();
    double totalMs = std::chrono::duration<double, std::milli>( tEnd - t0 ).count();

    g_solveCount++;
    g_totalSolveMs += solveMs;
    g_totalSetupMs += setupMs;
    g_totalExtractMs += extractMs;

    // Print every 100 solves + first 10
    if ( g_solveCount <= 10 || g_solveCount % 100 == 0 )
    {
        auto info = _lastCppSolution->get_additional_termination_information();
        const char *methodName = "?";
        switch ( static_cast<int>( info.solved_by ) )
        {
        case 0:
            methodName = "Concurrent";
            break;
        case 1:
            methodName = "PDLP";
            break;
        case 2:
            methodName = "DualSimplex";
            break;
        case 3:
            methodName = "Barrier";
            break;
        case 4:
            methodName = "Unset";
            break;
        }

        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo( &freeMem, &totalMem );

        fprintf( stderr,
                 "[cuopt] #%d: solve=%.1fms cuopt_t=%.3fs method=%s iters=%d "
                 "status=%d gap=%.2e gpu=%zuMB/%zuMB | cum: solve=%.0fms\n",
                 g_solveCount,
                 solveMs,
                 info.solve_time,
                 methodName,
                 info.number_of_steps_taken,
                 _terminationStatus,
                 info.gap,
                 freeMem / ( 1024 * 1024 ),
                 totalMem / ( 1024 * 1024 ),
                 g_totalSolveMs );
    }

    // Free GPU memory immediately — solution already extracted to host
    delete _lastCppSolution;
    _lastCppSolution = nullptr;
}

bool CuOptWrapper::optimal()
{
    return _terminationStatus == CUOPT_TERMINATION_STATUS_OPTIMAL;
}

bool CuOptWrapper::cutoffOccurred()
{
    return false;
}

bool CuOptWrapper::infeasible()
{
    return _terminationStatus == CUOPT_TERMINATION_STATUS_INFEASIBLE;
}

bool CuOptWrapper::timeout()
{
    return _terminationStatus == CUOPT_TERMINATION_STATUS_TIME_LIMIT;
}

bool CuOptWrapper::haveFeasibleSolution()
{
    return _hasSolution;
}

void CuOptWrapper::extractSolution( Map<String, double> &values, double &costOrObjective )
{
    values.clear();

    if ( _hasSolution )
    {
        for ( unsigned i = 0; i < _variableNames.size(); ++i )
            values[_variableNames[i]] = _lastSolutionValues[i];
    }

    costOrObjective = _lastObjectiveValue;
}

double CuOptWrapper::getObjectiveBound()
{
    if ( _hasSolution )
        return _lastDualObjectiveValue;

    if ( _objectiveSense == CUOPT_MINIMIZE )
        return -CUOPT_INFINITY;
    else
        return CUOPT_INFINITY;
}

void CuOptWrapper::reset()
{
    // Keep persistent problem and raft handle alive
    _terminationStatus = CUOPT_TERMINATION_STATUS_NO_TERMINATION;
    _lastObjectiveValue = 0;
    _lastSolutionValues.clear();
    _hasSolution = false;
}

void CuOptWrapper::resetModel()
{
    reset();
    delete _lastCppSolution;
    _lastCppSolution = nullptr;
    delete _cppProblem;
    _cppProblem = nullptr;
    delete _raftHandle;
    _raftHandle = nullptr;
    _problemInitialized = false;
    _variableInfo.clear();
    _variableNames.clear();
    _constraints.clear();
    _objectiveCoefficients.clear();
    _objectiveConstant = 0;
    _objectiveSense = CUOPT_MINIMIZE;
    _cachedRowOffsets.clear();
    _cachedColIndices.clear();
    _cachedValues.clear();
    _cachedConstraintLB.clear();
    _cachedConstraintUB.clear();
}

void CuOptWrapper::dumpModel( String )
{
}

void CuOptWrapper::log( const String &message )
{
    if ( GlobalConfiguration::GUROBI_LOGGING )
        printf( "CuOptWrapper: %s\n", message.ascii() );
}

#endif // ENABLE_CUOPT

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
