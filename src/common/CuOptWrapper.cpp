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

CuOptWrapper::CuOptWrapper()
    : _objectiveConstant( 0 )
    , _objectiveSense( CUOPT_MINIMIZE )
    , _timeoutInSeconds( 0 )
    , _verbosity( 0 )
    , _numThreads( 0 )
    , _solution( nullptr )
    , _terminationStatus( CUOPT_TERIMINATION_STATUS_NO_TERMINATION )
    , _lastObjectiveValue( 0 )
    , _hasSolution( false )
{
}

CuOptWrapper::~CuOptWrapper()
{
    destroySolutionIfNeeded();
}

void CuOptWrapper::destroySolutionIfNeeded()
{
    if ( _solution )
    {
        cuOptDestroySolution( &_solution );
        _solution = nullptr;
    }
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

    // Reset coefficients to zero
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

    // Reset coefficients to zero
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
    // No-op: cuOpt does not support cutoff
}

void CuOptWrapper::setTimeLimit( double seconds )
{
    _timeoutInSeconds = seconds;
}

void CuOptWrapper::solve()
{
    destroySolutionIfNeeded();
    _hasSolution = false;
    _terminationStatus = CUOPT_TERIMINATION_STATUS_NO_TERMINATION;

    cuopt_int_t numVariables = (cuopt_int_t)_variableNames.size();
    cuopt_int_t numConstraints = (cuopt_int_t)_constraints.size();

    // Build variable bounds and types arrays
    std::vector<cuopt_float_t> varLB( numVariables );
    std::vector<cuopt_float_t> varUB( numVariables );
    std::vector<char> varTypes( numVariables );

    for ( const auto &entry : _variableInfo )
    {
        unsigned idx = entry.second._index;
        varLB[idx] = entry.second._lb;
        varUB[idx] = entry.second._ub;
        varTypes[idx] = entry.second._type;
    }

    // Build objective coefficients
    std::vector<cuopt_float_t> objCoeffs( numVariables );
    for ( cuopt_int_t i = 0; i < numVariables; ++i )
        objCoeffs[i] = _objectiveCoefficients[i];

    // Build CSR constraint matrix
    std::vector<cuopt_int_t> rowOffsets;
    std::vector<cuopt_int_t> colIndices;
    std::vector<cuopt_float_t> values;
    std::vector<cuopt_float_t> constraintLB;
    std::vector<cuopt_float_t> constraintUB;

    rowOffsets.reserve( numConstraints + 1 );
    rowOffsets.push_back( 0 );

    for ( const auto &constraint : _constraints )
    {
        for ( const auto &term : constraint._terms )
        {
            colIndices.push_back( (cuopt_int_t)term.first );
            values.push_back( term.second );
        }
        rowOffsets.push_back( (cuopt_int_t)colIndices.size() );
        constraintLB.push_back( constraint._lb );
        constraintUB.push_back( constraint._ub );
    }

    // Create the problem
    cuOptOptimizationProblem problem = nullptr;
    cuopt_int_t status = cuOptCreateRangedProblem( numConstraints,
                                                   numVariables,
                                                   _objectiveSense,
                                                   (cuopt_float_t)_objectiveConstant,
                                                   objCoeffs.data(),
                                                   rowOffsets.data(),
                                                   colIndices.data(),
                                                   values.data(),
                                                   constraintLB.data(),
                                                   constraintUB.data(),
                                                   varLB.data(),
                                                   varUB.data(),
                                                   varTypes.data(),
                                                   &problem );

    if ( status != CUOPT_SUCCESS )
    {
        throw CommonError(
            CommonError::CUOPT_EXCEPTION,
            Stringf( "cuOptCreateRangedProblem failed with status %d", status ).ascii() );
    }

    // Create solver settings
    cuOptSolverSettings settings = nullptr;
    status = cuOptCreateSolverSettings( &settings );
    if ( status != CUOPT_SUCCESS )
    {
        cuOptDestroyProblem( &problem );
        throw CommonError(
            CommonError::CUOPT_EXCEPTION,
            Stringf( "cuOptCreateSolverSettings failed with status %d", status ).ascii() );
    }

    // Set time limit if specified
    if ( _timeoutInSeconds > 0 )
    {
        cuOptSetFloatParameter( settings, CUOPT_TIME_LIMIT, (cuopt_float_t)_timeoutInSeconds );
    }

    // Set verbosity
    if ( _verbosity == 0 )
    {
        cuOptSetIntegerParameter( settings, CUOPT_LOG_TO_CONSOLE, 0 );
    }
    else
    {
        cuOptSetIntegerParameter( settings, CUOPT_LOG_TO_CONSOLE, 1 );
    }

    // Solve
    status = cuOptSolve( problem, settings, &_solution );

    // Get termination status regardless of solve return code
    if ( _solution )
    {
        cuOptGetTerminationStatus( _solution, &_terminationStatus );

        // Extract solution values if we have a feasible solution
        if ( _terminationStatus == CUOPT_TERIMINATION_STATUS_OPTIMAL ||
             _terminationStatus == CUOPT_TERIMINATION_STATUS_PRIMAL_FEASIBLE ||
             _terminationStatus == CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND )
        {
            _lastSolutionValues.clear();
            _lastSolutionValues = Vector<double>( numVariables, 0.0 );
            std::vector<cuopt_float_t> solVals( numVariables );
            cuOptGetPrimalSolution( _solution, solVals.data() );
            for ( cuopt_int_t i = 0; i < numVariables; ++i )
                _lastSolutionValues[i] = solVals[i];

            cuopt_float_t objVal = 0;
            cuOptGetObjectiveValue( _solution, &objVal );
            _lastObjectiveValue = objVal;
            _hasSolution = true;
        }
    }

    // Clean up problem and settings (keep solution)
    cuOptDestroySolverSettings( &settings );
    cuOptDestroyProblem( &problem );

    if ( status != CUOPT_SUCCESS && !_solution )
    {
        throw CommonError( CommonError::CUOPT_EXCEPTION,
                           Stringf( "cuOptSolve failed with status %d", status ).ascii() );
    }
}

bool CuOptWrapper::optimal()
{
    return _terminationStatus == CUOPT_TERIMINATION_STATUS_OPTIMAL;
}

bool CuOptWrapper::cutoffOccurred()
{
    return false;
}

bool CuOptWrapper::infeasible()
{
    return _terminationStatus == CUOPT_TERIMINATION_STATUS_INFEASIBLE;
}

bool CuOptWrapper::timeout()
{
    return _terminationStatus == CUOPT_TERIMINATION_STATUS_TIME_LIMIT;
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
    if ( _solution )
    {
        cuopt_float_t bound = 0;
        cuopt_int_t status = cuOptGetSolutionBound( _solution, &bound );
        if ( status == CUOPT_SUCCESS )
            return bound;
    }

    if ( _objectiveSense == CUOPT_MINIMIZE )
        return -CUOPT_INFINITY;
    else
        return CUOPT_INFINITY;
}

void CuOptWrapper::reset()
{
    destroySolutionIfNeeded();
    _terminationStatus = CUOPT_TERIMINATION_STATUS_NO_TERMINATION;
    _lastObjectiveValue = 0;
    _lastSolutionValues.clear();
    _hasSolution = false;
}

void CuOptWrapper::resetModel()
{
    reset();
    _variableInfo.clear();
    _variableNames.clear();
    _constraints.clear();
    _objectiveCoefficients.clear();
    _objectiveConstant = 0;
    _objectiveSense = CUOPT_MINIMIZE;
}

void CuOptWrapper::dumpModel( String )
{
    // No-op: cuOpt does not support model dumping
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
