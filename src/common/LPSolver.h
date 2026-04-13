/*********************                                                        */
/*! \file LPSolver.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Liam Davis
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** Abstract base class for LP/MILP solvers (GurobiWrapper, CuOptWrapper).

 **/

#ifndef __LPSolver_h__
#define __LPSolver_h__

#include "LPSolverType.h"
#include "List.h"
#include "MString.h"
#include "Map.h"

/*
  Factory function: creates the appropriate LPSolver subclass.
  Returns nullptr for NATIVE.
*/
class LPSolver;
LPSolver *createLPSolver( LPSolverType type );

class LPSolver
{
public:
    enum VariableType {
        CONTINUOUS = 0,
        BINARY = 1,
        INTEGER = 2,
    };

    /*
      A term has the form: coefficient * variable
    */
    struct Term
    {
        Term( double coefficient, String variable )
            : _coefficient( coefficient )
            , _variable( variable )
        {
        }

        Term()
            : _coefficient( 0 )
            , _variable( "" )
        {
        }

        double _coefficient;
        String _variable;
    };

    virtual ~LPSolver()
    {
    }

    // Add a new variable to the model
    virtual void
    addVariable( String name, double lb, double ub, VariableType type = CONTINUOUS ) = 0;

    // Set the lower or upper bound for an existing variable
    virtual void setLowerBound( String name, double lb ) = 0;
    virtual void setUpperBound( String name, double ub ) = 0;

    virtual double getLowerBound( const String &name ) = 0;
    virtual double getUpperBound( const String &name ) = 0;

    // Add a new LEQ constraint, e.g. 3x + 4y <= -5
    virtual void addLeqConstraint( const List<Term> &terms, double scalar ) = 0;

    // Add a new GEQ constraint, e.g. 3x + 4y >= -5
    virtual void addGeqConstraint( const List<Term> &terms, double scalar ) = 0;

    // Add a new EQ constraint, e.g. 3x + 4y = -5
    virtual void addEqConstraint( const List<Term> &terms, double scalar ) = 0;

    // Add a piece-wise linear constraint
    virtual void addPiecewiseLinearConstraint( String sourceVariable,
                                               String targetVariable,
                                               unsigned numPoints,
                                               const double *xPoints,
                                               const double *yPoints ) = 0;

    // Add indicator constraints
    virtual void addLeqIndicatorConstraint( const String binVarName,
                                            const int binVal,
                                            const List<Term> &terms,
                                            double scalar ) = 0;

    virtual void addGeqIndicatorConstraint( const String binVarName,
                                            const int binVal,
                                            const List<Term> &terms,
                                            double scalar ) = 0;

    virtual void addEqIndicatorConstraint( const String binVarName,
                                           const int binVal,
                                           const List<Term> &terms,
                                           double scalar ) = 0;

    // Add a bilinear constraint
    virtual void
    addBilinearConstraint( const String input1, const String input2, const String output ) = 0;

    // A cost function to minimize, or an objective function to maximize
    virtual void setCost( const List<Term> &terms, double constant = 0 ) = 0;
    virtual void setObjective( const List<Term> &terms, double constant = 0 ) = 0;

    virtual double getOptimalCostOrObjective() = 0;

    virtual void setCutoff( double cutoff ) = 0;

    // Returns true iff an optimal solution has been found
    virtual bool optimal() = 0;

    // Returns true iff the cutoff value was used
    virtual bool cutoffOccurred() = 0;

    // Returns true iff the instance is infeasible
    virtual bool infeasible() = 0;

    // Returns true iff the instance timed out
    virtual bool timeout() = 0;

    // Returns true iff a feasible solution has been found
    virtual bool haveFeasibleSolution() = 0;

    // Specify a time limit, in seconds
    virtual void setTimeLimit( double seconds ) = 0;

    // Set verbosity
    virtual void setVerbosity( unsigned verbosity ) = 0;

    virtual bool containsVariable( String name ) const = 0;

    // Set number of threads
    virtual void setNumberOfThreads( unsigned threads ) = 0;

    virtual void nonConvex() = 0;

    // Solve and extract the solution
    virtual void solve() = 0;
    virtual void extractSolution( Map<String, double> &values, double &costOrObjective ) = 0;
    virtual double getObjectiveBound() = 0;

    virtual double getAssignment( const String &variable ) = 0;

    virtual bool existsAssignment( const String &variable ) = 0;

    virtual unsigned getNumberOfSimplexIterations() = 0;

    virtual unsigned getNumberOfNodes() = 0;

    virtual unsigned getStatusCode() = 0;

    virtual void updateModel() = 0;

    // Reset the underlying model
    virtual void reset() = 0;

    // Clear the underlying model and create a fresh model
    virtual void resetModel() = 0;

    // Dump the model to a file
    virtual void dumpModel( String name ) = 0;
};

#endif // __LPSolver_h__
