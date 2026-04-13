/*********************                                                        */
/*! \file CuOptWrapper.h
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

#ifndef __CuOptWrapper_h__
#define __CuOptWrapper_h__

#ifdef ENABLE_CUOPT

#include "LPSolver.h"
#include "List.h"
#include "MString.h"
#include "Map.h"
#include "Vector.h"

#include <cuopt/linear_programming/cuopt_c.h>

class CuOptWrapper : public LPSolver
{
public:
    CuOptWrapper();
    ~CuOptWrapper();

    // Add a new variable to the model
    void addVariable( String name, double lb, double ub, VariableType type = CONTINUOUS ) override;

    // Set the lower or upper bound for an existing variable
    void setLowerBound( String name, double lb ) override;
    void setUpperBound( String name, double ub ) override;

    inline double getLowerBound( const String &name ) override
    {
        return _variableInfo[name]._lb;
    }

    inline double getUpperBound( const String &name ) override
    {
        return _variableInfo[name]._ub;
    }

    // Add a new LEQ constraint, e.g. 3x + 4y <= -5
    void addLeqConstraint( const List<Term> &terms, double scalar ) override;

    // Add a new GEQ constraint, e.g. 3x + 4y >= -5
    void addGeqConstraint( const List<Term> &terms, double scalar ) override;

    // Add a new EQ constraint, e.g. 3x + 4y = -5
    void addEqConstraint( const List<Term> &terms, double scalar ) override;

    // Unsupported: throws CommonError
    void addPiecewiseLinearConstraint( String sourceVariable,
                                       String targetVariable,
                                       unsigned numPoints,
                                       const double *xPoints,
                                       const double *yPoints ) override;

    // Unsupported: throws CommonError
    void addLeqIndicatorConstraint( const String binVarName,
                                    const int binVal,
                                    const List<Term> &terms,
                                    double scalar ) override;

    // Unsupported: throws CommonError
    void addGeqIndicatorConstraint( const String binVarName,
                                    const int binVal,
                                    const List<Term> &terms,
                                    double scalar ) override;

    // Unsupported: throws CommonError
    void addEqIndicatorConstraint( const String binVarName,
                                   const int binVal,
                                   const List<Term> &terms,
                                   double scalar ) override;

    // Unsupported: throws CommonError
    void
    addBilinearConstraint( const String input1, const String input2, const String output ) override;

    // A cost function to minimize, or an objective function to maximize
    void setCost( const List<Term> &terms, double constant = 0 ) override;
    void setObjective( const List<Term> &terms, double constant = 0 ) override;

    inline double getOptimalCostOrObjective() override
    {
        return _lastObjectiveValue;
    }

    // No-op for cuOpt
    void setCutoff( double cutoff ) override;

    // Returns true iff an optimal solution has been found
    bool optimal() override;

    // No-op: always returns false
    bool cutoffOccurred() override;

    // Returns true iff the instance is infeasible
    bool infeasible() override;

    // Returns true iff the instance timed out
    bool timeout() override;

    // Returns true iff a feasible solution has been found
    bool haveFeasibleSolution() override;

    // Specify a time limit, in seconds
    void setTimeLimit( double seconds ) override;

    // No-op for cuOpt
    inline void setVerbosity( unsigned verbosity ) override
    {
        _verbosity = verbosity;
    }

    inline bool containsVariable( String name ) const override
    {
        return _variableInfo.exists( name );
    }

    // No-op for cuOpt
    inline void setNumberOfThreads( unsigned threads ) override
    {
        _numThreads = threads;
    }

    // No-op for cuOpt
    inline void nonConvex() override
    {
    }

    // Solve and extract the solution, or the best known bound on the
    // objective function
    void solve() override;
    void extractSolution( Map<String, double> &values, double &costOrObjective ) override;
    double getObjectiveBound() override;

    inline double getAssignment( const String &variable ) override
    {
        return _lastSolutionValues[_variableInfo[variable]._index];
    }

    inline bool existsAssignment( const String &variable ) override
    {
        return _variableInfo.exists( variable ) && _hasSolution;
    }

    inline unsigned getNumberOfSimplexIterations() override
    {
        return 0;
    }

    inline unsigned getNumberOfNodes() override
    {
        return 0;
    }

    inline unsigned getStatusCode() override
    {
        return _terminationStatus;
    }

    inline void updateModel() override
    {
        // No-op: cuOpt builds model on solve()
    }

    // Reset the underlying model
    void reset() override;

    // Clear the underlying model and create a fresh model
    void resetModel() override;

    // No-op for cuOpt
    void dumpModel( String name ) override;

private:
    struct VariableInfo
    {
        unsigned _index;
        double _lb;
        double _ub;
        char _type;
    };

    struct Constraint
    {
        Vector<std::pair<unsigned, double>> _terms; // (varIndex, coefficient)
        double _lb;
        double _ub;
    };

    // Buffered problem state
    Map<String, VariableInfo> _variableInfo;
    Vector<String> _variableNames;
    Vector<Constraint> _constraints;
    Vector<double> _objectiveCoefficients;
    double _objectiveConstant;
    cuopt_int_t _objectiveSense;

    // Solver settings
    double _timeoutInSeconds;
    unsigned _verbosity;
    unsigned _numThreads;

    // Solution state
    cuOptSolution _solution;
    cuopt_int_t _terminationStatus;
    double _lastObjectiveValue;
    Vector<double> _lastSolutionValues;
    bool _hasSolution;

    void destroySolutionIfNeeded();

    static void log( const String &message );
};

#else

#include "LPSolver.h"
#include "MString.h"
#include "Map.h"

class CuOptWrapper : public LPSolver
{
public:
    /*
      This is a DUMMY class, for compilation purposes when cuOpt is
      disabled.
    */
    CuOptWrapper()
    {
    }
    ~CuOptWrapper()
    {
    }

    void addVariable( String, double, double, VariableType type = CONTINUOUS ) override
    {
        (void)type;
    }
    void setLowerBound( String, double ) override{};
    void setUpperBound( String, double ) override{};
    double getLowerBound( const String & ) override
    {
        return 0;
    };
    double getUpperBound( const String & ) override
    {
        return 0;
    };
    void addLeqConstraint( const List<Term> &, double ) override
    {
    }
    void addGeqConstraint( const List<Term> &, double ) override
    {
    }
    void addEqConstraint( const List<Term> &, double ) override
    {
    }
    void addPiecewiseLinearConstraint( String,
                                       String,
                                       unsigned,
                                       const double *,
                                       const double * ) override
    {
    }
    void addLeqIndicatorConstraint( const String, const int, const List<Term> &, double ) override
    {
    }
    void addGeqIndicatorConstraint( const String, const int, const List<Term> &, double ) override
    {
    }
    void addEqIndicatorConstraint( const String, const int, const List<Term> &, double ) override
    {
    }
    void addBilinearConstraint( const String, const String, const String ) override
    {
    }
    void setCost( const List<Term> &, double /* constant */ = 0 ) override
    {
    }
    void setObjective( const List<Term> &, double /* constant */ = 0 ) override
    {
    }
    double getOptimalCostOrObjective() override
    {
        return 0;
    };
    void setCutoff( double ) override{};
    void solve() override
    {
    }
    void extractSolution( Map<String, double> &, double & ) override
    {
    }
    void reset() override
    {
    }
    void resetModel() override
    {
    }
    bool optimal() override
    {
        return true;
    }
    bool cutoffOccurred() override
    {
        return false;
    };
    bool infeasible() override
    {
        return false;
    };
    bool timeout() override
    {
        return false;
    };
    bool haveFeasibleSolution() override
    {
        return true;
    };
    void setTimeLimit( double ) override{};
    void setVerbosity( unsigned ) override{};
    bool containsVariable( String /*name*/ ) const override
    {
        return false;
    };
    void setNumberOfThreads( unsigned ) override{};
    void nonConvex() override{};
    double getObjectiveBound() override
    {
        return 0;
    };
    double getAssignment( const String & ) override
    {
        return 0;
    };
    unsigned getNumberOfSimplexIterations() override
    {
        return 0;
    };
    unsigned getNumberOfNodes() override
    {
        return 0;
    };
    unsigned getStatusCode() override
    {
        return 0;
    };
    void updateModel() override{};
    bool existsAssignment( const String & ) override
    {
        return false;
    };

    void dumpModel( String ) override
    {
    }
    static void log( const String & );
};

#endif // ENABLE_CUOPT

#endif // __CuOptWrapper_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
