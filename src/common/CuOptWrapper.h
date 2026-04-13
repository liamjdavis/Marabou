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

#include "List.h"
#include "MString.h"
#include "Map.h"
#include "Vector.h"

#include <cuopt/linear_programming/cuopt_c.h>

class CuOptWrapper
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

    CuOptWrapper();
    ~CuOptWrapper();

    // Add a new variable to the model
    void addVariable( String name, double lb, double ub, VariableType type = CONTINUOUS );

    // Set the lower or upper bound for an existing variable
    void setLowerBound( String name, double lb );
    void setUpperBound( String name, double ub );

    inline double getLowerBound( const String &name )
    {
        return _variableInfo[name]._lb;
    }

    inline double getUpperBound( const String &name )
    {
        return _variableInfo[name]._ub;
    }

    // Add a new LEQ constraint, e.g. 3x + 4y <= -5
    void addLeqConstraint( const List<Term> &terms, double scalar );

    // Add a new GEQ constraint, e.g. 3x + 4y >= -5
    void addGeqConstraint( const List<Term> &terms, double scalar );

    // Add a new EQ constraint, e.g. 3x + 4y = -5
    void addEqConstraint( const List<Term> &terms, double scalar );

    // Unsupported: throws CommonError
    void addPiecewiseLinearConstraint( String sourceVariable,
                                       String targetVariable,
                                       unsigned numPoints,
                                       const double *xPoints,
                                       const double *yPoints );

    // Unsupported: throws CommonError
    void addLeqIndicatorConstraint( const String binVarName,
                                    const int binVal,
                                    const List<Term> &terms,
                                    double scalar );

    // Unsupported: throws CommonError
    void addGeqIndicatorConstraint( const String binVarName,
                                    const int binVal,
                                    const List<Term> &terms,
                                    double scalar );

    // Unsupported: throws CommonError
    void addEqIndicatorConstraint( const String binVarName,
                                   const int binVal,
                                   const List<Term> &terms,
                                   double scalar );

    // Unsupported: throws CommonError
    void addBilinearConstraint( const String input1, const String input2, const String output );

    // A cost function to minimize, or an objective function to maximize
    void setCost( const List<Term> &terms, double constant = 0 );
    void setObjective( const List<Term> &terms, double constant = 0 );

    inline double getOptimalCostOrObjective()
    {
        return _lastObjectiveValue;
    }

    // No-op for cuOpt
    void setCutoff( double cutoff );

    // Returns true iff an optimal solution has been found
    bool optimal();

    // No-op: always returns false
    bool cutoffOccurred();

    // Returns true iff the instance is infeasible
    bool infeasible();

    // Returns true iff the instance timed out
    bool timeout();

    // Returns true iff a feasible solution has been found
    bool haveFeasibleSolution();

    // Specify a time limit, in seconds
    void setTimeLimit( double seconds );

    // No-op for cuOpt
    inline void setVerbosity( unsigned verbosity )
    {
        _verbosity = verbosity;
    }

    inline bool containsVariable( String name ) const
    {
        return _variableInfo.exists( name );
    }

    // No-op for cuOpt
    inline void setNumberOfThreads( unsigned threads )
    {
        _numThreads = threads;
    }

    // No-op for cuOpt
    inline void nonConvex()
    {
    }

    // Solve and extract the solution, or the best known bound on the
    // objective function
    void solve();
    void extractSolution( Map<String, double> &values, double &costOrObjective );
    double getObjectiveBound();

    inline double getAssignment( const String &variable )
    {
        return _lastSolutionValues[_variableInfo[variable]._index];
    }

    inline bool existsAssignment( const String &variable )
    {
        return _variableInfo.exists( variable ) && _hasSolution;
    }

    inline unsigned getNumberOfSimplexIterations()
    {
        return 0;
    }

    inline unsigned getNumberOfNodes()
    {
        return 0;
    }

    inline unsigned getStatusCode()
    {
        return _terminationStatus;
    }

    inline void updateModel()
    {
        // No-op: cuOpt builds model on solve()
    }

    // Reset the underlying model
    void reset();

    // Clear the underlying model and create a fresh model
    void resetModel();

    // No-op for cuOpt
    void dumpModel( String name );

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

#include "MString.h"
#include "Map.h"

class CuOptWrapper
{
public:
    /*
      This is a DUMMY class, for compilation purposes when cuOpt is
      disabled.
    */
    enum VariableType {
        CONTINUOUS = 0,
        BINARY = 1,
        INTEGER = 2,
    };

    struct Term
    {
        Term( double, String )
        {
        }
        Term()
        {
        }
    };

    CuOptWrapper()
    {
    }
    ~CuOptWrapper()
    {
    }

    void addVariable( String, double, double, VariableType type = CONTINUOUS )
    {
        (void)type;
    }
    void setLowerBound( String, double ){};
    void setUpperBound( String, double ){};
    double getLowerBound( const String & )
    {
        return 0;
    };
    double getUpperBound( const String & )
    {
        return 0;
    };
    void addLeqConstraint( const List<Term> &, double )
    {
    }
    void addGeqConstraint( const List<Term> &, double )
    {
    }
    void addEqConstraint( const List<Term> &, double )
    {
    }
    void addPiecewiseLinearConstraint( String, String, unsigned, const double *, const double * )
    {
    }
    void addLeqIndicatorConstraint( const String, const int, const List<Term> &, double )
    {
    }
    void addGeqIndicatorConstraint( const String, const int, const List<Term> &, double )
    {
    }
    void addEqIndicatorConstraint( const String, const int, const List<Term> &, double )
    {
    }
    void addBilinearConstraint( const String, const String, const String )
    {
    }
    void setCost( const List<Term> &, double /* constant */ = 0 )
    {
    }
    void setObjective( const List<Term> &, double /* constant */ = 0 )
    {
    }
    double getOptimalCostOrObjective()
    {
        return 0;
    };
    void setCutoff( double ){};
    void solve()
    {
    }
    void extractSolution( Map<String, double> &, double & )
    {
    }
    void reset()
    {
    }
    void resetModel()
    {
    }
    bool optimal()
    {
        return true;
    }
    bool cutoffOccurred()
    {
        return false;
    };
    bool infeasible()
    {
        return false;
    };
    bool timeout()
    {
        return false;
    };
    bool haveFeasibleSolution()
    {
        return true;
    };
    void setTimeLimit( double ){};
    void setVerbosity( unsigned ){};
    bool containsVariable( String /*name*/ ) const
    {
        return false;
    };
    void setNumberOfThreads( unsigned ){};
    void nonConvex(){};
    double getObjectiveBound()
    {
        return 0;
    };
    double getAssignment( const String & )
    {
        return 0;
    };
    unsigned getNumberOfSimplexIterations()
    {
        return 0;
    };
    unsigned getNumberOfNodes()
    {
        return 0;
    };
    unsigned getStatusCode()
    {
        return 0;
    };
    void updateModel(){};
    bool existsAssignment( const String & )
    {
        return false;
    };

    void dump()
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
