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

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <raft/core/handle.hpp>
#include <vector>

using CppProblem = cuopt::linear_programming::optimization_problem_t<int, double>;
using CppSettings = cuopt::linear_programming::pdlp_solver_settings_t<int, double>;
using CppSolution = cuopt::linear_programming::optimization_problem_solution_t<int, double>;

class CuOptWrapper : public LPSolver
{
public:
    CuOptWrapper();
    ~CuOptWrapper();

    void addVariable( String name, double lb, double ub, VariableType type = CONTINUOUS ) override;
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

    void addLeqConstraint( const List<Term> &terms, double scalar ) override;
    void addGeqConstraint( const List<Term> &terms, double scalar ) override;
    void addEqConstraint( const List<Term> &terms, double scalar ) override;

    void addPiecewiseLinearConstraint( String sourceVariable,
                                       String targetVariable,
                                       unsigned numPoints,
                                       const double *xPoints,
                                       const double *yPoints ) override;

    void addLeqIndicatorConstraint( const String binVarName,
                                    const int binVal,
                                    const List<Term> &terms,
                                    double scalar ) override;

    void addGeqIndicatorConstraint( const String binVarName,
                                    const int binVal,
                                    const List<Term> &terms,
                                    double scalar ) override;

    void addEqIndicatorConstraint( const String binVarName,
                                   const int binVal,
                                   const List<Term> &terms,
                                   double scalar ) override;

    void
    addBilinearConstraint( const String input1, const String input2, const String output ) override;

    void setCost( const List<Term> &terms, double constant = 0 ) override;
    void setObjective( const List<Term> &terms, double constant = 0 ) override;

    inline double getOptimalCostOrObjective() override
    {
        return _lastObjectiveValue;
    }

    void setCutoff( double cutoff ) override;
    bool optimal() override;
    bool cutoffOccurred() override;
    bool infeasible() override;
    bool timeout() override;
    bool haveFeasibleSolution() override;
    void setTimeLimit( double seconds ) override;

    inline void setVerbosity( unsigned verbosity ) override
    {
        _verbosity = verbosity;
    }

    inline bool containsVariable( String name ) const override
    {
        return _variableInfo.exists( name );
    }

    inline void setNumberOfThreads( unsigned threads ) override
    {
        _numThreads = threads;
    }

    inline void nonConvex() override
    {
    }

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
        _hasSolution = false;
    }

    void reset() override;
    void resetModel() override;
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
        Vector<std::pair<unsigned, double>> _terms;
        double _lb;
        double _ub;
    };

    // Buffered problem state
    Map<String, VariableInfo> _variableInfo;
    Vector<String> _variableNames;
    Vector<Constraint> _constraints;
    Vector<double> _objectiveCoefficients;
    double _objectiveConstant;
    int _objectiveSense;

    // Solver settings
    double _timeoutInSeconds;
    unsigned _verbosity;
    unsigned _numThreads;

    // Solution state
    int _terminationStatus;
    double _lastObjectiveValue;
    double _lastDualObjectiveValue;
    Vector<double> _lastSolutionValues;
    bool _hasSolution;

    // C++ API persistent state
    raft::handle_t *_raftHandle;
    CppProblem *_cppProblem;
    CppSolution *_lastCppSolution;
    bool _problemInitialized;

    // Cached CSR (constraint matrix unchanged between solves)
    std::vector<int> _cachedRowOffsets;
    std::vector<int> _cachedColIndices;
    std::vector<double> _cachedValues;
    std::vector<double> _cachedConstraintLB;
    std::vector<double> _cachedConstraintUB;

    static void log( const String &message );
};

#else

#include "LPSolver.h"
#include "MString.h"
#include "Map.h"

class CuOptWrapper : public LPSolver
{
public:
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
