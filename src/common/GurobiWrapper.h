/*********************                                                        */
/*! \file GurobiWrapper.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Teruhiro Tagomori
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

#ifndef __GurobiWrapper_h__
#define __GurobiWrapper_h__

#ifdef ENABLE_GUROBI

#include "LPSolver.h"
#include "MString.h"
#include "Map.h"
#include "gurobi_c++.h"

class GurobiWrapper : public LPSolver
{
public:
    GurobiWrapper();
    ~GurobiWrapper();

    // Add a new variable to the model
    void addVariable( String name, double lb, double ub, VariableType type = CONTINUOUS ) override;

    // Set the lower or upper bound for an existing variable
    void setLowerBound( String name, double lb ) override;
    void setUpperBound( String name, double ub ) override;

    inline double getLowerBound( const String &name ) override
    {
        return _model->getVarByName( name.ascii() ).get( GRB_DoubleAttr_LB );
    }

    inline double getUpperBound( const String &name ) override
    {
        return _model->getVarByName( name.ascii() ).get( GRB_DoubleAttr_UB );
    }

    // Add a new LEQ constraint, e.g. 3x + 4y <= -5
    void addLeqConstraint( const List<Term> &terms, double scalar ) override;

    // Add a new GEQ constraint, e.g. 3x + 4y >= -5
    void addGeqConstraint( const List<Term> &terms, double scalar ) override;

    // Add a new EQ constraint, e.g. 3x + 4y = -5
    void addEqConstraint( const List<Term> &terms, double scalar ) override;

    // Add a piece-wise linear constraint
    void addPiecewiseLinearConstraint( String sourceVariable,
                                       String targetVariable,
                                       unsigned numPoints,
                                       const double *xPoints,
                                       const double *yPoints ) override;

    // Add a new LEQ indicator constraint
    void addLeqIndicatorConstraint( const String binVarName,
                                    const int binVal,
                                    const List<Term> &terms,
                                    double scalar ) override;

    // Add a new GEQ indicator constraint
    void addGeqIndicatorConstraint( const String binVarName,
                                    const int binVal,
                                    const List<Term> &terms,
                                    double scalar ) override;

    // Add a new EQ indicator constraint
    void addEqIndicatorConstraint( const String binVarName,
                                   const int binVal,
                                   const List<Term> &terms,
                                   double scalar ) override;

    // Add a bilinear constraint
    void
    addBilinearConstraint( const String input1, const String input2, const String output ) override;

    // A cost function to minimize, or an objective function to maximize
    void setCost( const List<Term> &terms, double constant = 0 ) override;
    void setObjective( const List<Term> &terms, double constant = 0 ) override;

    inline double getOptimalCostOrObjective() override
    {
        return _model->get( GRB_DoubleAttr_ObjVal );
    }

    // Set a cutoff value for the objective function. For example, if
    // maximizing x with cutoff value 0, Gurobi will return the
    // optimal value if greater than 0, and 0 if the optimal value is
    // less than 0.
    void setCutoff( double cutoff ) override;

    // Returns true iff an optimal solution has been found
    bool optimal() override;

    // Returns true iff the cutoff value was used
    bool cutoffOccurred() override;

    // Returns true iff the instance is infeasible
    bool infeasible() override;

    // Returns true iff the instance timed out
    bool timeout() override;

    // Returns true iff a feasible solution has been found
    bool haveFeasibleSolution() override;

    // Specify a time limit, in seconds
    void setTimeLimit( double seconds ) override;

    // Set verbosity
    inline void setVerbosity( unsigned verbosity ) override
    {
        _model->getEnv().set( GRB_IntParam_OutputFlag, verbosity );
    }

    inline bool containsVariable( String name ) const override
    {
        return _nameToVariable.exists( name );
    }

    // Set number of threads
    inline void setNumberOfThreads( unsigned threads ) override
    {
        _model->getEnv().set( GRB_IntParam_Threads, threads );
    }

    inline void nonConvex() override
    {
        _model->getEnv().set( GRB_IntParam_NonConvex, 2 );
    }

    // Solve and extract the solution, or the best known bound on the
    // objective function
    void solve() override;
    void extractSolution( Map<String, double> &values, double &costOrObjective ) override;
    double getObjectiveBound() override;

    inline double getAssignment( const String &variable ) override
    {
        return _nameToVariable[variable]->get( GRB_DoubleAttr_X );
    }

    // Check if the assignment exists or not.
    inline bool existsAssignment( const String &variable ) override
    {
        return _nameToVariable.exists( variable ) && _model->get( GRB_IntAttr_SolCount ) > 0;
    }

    inline unsigned getNumberOfSimplexIterations() override
    {
        return _model->get( GRB_DoubleAttr_IterCount );
    }

    inline unsigned getNumberOfNodes() override
    {
        return _model->get( GRB_DoubleAttr_NodeCount );
    }

    inline unsigned getStatusCode() override
    {
        return _model->get( GRB_IntAttr_Status );
    }

    inline void updateModel() override
    {
        _model->update();
    }

    // Reset the underlying model
    void reset() override;

    // Clear the underlying model and create a fresh model
    void resetModel() override;

    // Dump the model to a file. Note that the suffix of the file is
    // used by Gurobi to determine the format. Using ".lp" is a good
    // default
    void dumpModel( String name ) override;

private:
    GRBEnv *_environment;
    GRBModel *_model;
    Map<String, GRBVar *> _nameToVariable;
    double _timeoutInSeconds;

    void addConstraint( const List<Term> &terms, double scalar, char sense );
    // Add a new indicator constraint
    void addIndicatorConstraint( const String binVarName,
                                 const int binVal,
                                 const List<Term> &terms,
                                 double scalar,
                                 char sense );

    void freeModelIfNeeded();
    void freeMemoryIfNeeded();

    static void log( const String &message );
};

#else

#include "LPSolver.h"
#include "MString.h"
#include "Map.h"

class GurobiWrapper : public LPSolver
{
public:
    /*
      This is a DUMMY class, for compilation purposes when Gurobi is
      disabled.
    */
    GurobiWrapper()
    {
    }
    ~GurobiWrapper()
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

#endif // ENABLE_GUROBI

#endif // __GurobiWrapper_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
