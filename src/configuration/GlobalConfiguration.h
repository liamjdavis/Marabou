/*********************                                                        */
/*! \file FloatUtils.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2016-2017 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **/

#ifndef __GlobalConfiguration_h__
#define __GlobalConfiguration_h__

class GlobalConfiguration
{
public:
    static void print();

    // The default epsilon used for comparing doubles
    static const double DEFAULT_EPSILON_FOR_COMPARISONS;

    // The precision level when convering doubles to strings
    static const unsigned DEFAULT_DOUBLE_TO_STRING_PRECISION;

    // The number of accumualted eta matrices, after which the basis will be refactorized
	static const unsigned REFACTORIZATION_THRESHOLD;

    // How often should the main loop print statistics?
    static const unsigned STATISTICS_PRINTING_FREQUENCY;

    // Tolerance when checking whether the value computed for a basic variable is out of bounds
    static const double BOUND_COMPARISON_TOLERANCE;

    // Tolerance when checking whether a basic variable depends on a non-basic variable, by looking
    // at the change column, as part of a pivot operation.
    static const double PIVOT_CHANGE_COLUMN_TOLERANCE;

    // Toggle query-preprocessing on/off.
	static const bool PREPROCESS_INPUT_QUERY;

    // Assuming the preprocessor is on, toggle whether or not it will attempt to perform variable
    // elimination.
    static const bool PREPROCESSOR_ELIMINATE_VARIABLES;

    // Assuming the preprocessor is on, toggle whether or not PL constraints will be called upon
    // to add auxiliary variables and equations.
    static const bool PREPROCESSOR_PL_CONSTRAINTS_ADD_AUX_EQUATIONS;

    // How often should the main loop check the current degradation?
    static const unsigned DEGRADATION_CHECKING_FREQUENCY;

    // The threshold of degradation above which restoration is required
    static const double DEGRADATION_THRESHOLD;

    // If a pivot element in a simplex element is smaller than this threshold, the engine will attempt
    // to pick another element.
    static const double ACCEPTABLE_SIMPLEX_PIVOT_THRESHOLD;

    // How many potential pivots should the engine inspect (at most) in every simplex iteration?
    static const unsigned MAX_SIMPLEX_PIVOT_SEARCH_ITERATIONS;

    // The number of violations of a constraints after which the SMT core will initiate a case split
    static const unsigned CONSTRAINT_VIOLATION_THRESHOLD;

    // How often should we perform full bound tightening, on the entire contraints matrix A.
    static const unsigned BOUND_TIGHTING_ON_CONSTRAINT_MATRIX_FREQUENCY;

    // How often should projected steepest edge reset the reference space?
    static const unsigned PSE_ITERATIONS_BEFORE_RESET;

    // An error threshold which, when crossed, causes projected steepest edge to reset the reference space
    static const double PSE_GAMMA_ERROR_THRESHOLD;

    // When doing bound tightening using the explicit basis matrix, should the basis matrix be inverted?
    static const bool EXPLICIT_BASIS_BOUND_TIGHTENING_INVERT_BASIS;

    // Logging
    static const bool ENGINE_LOGGING;
    static const bool TABLEAU_LOGGING;
    static const bool SMT_CORE_LOGGING;
    static const bool DANTZIGS_RULE_LOGGING;
    static const bool BASIS_FACTORIZATION_LOGGING;
    static const bool PROJECTED_STEEPEST_EDGE_LOGGING;
};

#endif // __GlobalConfiguration_h__

//
// Local Variables:
// compile-command: "make -C .. "
// tags-file-name: "../TAGS"
// c-basic-offset: 4
// End:
//
