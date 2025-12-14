/**
** \verbatim
** Top contributors (to current version):
**   Omri Isac, Guy Katz
** This file is part of the Marabou project.
** Copyright (c) 2017-2025 by the authors listed in the file AUTHORS
** in the top-level source directory) and their institutional affiliations.
** All rights reserved. See the file COPYING in the top-level source
** directory for licensing information.\endverbatim
**
** [[ Add lengthier description here ]]
**/

#ifndef __AletheProofWriter_h__
#define __AletheProofWriter_h__

#include "GroundBoundManager.h"
#include "PiecewiseLinearCaseSplit.h"
#include "SmtLibWriter.h"
#include "SparseMatrix.h"
#include "SparseUnsortedList.h"
#include "Stack.h"
#include "UnsatCertificateNode.h"
#include "UnsatCertificateUtils.h"
#include "Vector.h"
#include "gmp.h"
#include "gmpxx.h"

class AletheProofWriter
{
public:
    AletheProofWriter( unsigned explanationSize,
                       const Vector<double> &upperBounds,
                       const Vector<double> &lowerBounds,
                       const GroundBoundManager &groundBoundManager,
                       const SparseMatrix *tableau,
                       const List<PiecewiseLinearConstraint *> &problemConstraints);


    void writeInstanceToFile( IFile &file );

    void writeChildrenConclusion( const UnsatCertificateNode *node );

    unsigned assignId();

    void writeDelegatedLeaf( const UnsatCertificateNode *node );

    void writeLemma( const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &lemmaEntry );

    void writeContradiction( const SparseUnsortedList &contradiction, unsigned nodeId );

private:
    const SparseMatrix *_initialTableau;
    Vector<String> _tableauAssumptions; // For easy access
    Vector<Stack<std::tuple<int, double>>> _currentUpperBounds;
    Vector<Stack<std::tuple<int, double>>> _currentLowerBounds;
    const GroundBoundManager &_groundBoundManager;
    Vector<PiecewiseLinearConstraint *> _plc;

    List<String> _proof;
    List<String> _assumptions;

    unsigned _n;
    unsigned _m;
    unsigned _stepCounter;

    Map<unsigned, PiecewiseLinearConstraint *> _varToPlc;
    Map<unsigned, List<Tightening>> _idToSplits;
    Map<unsigned, List<Tightening>> _nodeToSplits;

    void writeBoundAssumptions();

    void writePLCAssumption();

    void writeTableauAssumptions();

    void writeReluLemma( const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &lemmaEntry, const ReluConstraint *relu );

    void insertCurrentBoundsToVec( bool isUpper, Vector<double> &boundsVec );

    String getNegatedSplitsClause( const List<PiecewiseLinearCaseSplit> &splits ) const;

    String getSplitsResSteps( const List<PiecewiseLinearCaseSplit> &splits ) const;

    List<PiecewiseLinearCaseSplit> getPathSplits( const UnsatCertificateNode *node ) const;

    String getSplitsAsClause( const List<PiecewiseLinearCaseSplit> &splits ) const;

    String getSplitAsClause( const PiecewiseLinearCaseSplit &split ) const;

    String getBoundAsClause( const Tightening &bound ) const;

    String convertTableauAssumptionToClause( unsigned index ) const;

    bool isSplitActive( const PiecewiseLinearCaseSplit &split ) const;

    void linearCombinationMpq( const std::vector<mpq_t> &explainedRow,
                               const SparseUnsortedList &expl ) const;

    void farkasStrings( const SparseUnsortedList &expl,
                        unsigned entryId,
                        String &farkasArgs,
                        String &farkasClause,
                        String &farkasParticipants,
                        String &negatedSplitClause,
                        int explainerVar,
                        bool isUpper );
};

#endif // __AletheProofWriter_h__