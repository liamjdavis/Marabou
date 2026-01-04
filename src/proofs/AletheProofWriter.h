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
#include "tracer.hpp"

class CdclCore;
class AletheProofWriter
#if BUILD_CADICAL
    : public CaDiCaL::Tracer
#endif
{
public:
    struct AletheStepEntry
    {
        AletheStepEntry( int64_t id,
                         std::vector<int> clause,
                         std::vector<int64_t> antecedents,
                         std::shared_ptr<GroundBoundManager::GroundBoundEntry> gbEntry,
                         SparseUnsortedList contradiction,
                         int propagatedLit )
            : id( id )
            , clause( clause )
            , antecedents( antecedents )
            , gbEntry( gbEntry )
            , contradiction( contradiction )
            , propagatedLit( propagatedLit )
        {
        }
        int64_t id;
        std::vector<int> clause;
        std::vector<int64_t> antecedents;
        std::shared_ptr<GroundBoundManager::GroundBoundEntry> gbEntry;
        SparseUnsortedList contradiction;
        int propagatedLit;
    };


    static const unsigned ALETHE_WRITER_PRECISION;

    AletheProofWriter( unsigned explanationSize,
                       const Vector<double> &upperBounds,
                       const Vector<double> &lowerBounds,
                       const GroundBoundManager &groundBoundManager,
                       const SparseMatrix *tableau,
                       const List<PiecewiseLinearConstraint *> &problemConstraints,
                       const String &queryId,
                       const String &proofFileName,
                       const String &proofDir,
                       const CdclCore *cdclCore );

    void writeInstanceToFile( IFile &file );

    void writeChildrenConclusion( const UnsatCertificateNode *node );

    unsigned assignId();

    void writeDelegatedLeaf( const UnsatCertificateNode *node );

    void writeLemma( const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &lemmaEntry );

    void writeContradiction( const SparseUnsortedList &contradiction, int64_t id );

    void flushAssumptions();

    void flushProof();

    void finalizeProof();

    void deleteProof();

    void deleteCombinedProof();

    String getFileName() const;

#if BUILD_CADICAL
    void writeDelegatedLeaf( int64_t id, const std::vector<int> &clause );
    void add_derived_clause( int64_t id,
                             bool redundant,
                             int witness,
                             const std::vector<int> &clause,
                             const std::vector<int64_t> &antecedents );

    void add_original_clause( int64_t id,
                              bool redundant,
                              const std::vector<int> &clause,
                              bool restored = false );

    void setLastContradiction( const SparseUnsortedList &contradiction );
    const Set<int> &getLastContradictionClause() const;

    void setLastContradictionClause( Set<int> &clause );

    void addDummyContradiction();

    void addEntryToStack( const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &entry );
    void writeLemmaResolution( const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &entry,
                               int propagatedLit,
                               int64_t id );
    bool hasInfo() const;
    bool lemmaExistsAsReasonClause( int64_t id ) const;
#endif

    void createCombinedProofFile();

    void removeProofDirectory() const;

private:
    const SparseMatrix *_initialTableau;
    Vector<String> _tableauAssumptions; // For easy access
    Vector<double> _baseUpperBounds;
    Vector<double> _baseLowerBounds;
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

    String _queryId;
    File _proofFile;
    String _proofFileName;
    String _proofDir;
    File _combinedProofFile;
    String _combinedProofFilename;
    List<AletheStepEntry> _proofEntries;

#if BUILD_CADICAL
    const CdclCore *_cdclCore;
    Vector<std::shared_ptr<GroundBoundManager::GroundBoundEntry>> _lastExplainedEntries;
    SparseUnsortedList _lastContradiction;
    Set<int> _lastContradictionClause;
    Map<int64_t, unsigned> _satIdToCdclVar;
    String clauseToPhases( const std::vector<int> &clause );
    void writeDerivedClauseContent( int64_t id,
                                    const std::vector<int> &clause,
                                    const std::vector<int64_t> &antecedents );

#endif

    void writeBoundAssumptions();

    void writePLCAssumption();

    void writeTableauAssumptions();

    bool writeReluLemma( const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &lemmaEntry,
                         const ReluConstraint *relu );

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
                        bool isUpper,
                        const Set<int> &deps );
};

#endif // __AletheProofWriter_h__