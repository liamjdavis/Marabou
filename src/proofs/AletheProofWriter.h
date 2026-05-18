/**
** \verbatim
** Top contributors (to current version):
**   Omri Isac, Guy Katz
** This file is part of the Marabou project.
** Copyright (c) 2017-2026 by the authors listed in the file AUTHORS
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

#include <mutex>
#include <utility>

class CdclCore;
class AletheProofWriter
#if BUILD_CADICAL
    : public CaDiCaL::Tracer
#endif
{
public:
    /*
     Helper struct to store resolution steps separately, and write them at the end to if necessary.
     */
    struct AletheStepEntry
    {
        AletheStepEntry( int64_t id,
                         std::vector<int> clause,
                         std::vector<int64_t> antecedents,
                         std::shared_ptr<GroundBoundManager::GroundBoundEntry> gbEntry,
                         const SparseUnsortedList &contradiction,
                         int propagatedLit )
            : id( id )
            , clause( std::move( clause ) )
            , antecedents( std::move( antecedents ) )
            , gbEntry( std::move( gbEntry ) )
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

    /*
     Class configuration for precision for writing proof bound elements.
     */
    static const unsigned ALETHE_WRITER_PRECISION;

    AletheProofWriter( unsigned explanationSize,
                       const Vector<double> &upperBounds,
                       const Vector<double> &lowerBounds,
                       const GroundBoundManager &groundBoundManager,
                       const SparseMatrix *tableau,
                       const List<PiecewiseLinearConstraint *> &problemConstraints,
                       const String &queryId,
                       const CdclCore *cdclCore );

    // -------  NON-CDCL SOLVING  -------

    /*
     Write whole proof info to a file
     */
    void writeInstanceToFile( IFile &file );

    /*
     Write steps to conclude UNSAT of a node from the UNSAT of its children
    */
    void writeChildrenConclusion( const UnsatCertificateNode *node );

    /*
     Get the next unique ID to a node, and increment it
    */
    unsigned assignId();

    /*
     Write proof hole for a delegated leaf node
    */
    void writeDelegatedLeaf( const UnsatCertificateNode *node );

    // -------  BOTH CDCL and NON-CDCL SOLVING  -------

    /*
     Sets the initial tableau constraints that define the query
     */
    void setInitialTableau( const SparseMatrix *tableau );

    /*
     Add proof steps to prove a PLC lemma
    */
    void writeLemma( const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &lemmaEntry );

    /*
     Add proof steps to prove a UNSAT of a leaf
    */
    void writeContradiction( const SparseUnsortedList &contradiction, int64_t id, UnsatCertificateNode *node );

    /*
     Create a proof file for the proof
    */
    static void initializeProofFile( const String &filename );

    /*
     Delete the proof file and its content
     */
    static void deleteProof();

    /*
     Get the name of the proof file
    */
    static const String &getProofFilename();

    /*
     Write query assumptions to the proof file
    */
    void flushAssumptions();

    /*
     Write current proof steps to the proof file
    */
    void flushProof();

    /*
     Write concluding resolution steps stored as AletheStepEntry to the proof file + flush remaining
     proof steps. When used in SnC, this concludes the proof of each worker.
    */
    void finalizeProof();

    // -------  CDCL SOLVING  -------

    /*
     On SnC mode, conclude overall UNSAT from all workers UNSAT proofs.
     */
    static void writeFinalStepsToProof();

#if BUILD_CADICAL
    /*
     Write proof hole for a delegated leaf clause
    */
    void writeDelegatedLeaf( int64_t id, const std::vector<int> &clause );

    /*
     An IPASIR-UP method for adding SAT-derived clauses to the proof
    */
    void add_derived_clause( int64_t id,
                             bool redundant,
                             int witness,
                             const std::vector<int> &clause,
                             const std::vector<int64_t> &antecedents );

    /*
     An IPASIR-UP method for adding Theory-derived clauses to the proof
    */
    void add_original_clause( int64_t id,
                              bool redundant,
                              const std::vector<int> &clause,
                              bool restored = false );

    /*
     Store and access the last contradiction added to the proof
    */
    void setLastContradiction( const SparseUnsortedList &contradiction );
    const Set<int> &getLastContradictionClause() const;

    void setLastContradictionClause( Set<int> &clause );

    /*
     Add dummy contradiction to delegated leaves for identification and bug aviodance.
    */
    void addDummyContradiction();

    /*
     Add a proof entry to the struct
     */
    void addEntryToStack( const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &entry );

    /*
     Check if current search state has any proof information
    */
    bool hasInfo() const;

    /*
     Check if a lemma was already added as a reason clause
    */
    bool lemmaExistsAsReasonClause( int64_t id ) const;
#endif

private:
    /*
     Static fields for handling multiple workers
    */
    static Map<String, Pair<String, Vector<int>>> unsatJobFinalSteps;
    static std::mutex unsatJobFinalStepsMutex;
    static File proofFile;
    static String proofFilename;
    static std::mutex proofFileMutex;

    /*
     Information of original query
    */
    const SparseMatrix *_initialTableau;
    Vector<String> _tableauAssumptions; // For easy access
    Vector<double> _baseUpperBounds;
    Vector<double> _baseLowerBounds;
    const GroundBoundManager &_groundBoundManager;
    Vector<PiecewiseLinearConstraint *> _plc;
    unsigned _n;
    unsigned _m;

    /*
     Proof steps and assumptions
     */
    List<String> _proof;
    List<String> _assumptions;
    unsigned _stepCounter;

    /*
     Map first order variable to SAT solver variable
     */
    Map<unsigned, PiecewiseLinearConstraint *> _varToPlc;

    /*
     Map proof info. to their corresponding case splits
    */
    Map<unsigned, List<Tightening>> _idToSplits;
    Map<unsigned, List<Tightening>> _nodeToSplits;

    /*
     Unique ID for the worker
    */
    String _queryId;

    /*
     Information for lazily writing resolution steps
    */
    List<AletheStepEntry> _proofEntries;

#if BUILD_CADICAL
    /*
      Connect to CDCLCore info
    */
    const CdclCore *_cdclCore;

    /*
     Store information for last derived lemma and contradiction
    */
    Vector<std::shared_ptr<GroundBoundManager::GroundBoundEntry>> _lastExplainedEntries;
    SparseUnsortedList _lastContradiction;
    Set<int> _lastContradictionClause;

    /*
     Map reason clauses to the variable they explain
    */
    Map<int64_t, unsigned> _satIdToCdclVar;

    /*
     Convert a clause from SAT based numerals to defined constraints
    */
    static String clauseToPhases( const std::vector<int> &clause );

    /*
     Add stored information of derived clause to the proof
    */
    void writeDerivedClauseContent( int64_t id,
                                    const std::vector<int> &clause,
                                    const std::vector<int64_t> &antecedents );
    /*
     Add a trivial clause for representig a trivially solved SnC subproblem
    */
    void writeSncLitTrivialClause( int64_t id, unsigned sncVar );

    /*
     A helper function that mimics a simple boolean resolution
    */
    static Vector<int> resolution( const Vector<int> &c1, const Vector<int> &c2 );

    /*
     Add proof steps for lemma resolution.
    */
    void writeLemmaResolution( const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &entry,
                               int propagatedLit,
                               int64_t id );
#endif

    /*
     Add original query assumptions to the proof file
    */
    void writeBoundAssumptions();

    void writePLCAssumption();

    void writeTableauAssumptions();

    /*
     Add proof steps for proving a lemma learned from a ReLU activation constraint/
    */
    bool writeReluLemma( const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &lemmaEntry,
                         const ReluConstraint *relu );

    /*
     Collect all case splits of path to a proof node
     */
    List<PiecewiseLinearCaseSplit> getPathSplits( const UnsatCertificateNode *node ) const;

    /*
     Convert multiple Marabou objects into their corresponding Alethe clause
    */
    String getNegatedSplitsClause( const List<PiecewiseLinearCaseSplit> &splits ) const;

    String getSplitsAsClause( const List<PiecewiseLinearCaseSplit> &splits ) const;

    String getSplitAsClause( const PiecewiseLinearCaseSplit &split ) const;

    String getBoundAsClause( const Tightening &bound ) const;

    String convertTableauAssumptionToClause( unsigned index ) const;

    /*
     Check if a case split object represents the active ReLU phase
     */
    bool isSplitActive( const PiecewiseLinearCaseSplit &split ) const;

    /*
     Compute linear combintaions from proof vectors using arbitrary precision arithmetic
    */
    void linearCombinationMpq( const std::vector<mpq_t> &explainedRow,
                               const SparseUnsortedList &expl ) const;

    /*
     A helper function that converts proof vector information to la_generic arguments and clauses as
     Strings
    */
    void farkasStrings( const SparseUnsortedList &expl,
                        unsigned entryId,
                        String &farkasArgs,
                        String &farkasClause,
                        String &farkasParticipants,
                        String &negatedSplitClause,
                        int explainerVar,
                        bool isUpper,
                        const Set<int> &deps,
                        UnsatCertificateNode *node );
};

#endif // __AletheProofWriter_h__