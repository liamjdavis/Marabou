/*********************                                                        */
/*! \file AletheProofWriter.cpp
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

#include "AletheProofWriter.h"

#include "CdclCore.h"
#include "ConstSimpleData.h"
#include "HeapData.h"
#include "MString.h"
#include "Options.h"

#include <filesystem>

namespace fs = std::filesystem;

const unsigned AletheProofWriter::ALETHE_WRITER_PRECISION =
    (unsigned)1 / GlobalConfiguration::LEMMA_CERTIFICATION_TOLERANCE;

Map<String, Pair<String, Vector<int>>> AletheProofWriter::unsatJobFinalSteps{};
std::mutex AletheProofWriter::unsatJobFinalStepsMutex{};
File AletheProofWriter::proofFile( "" );
String AletheProofWriter::proofFilename;
std::mutex AletheProofWriter::proofFileMutex{};

AletheProofWriter::AletheProofWriter( unsigned explanationSize,
                                      const Vector<double> &upperBounds,
                                      const Vector<double> &lowerBounds,
                                      const GroundBoundManager &groundBoundManager,
                                      const SparseMatrix *tableau,
                                      const List<PiecewiseLinearConstraint *> &problemConstraints,
                                      const String &queryId,
                                      const CdclCore *cdclCore )
    : _initialTableau( tableau )
    , _baseUpperBounds( upperBounds )
    , _baseLowerBounds( lowerBounds )
    , _groundBoundManager( groundBoundManager )
    , _plc( problemConstraints.begin(), problemConstraints.end() )
    , _proof()
    , _assumptions()
    , _n( upperBounds.size() )
    , _m( explanationSize )
    , _stepCounter( 1 )
    , _varToPlc()
    , _idToSplits()
    , _nodeToSplits()
    , _queryId( queryId )
    , _proofEntries( {} )
#if BUILD_CADICAL
    , _cdclCore( cdclCore )
    , _lastExplainedEntries( {} )
#endif
{
    for ( const auto &plc : problemConstraints )
    {
        for ( const auto var : plc->getParticipatingVariables() )
            _varToPlc.insert( var, plc );

        _varToPlc.insert( plc->getTableauAuxVars().front(), plc );
    }

    // Write only necessary lines upon initialization
    writeTableauAssumptions();
    _lastContradiction.initializeToEmpty();
}

void AletheProofWriter::writeTableauAssumptions()
{
    ASSERT( _assumptions.empty() );

    List<String> smtLib = SmtLibWriter::convertToSmtLib(
        _m,
        _n,
        _baseUpperBounds,
        _baseLowerBounds,
        _initialTableau,
        List<Equation>(),
        List<PiecewiseLinearConstraint *>( _plc.begin(), _plc.end() ) );

    unsigned counter = 0;
    String assumptionTitle;

    // Convert assertions to assumptions
    for ( auto line : smtLib )
    {
        // Ignore header and footer
        if ( line.contains( "declare" ) || line.contains( "set-logic" ) ||
             line.contains( "check" ) || line.contains( "exit" ) || line.contains( "<=" ) ||
             line.contains( ">=" ) )
            continue;

        ASSERT( line.contains( "=" ) );
        line = line.substring( 0, line.length() - 2 );
        assumptionTitle = "e" + std::to_string( counter ) + "(!";

        line.replace( "assert ", String( "assume " ) + assumptionTitle );
        line += ":named e" + std::to_string( counter ) + "))\n";
        ++counter;

        _assumptions.append( line );
        _tableauAssumptions.append( line );
    }
}

void AletheProofWriter::writeBoundAssumptions()
{
    for ( unsigned i = 0; i < _n; ++i )
    {
        String s = std::to_string( i );
        String upper = String( "(assume u" ) + s + "(!(<= x" + s + " " +
                       SmtLibWriter::signedValue( _baseUpperBounds[i] ) + "):named u" + s + "))\n";
        String lower = String( "(assume l" ) + s + "(!(>= x" + s + " " +
                       SmtLibWriter::signedValue( _baseLowerBounds[i] ) + "):named l" + s + "))\n";
        _assumptions.append( { upper, lower } );
    }
}

void AletheProofWriter::writePLCAssumption()
{
    List<String> plcAssumptions = List<String>();
    List<String> plcSplits = List<String>();

    for ( const auto &plc : _plc )
    {
        List<PiecewiseLinearCaseSplit> splitsInFixedOrder = {};
        // TODO support additional types
        int constraintInt = Options::get()->getBool( Options::SOLVE_WITH_CDCL )
                              ? plc->getVariableForDecision()
                              : plc->getTableauAuxVars().front();

        String constraintNum = std::to_string( constraintInt );
        String plcAssumption = "";

        if ( plc->getType() == RELU )
        {
            splitsInFixedOrder.append( { plc->getCaseSplit( RELU_PHASE_ACTIVE ),
                                         plc->getCaseSplit( RELU_PHASE_INACTIVE ) } );

            ReluConstraint *relu = (ReluConstraint *)plc;
            String f = std::to_string( relu->getF() );
            String b = std::to_string( relu->getB() );
            String aux = std::to_string( relu->getAux() );
            String counterpartAux = std::to_string( plc->getTableauAuxVars().front() );
            String bEqualsF = String( "(= x" ) + b + " x" + f + ")";

            plcAssumption += String( "(assume relu" ) + constraintNum + " (ite (!(>= x" + b +
                             " 0.0):named a" + constraintNum + ")" + bEqualsF + "(<= x" + f +
                             " 0.0)))\n";
            plcAssumptions.append( plcAssumption );

            String ite1 = String( "(step ri1_" ) + constraintNum + " (cl (>= x" + b + " 0.0)(<= x" +
                          f + " 0.0)):rule ite1 :premises(relu" + constraintNum + "))\n";
            String ite2 = String( "(step ri2_" ) + constraintNum + " (cl (not (>= x" + b +
                          " 0.0))" + bEqualsF + "):rule ite2 :premises(relu" + constraintNum +
                          "))\n";
            String tot = String( "(step _bt" ) + constraintNum + " (cl (or (not (>= x" + b +
                         " 0.0))(>= x" + b + " 0.0))):rule la_tautology)\n";
            tot += String( "(step bt" ) + constraintNum + " (cl (not (>= x" + b + " 0.0))(>= x" +
                   b + " 0.0)):rule or :premises(_bt" + constraintNum + "))\n";

            plcSplits.append( { ite1, ite2, tot } );
            unsigned identifierInt = relu->getTableauAuxVars().front();
            String tableauEq = "e" + std::to_string( identifierInt - ( _n - _m ) );
            String tableauLit = convertTableauAssumptionToClause( identifierInt - ( _n - _m ) );

            String activeBound1 = String( "(step ab1_" ) + constraintNum + " (cl (not " + bEqualsF +
                                  ")" + tableauLit + "(<= x" + aux + " 0.0)(not (>= x" +
                                  counterpartAux + " 0.0))):rule la_generic :args(1 -1 1 1))\n";
            activeBound1 += String( "(step eq" ) + constraintNum + "_a0" + " (cl (not (>= x" + b +
                            " 0.0))(<= x" + aux + " 0.0)):rule resolution :premises(ab1_" +
                            constraintNum + " ri2_" + constraintNum + " l" + counterpartAux + " " +
                            tableauEq + "))\n";

            String activeBound2 = String( "(step ab2_" ) + constraintNum + " (cl (>= x" + b +
                                  " 0.0)" + tableauLit + "(not (<= x" + aux + " 0.0))(not (<= x" +
                                  counterpartAux + " 0.0))(not (>= x" + f +
                                  " 0.0))):rule la_generic :args(1 1 1 1 -1))\n";
            activeBound2 += String( "(step eq" ) + constraintNum + "_a1" + " (cl (>= x" + b +
                            " 0.0)(not (<= x" + aux + " 0.0))):rule resolution :premises(ab2_" +
                            constraintNum + " ri1_" + constraintNum + " u" + counterpartAux + " l" +
                            f + " " + tableauEq + "))\n";

            String inactiveBound1 = String( "(step ib1_" ) + constraintNum + " (cl (not " +
                                    bEqualsF + ")(not(<= x" + b + " 0.0))(<= x" + f +
                                    " 0.0)):rule la_generic :args(1 1 1))\n";
            inactiveBound1 += String( "(step eq" ) + constraintNum + "_i0" + " (cl (not (<= x" + b +
                              " 0.0))(<= x" + f + " 0.0)):rule resolution :premises(ib1_" +
                              constraintNum + " ri1_" + constraintNum + " ri2_" + constraintNum +
                              "))\n";

            String inactiveBound2 = String( "(step ib2_" ) + constraintNum + " (cl (not " +
                                    bEqualsF + ")(<= x" + b + " 0.0)(not (<= x" + f +
                                    " 0.0))):rule la_generic :args(-1 1 1))\n";
            inactiveBound2 += String( "(step eq" ) + constraintNum + "_i1" + " (cl (not (<= x" + f +
                              " 0.0))(<= x" + b + " 0.0)):rule resolution :premises(ib2_" +
                              constraintNum + " ri2_" + constraintNum + " bt" + constraintNum +
                              "))\n";

            plcSplits.append( { activeBound1, activeBound2, inactiveBound1, inactiveBound2 } );
        }
    }

    _assumptions.append( plcAssumptions );
    _assumptions.append( plcSplits );
}

void AletheProofWriter::writeContradiction( const SparseUnsortedList &contradiction, int64_t id )
{
    String farkasArgs = "";
    String farkasClause = "";
    String farkasParticipants = "";
    String negatedSplitsClause = "";
    farkasStrings( contradiction,
                   _groundBoundManager.getCounter(),
                   farkasArgs,
                   farkasClause,
                   farkasParticipants,
                   negatedSplitsClause,
                   -id,
                   true,
                   {} );

    farkasClause = String( "(cl " ) + farkasClause + ")";
    farkasArgs = String( "(" ) + farkasArgs + "))\n";
#ifdef BUILD_CADICAL
    if ( _cdclCore )
    {
        std::vector<int> contradictionClause =
            std::vector<int>( _lastContradictionClause.begin(), _lastContradictionClause.end() );
        negatedSplitsClause += clauseToPhases( contradictionClause );

        for ( const int lit : contradictionClause )
        {
            String identifier = std::to_string( abs( lit ) );
            if ( lit > 0 )
                farkasParticipants += String( "eq" ) + identifier + "_a0 ";
            else
                farkasParticipants += String( "eq" ) + identifier + "_i1 ri1_" + identifier + " ";
        }
    }
#endif
    String ruleName = GlobalConfiguration::DEDICATED_ALEHTE_RULE ? "bounded_farkas" : "la_generic";
    String laGeneric = String( "(step t" ) + _queryId + "_" + std::to_string( id ) + " " +
                       farkasClause + ":rule " + ruleName + " :args" + farkasArgs;

    String res = String( "(step r" ) + _queryId + "_" + std::to_string( id ) + " (cl " +
                 negatedSplitsClause + "):rule resolution :premises(t" + _queryId + "_" +
                 std::to_string( id ) + " " + farkasParticipants + "))\n";

    _proof.append( { laGeneric, res } );
}

void AletheProofWriter::finalizeProof()
{
    for ( auto stepEntry : _proofEntries )
    {
        // Lemma Resolution
        if ( stepEntry.gbEntry && stepEntry.id >= 0 )
            writeLemmaResolution( stepEntry.gbEntry, stepEntry.propagatedLit, stepEntry.id );
        // Delegation Leaf
        else if ( stepEntry.contradiction.getSize() && stepEntry.contradiction.empty() )
            writeDelegatedLeaf( stepEntry.id, stepEntry.clause );
        // Derived Clauses
        else if ( stepEntry.id >= 0 )
            writeDerivedClauseContent( stepEntry.id, stepEntry.clause, stepEntry.antecedents );
    }

    AletheProofWriter::proofFileMutex.lock();
    AletheProofWriter::proofFile.open( File::MODE_WRITE_APPEND );

    for ( const String &s : _proof )
        AletheProofWriter::proofFile.write( s );

    AletheProofWriter::proofFile.close();
    AletheProofWriter::proofFileMutex.unlock();

    if ( !_cdclCore )
        return;

    String resId;
    if ( !_proofEntries.empty() )
        resId = String( "r" ) + _queryId + "_" + std::to_string( _proofEntries.back().id );
    else
    {
        List<String> finalStepTokens = _proof.back().tokenize( " " );
        finalStepTokens.popFront();
        resId = finalStepTokens.popFront();
    }

    AletheProofWriter::unsatJobFinalStepsMutex.lock();
    AletheProofWriter::unsatJobFinalSteps.insert(
        _queryId, Pair<String, Vector<int>>( resId, _cdclCore->getSncLits() ) );
    AletheProofWriter::unsatJobFinalStepsMutex.unlock();

    _proof.clear();
}

void AletheProofWriter::deleteProof()
{
    AletheProofWriter::proofFileMutex.lock();
    AletheProofWriter::proofFile.open( File::MODE_WRITE_TRUNCATE );
    AletheProofWriter::proofFile.write( "" );
    AletheProofWriter::proofFile.close();
    std::remove( AletheProofWriter::proofFilename.ascii() );
    AletheProofWriter::proofFileMutex.unlock();
}

void AletheProofWriter::writeInstanceToFile( IFile &file )
{
    file.open( File::MODE_WRITE_TRUNCATE );
    writeBoundAssumptions();
    writePLCAssumption();
    for ( const String &s : _assumptions )
        file.write( s );

    for ( auto stepEntry : _proofEntries )
    {
        // Lemma Resolution
        if ( stepEntry.gbEntry && stepEntry.id >= 0 )
            writeLemmaResolution( stepEntry.gbEntry, stepEntry.propagatedLit, stepEntry.id );
        // Delegation Leaf
        else if ( stepEntry.contradiction.getSize() && stepEntry.contradiction.empty() )
            writeDelegatedLeaf( stepEntry.id, stepEntry.clause );
        // Derived Clauses
        else if ( stepEntry.id >= 0 )
            writeDerivedClauseContent( stepEntry.id, stepEntry.clause, stepEntry.antecedents );
    }

    for ( const String &s : _proof )
        file.write( s );

    file.close();
}

void AletheProofWriter::writeChildrenConclusion( const UnsatCertificateNode *node )
{
    if ( !node->isValidNonLeaf() )
        return;

    List<unsigned> childrenIndices = {};
    for ( const auto &child : node->getChildren() )
        childrenIndices.append( child->getId() );

    ASSERT( node->isValidNonLeaf() );
    ASSERT( childrenIndices.size() == 2 );
    PiecewiseLinearCaseSplit firstChildSplit = node->getChildren().front()->getSplit();
    PiecewiseLinearCaseSplit secondChildSplit = node->getChildren().back()->getSplit();

    List<Tightening> tighteningDeps = _nodeToSplits[node->getChildren().front()->getId()];
    tighteningDeps.append( _nodeToSplits[node->getChildren().back()->getId()] );
    List<Tightening> filteredTighteneings = {};
    Set<int> phaseIdentifiers = {};
    List<PiecewiseLinearCaseSplit> splitDeps = {};

    for ( const auto &tightening : tighteningDeps )
    {
        if ( firstChildSplit.getBoundTightenings().exists( tightening ) ||
             secondChildSplit.getBoundTightenings().exists( tightening ) )
            continue;

        PiecewiseLinearConstraint *plc = _varToPlc[tightening._variable];

        for ( const auto &caseSplit : plc->getAllCases() )
        {
            PiecewiseLinearCaseSplit split = plc->getCaseSplit( caseSplit );
            if ( split.getBoundTightenings().exists( tightening ) )
                phaseIdentifiers.insert( isSplitActive( split )
                                             ? (int)plc->getTableauAuxVars().front()
                                             : -(int)plc->getTableauAuxVars().front() );
        }
    }

    for ( const auto phase : phaseIdentifiers )
    {
        if ( !phaseIdentifiers.exists( -phase ) )
        {
            PiecewiseLinearCaseSplit splitToAdd;
            PiecewiseLinearConstraint *plc = _varToPlc[abs( phase )];

            // TODO support additional types of splits
            if ( plc->getType() == RELU )
                splitToAdd = phase > 0 ? plc->getCaseSplit( RELU_PHASE_ACTIVE )
                                       : plc->getCaseSplit( RELU_PHASE_INACTIVE );

            splitDeps.append( splitToAdd );
            filteredTighteneings.append( splitToAdd.getBoundTightenings().front() );
        }
    }

    _nodeToSplits.insert( node->getId(), filteredTighteneings );

    ASSERT( node->isValidNonLeaf() );
    ASSERT( childrenIndices.size() == 2 )
    String resLine = String( "(step r_" + std::to_string( node->getId() ) + " (cl " ) +
                     getNegatedSplitsClause( splitDeps ) + "):rule resolution :premises(r_" +
                     std::to_string( childrenIndices.front() ) + " r_" +
                     std::to_string( childrenIndices.back() ) + "))\n";

    _proof.append( resLine );
}

String
AletheProofWriter::getNegatedSplitsClause( const List<PiecewiseLinearCaseSplit> &splits ) const
{
    if ( splits.empty() )
        return "";

    String clause = "";
    for ( const auto &split : splits )
    {
        String isActive = isSplitActive( split ) ? "(not a" : "a";
        PiecewiseLinearConstraint *plc = _varToPlc[split.getBoundTightenings().front()._variable];
        int constraintInt = Options::get()->getBool( Options::SOLVE_WITH_CDCL )
                              ? plc->getVariableForDecision()
                              : plc->getTableauAuxVars().front();
        String plcNum = std::to_string( constraintInt );
        String suffix = isSplitActive( split ) ? ")" : "";
        clause += String( " " ) + isActive + plcNum + suffix;
    }
    return clause;
}

String AletheProofWriter::getBoundAsClause( const Tightening &bound ) const
{
    if ( bound._type == Tightening::UB )
        return String( "(<= x" + std::to_string( bound._variable ) + " " ) +
               SmtLibWriter::signedValue( bound._value ) + ")";

    return String( "(>= x" + std::to_string( bound._variable ) + " " ) +
           SmtLibWriter::signedValue( bound._value ) + ")";
}

String AletheProofWriter::getSplitAsClause( const PiecewiseLinearCaseSplit &split ) const
{
    ASSERT( split.getEquations().empty() );
    String clause = "(and ";
    for ( const auto &bound : split.getBoundTightenings() )
        clause += getBoundAsClause( bound );

    clause += ")";
    return clause;
}

String AletheProofWriter::getSplitsAsClause( const List<PiecewiseLinearCaseSplit> &splits ) const
{
    String clause = "";
    for ( const auto &split : splits )
        clause += getSplitAsClause( split );
    return clause;
}

bool AletheProofWriter::isSplitActive( const PiecewiseLinearCaseSplit &split ) const
{
    ASSERT( split.getEquations().empty() )
    return split.getBoundTightenings().back()._type == Tightening::LB ||
           split.getBoundTightenings().front()._type == Tightening::LB;
}

List<PiecewiseLinearCaseSplit>
AletheProofWriter::getPathSplits( const UnsatCertificateNode *node ) const
{
    List<PiecewiseLinearCaseSplit> pathSplits = List<PiecewiseLinearCaseSplit>();
    const UnsatCertificateNode *cur = node;
    while ( cur && !cur->getSplit().getBoundTightenings().empty() )
    {
        pathSplits.append( cur->getSplit() );
        cur = cur->getParent();
    }

    return pathSplits;
}

void AletheProofWriter::writeLemma(
    const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &lemmaEntry )
{
    if ( !lemmaEntry->lemma || lemmaEntry->lemma->wasWritten() || !lemmaEntry->lemma->getToCheck() )
        return;

    PiecewiseLinearConstraint *matchedConstraint = _varToPlc[lemmaEntry->lemma->getAffectedVar()];
    bool matched = false;
    // TODO add support for all types of PLCs
    if ( matchedConstraint && matchedConstraint->getType() == RELU )
        matched = writeReluLemma( lemmaEntry, (ReluConstraint *)matchedConstraint );

    if ( matched )
        lemmaEntry->lemma->setWritten();
}

bool AletheProofWriter::writeReluLemma(
    const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &lemmaEntry,
    const ReluConstraint *relu )
{
    ASSERT( lemmaEntry->lemma && lemmaEntry->lemma->getConstraintType() == RELU );
    const std::shared_ptr<PLCLemma> lemma = lemmaEntry->lemma;

    unsigned causingVar = lemma->getCausingVars().front();
    unsigned affectedVar = lemma->getAffectedVar();
    double targetBound = lemma->getMinTargetBound();
    double bound = lemma->getBound();

    String id = std::to_string( lemma->getId() );
    const List<SparseUnsortedList> &explanations = lemma->getExplanations();
    Tightening::BoundType causingVarBound = lemma->getCausingVarBound();
    Tightening::BoundType affectedVarBound = lemma->getAffectedVarBound();

    ASSERT( relu == _varToPlc[affectedVar] );
    ASSERT( explanations.size() == 1 );

    String farkasArgs = "";
    String farkasClause = "";
    String farkasParticipants = "";
    String negatedSplitsClause = "";
    String causeBound = getBoundAsClause( Tightening( causingVar, targetBound, causingVarBound ) );

    farkasStrings( explanations.front(),
                   lemmaEntry->id,
                   farkasArgs,
                   farkasClause,
                   farkasParticipants,
                   negatedSplitsClause,
                   causingVar,
                   causingVarBound == Tightening::UB,
                   lemmaEntry->deps );
#ifdef BUILD_CADICAL
    if ( _cdclCore )
    {
        std::vector<int> entryClause =
            std::vector<int>( lemmaEntry->clause.begin(), lemmaEntry->clause.end() );

        negatedSplitsClause += clauseToPhases( entryClause );

        for ( const int lit : entryClause )
        {
            String identifier = std::to_string( abs( lit ) );
            if ( lit > 0 )
                farkasParticipants += String( "eq" ) + identifier + "_a0 ";
            else
                farkasParticipants += String( "eq" ) + identifier + "_i1 ri1_" + identifier + " ";
        }
    }
#endif

    farkasClause = String( "(cl " ) + causeBound + farkasClause + ")";
    farkasArgs = String( "(1 " ) + farkasArgs + "))\n";

    String ruleName = GlobalConfiguration::DEDICATED_ALEHTE_RULE ? "bounded_farkas" : "la_generic";
    String laGeneric = String( "(step fl" ) + _queryId + "_" + id + " " + farkasClause + ":rule " +
                       ruleName + " :args" + farkasArgs;

    String res = String( "(step cr" ) + _queryId + "_" + id + " (cl " + negatedSplitsClause +
                 causeBound + "):rule resolution :premises(fl" + _queryId + "_" + id + " " +
                 farkasParticipants + "))\n";

    unsigned b = relu->getB();
    unsigned f = relu->getF();
    unsigned aux = relu->getAux();
    int constraintInt = Options::get()->getBool( Options::SOLVE_WITH_CDCL )
                          ? relu->getVariableForDecision()
                          : relu->getTableauAuxVars().front();

    String identifier = std::to_string( constraintInt );
    bool matched = false;

    String proofRule = "";
    String proofRuleRes = "";
    String tempString = "";

    if ( targetBound > 0 && ( causingVar == f || causingVar == aux ) )
        tempString += String( "(not " ) +
                      getBoundAsClause( Tightening( causingVar, 0, Tightening::UB ) ) + ")(not " +
                      causeBound + ")";
    else
        tempString += String( "(not" ) + causeBound + ")" +
                      getBoundAsClause( Tightening(
                          causingVar, 0, targetBound > 0 ? Tightening::LB : Tightening::UB ) );

    if ( targetBound != 0 )
    {
        proofRule = String( "(step taut" ) + _queryId + "_" + id + " (cl (or " + tempString +
                    ")):rule la_tautology)\n";
        proofRule += String( "(step ts" ) + _queryId + "_" + id + " (cl " + tempString +
                     "):rule or :premises(taut" + _queryId + "_" + id + "))\n";
    }
    String conclusion = getBoundAsClause( Tightening( affectedVar, bound, affectedVarBound ) );

    String pref = String( "(step rl" ) + _queryId + "_" + id + " (cl " + negatedSplitsClause +
                  conclusion + "):rule resolution :premises(cr" + _queryId + "_" + id;
    if ( ( causingVar == f || causingVar == b ) && causingVarBound == Tightening::LB &&
         affectedVar == aux && affectedVarBound == Tightening::UB && targetBound > 0 )
    {
        matched = true;
        proofRuleRes = pref + " ts" + _queryId + "_" + id;

        if ( causingVar == b )
            proofRuleRes += String( " eq" ) + identifier + "_a0";
        else
            proofRuleRes += String( " eq" ) + identifier + "_a0 ri1_" + identifier;

        proofRuleRes += "))\n";
    }
    else if ( causingVar == b && causingVarBound == Tightening::LB && affectedVar == aux &&
              affectedVarBound == Tightening::UB && targetBound == 0 )
    {
        matched = true;
        proofRuleRes = pref + " eq" + identifier + "_a0))\n";
    }
    // If lb of aux is positive, then ub of f is 0
    else if ( causingVar == aux && causingVarBound == Tightening::LB && affectedVar == f &&
              affectedVarBound == Tightening::UB && targetBound > 0 )
    {
        matched = true;
        proofRuleRes = pref + " ts" + _queryId + "_" + id + " eq" + identifier + "_a0 ri1_" +
                       identifier + "))\n";
    }

    // If ub of b is non positive, then ub of f is 0
    else if ( causingVar == b && causingVarBound == Tightening::UB && affectedVar == f &&
              affectedVarBound == Tightening::UB && targetBound < 0 )
    {
        matched = true;
        proofRuleRes = pref + " ts" + _queryId + "_" + id + " eq" + identifier + "_i0))\n";
    }
    // Propagate 0 ub from f to b
    else if ( causingVar == f && causingVarBound == Tightening::UB && affectedVar == b &&
              affectedVarBound == Tightening::UB && targetBound == 0 )
    {
        matched = true;
        proofRuleRes = pref + " eq" + identifier + "_i1))\n";
    }
    else if ( causingVar == b && causingVarBound == Tightening::UB && affectedVar == f &&
              affectedVarBound == Tightening::UB && targetBound == 0 )
    {
        matched = true;
        proofRuleRes = pref + +" eq" + identifier + "_i0))\n";
    }
    // If ub of aux is 0, then lb of b is 0
    else if ( causingVar == aux && causingVarBound == Tightening::UB && affectedVar == b &&
              affectedVarBound == Tightening::LB && targetBound == 0 )

    {
        matched = true;
        proofRuleRes = pref + " eq" + identifier + "_a1))\n";
    }
    // If lb of b is negative x, then ub of aux is -x
    else if ( causingVar == b && causingVarBound == Tightening::LB && affectedVar == aux &&
              affectedVarBound == Tightening::UB && targetBound < 0 )
    {
        matched = true;

        unsigned identifierInt = relu->getTableauAuxVars().front();
        String tautClause = String( "(not " ) +
                            getBoundAsClause( Tightening( affectedVar, 0, Tightening::UB ) ) + ")" +
                            conclusion;
        proofRule = String( "(step taut" ) + _queryId + "_" + id + " (cl (or " + tautClause +
                    ")):rule la_tautology)\n";

        proofRule += String( "(step ts" ) + _queryId + "_" + id + " (cl " + tautClause +
                     "):rule or :premises(taut" + _queryId + "_" + id + "))\n";

        String counterpartBound =
            getBoundAsClause( Tightening( identifierInt, 0, Tightening::LB ) );
        String subConclusion = getBoundAsClause( Tightening( f, 0, Tightening::UB ) );
        String tableauLit = convertTableauAssumptionToClause( identifierInt - ( _n - _m ) );
        String subFarkasClause = String( " (cl (not " ) + causeBound + ")" + conclusion + "(not " +
                                 subConclusion + ")(not " + counterpartBound + ")" + tableauLit;

        proofRule += String( "(step ifl" ) + _queryId + "_" + id + subFarkasClause +
                     "):rule la_generic :args(1 1 -1 1 -1))\n";

        proofRuleRes += pref + +" e" + std::to_string( identifierInt - ( _n - _m ) ) + " l" +
                        std::to_string( identifierInt ) + " ifl" + _queryId + "_" + id + " ts" +
                        _queryId + "_" + id + " eq" + identifier + "_a0 ri1_" + identifier + "))\n";
    }
    // If ub of b is positive, then propagate to f
    else if ( causingVar == b && causingVarBound == Tightening::UB && affectedVar == f &&
              affectedVarBound == Tightening::UB && targetBound > 0 )
    {
        matched = true;

        unsigned identifierInt = relu->getTableauAuxVars().front();
        String tautClause = String( "(not " ) +
                            getBoundAsClause( Tightening( affectedVar, 0, Tightening::UB ) ) + ")" +
                            conclusion;

        proofRule = String( "(step taut" ) + _queryId + "_" + id + " (cl (or " + tautClause +
                    ")):rule la_tautology)\n";

        proofRule += String( "(step ts" ) + _queryId + "_" + id + " (cl " + tautClause +
                     "):rule or :premises(taut" + _queryId + "_" + id + "))\n";

        String counterpartBound =
            getBoundAsClause( Tightening( identifierInt, 0, Tightening::UB ) );
        String subConclusion = getBoundAsClause( Tightening( aux, 0, Tightening::UB ) );
        String tableauLit = convertTableauAssumptionToClause( identifierInt - ( _n - _m ) );
        String subFarkasClause = String( " (cl (not " ) + causeBound + ")" + conclusion + "(not " +
                                 subConclusion + ")(not " + counterpartBound + ")" + tableauLit;

        proofRule += String( "(step ifl" ) + _queryId + "_" + id + subFarkasClause +
                     "):rule la_generic :args(1 1 -1 1 1))\n";

        proofRuleRes += pref + " e" + std::to_string( identifierInt - ( _n - _m ) ) + " u" +
                        std::to_string( identifierInt ) + " ifl" + _queryId + "_" + id + " ts" +
                        _queryId + "_" + id + " eq" + identifier + "_a0 ri1_" + identifier + "))\n";
    }

    if ( matched )
        _proof.append( { laGeneric, res, proofRule, proofRuleRes } );
    return matched;
}

void AletheProofWriter::linearCombinationMpq( const std::vector<mpq_t> &explainedRow,
                                              const SparseUnsortedList &expl ) const
{
    SparseUnsortedList tableauRow( _n );
    for ( const auto &entry : expl )
    {
        if ( entry._value == 0 )
            continue;

        _initialTableau->getRow( entry._index, &tableauRow );
        for ( const auto &tableauEntry : tableauRow )
        {
            if ( tableauEntry._value != 0 )
            {
                mpq_t tempval, tempEntry, tempTableauEntry;
                mpq_init( tempval );
                mpq_init( tempEntry );
                mpq_init( tempTableauEntry );
                mpq_set_d( tempTableauEntry, tableauEntry._value );
                mpq_set_d( tempEntry, entry._value );
                mpq_mul( tempval, tempEntry, tempTableauEntry );
                mpq_add( const_cast<mpq_ptr>( explainedRow[tableauEntry._index] ),
                         explainedRow[tableauEntry._index],
                         tempval );
                mpq_clear( tempval );
                mpq_clear( tempEntry );
                mpq_clear( tempTableauEntry );
            }
        }
    }
}

void AletheProofWriter::farkasStrings( const SparseUnsortedList &expl,
                                       unsigned entryId,
                                       String &farkasArgs,
                                       String &farkasClause,
                                       String &farkasParticipants,
                                       String &negatedSplitClause,
                                       int explainedVar,
                                       bool isUpper,
                                       const Set<int> &deps )
{
    std::vector<mpq_t> explainedRow = std::vector<mpq_t>( _n );
    for ( const auto num : explainedRow )
        mpq_init( num );

    linearCombinationMpq( explainedRow, expl );
    bool isLemma = explainedVar >= 0;
    if ( isLemma )
    {
        mpq_t temp;
        mpq_init( temp );
        mpq_set_d( temp, 1 );
        mpq_add(
            const_cast<mpq_ptr>( explainedRow[explainedVar] ), explainedRow[explainedVar], temp );
        mpq_clear( temp );
    }

    farkasClause = "";
    farkasArgs = "";
    farkasParticipants = "";
    List<Tightening> splitDeps;

    for ( const auto entry : expl )
        if ( entry._value != 0 )
        {
            farkasClause += String( "(not e" + std::to_string( entry._index ) ) + ")";

            mpq_class temp( isUpper ? -entry._value : entry._value );
            farkasArgs += temp.get_str() + " ";
            farkasParticipants += String( "e" + std::to_string( entry._index ) ) + " ";
        }

    for ( unsigned i = 0; i < _n; ++i )
    {
        mpq_class temp( explainedRow[i] );
        if ( mpq_sgn( explainedRow[i] ) == 0 )
            continue;

        bool useEntryUpperBound = ( mpq_sgn( explainedRow[i] ) > 0 && isUpper ) ||
                                  ( mpq_sgn( explainedRow[i] ) < 0 && !isUpper );

        String boundString = useEntryUpperBound ? "u" : "l";
        String ineqString = useEntryUpperBound ? "<=" : ">=";
        Tightening::BoundType boundType = useEntryUpperBound ? Tightening::UB : Tightening::LB;
        const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &gbEntry =
            _groundBoundManager.getGroundBoundEntryUpToId( i, boundType, entryId );

        bool overrideGmp = ( mpq_cmp_si( explainedRow[i], 1, ALETHE_WRITER_PRECISION ) < 0 &&
                             mpq_cmp_si( explainedRow[i], -1, ALETHE_WRITER_PRECISION ) > 0 );

        int lemId = gbEntry->lemma ? gbEntry->lemma->getId() : -1;
        double bound = gbEntry->val;
        bool isLemmaIncluded = lemId >= 0 && gbEntry->lemma->getToCheck() &&
                               gbEntry->lemma->wasWritten() && ( !isLemma || deps.exists( lemId ) );
        bool useSplitBound = ( lemId < 0 && gbEntry->isPhaseFixing );

        if ( !GlobalConfiguration::DEDICATED_ALEHTE_RULE )
            farkasArgs += temp.get_str() + " ";

        if ( ( isLemmaIncluded || useSplitBound ) && !overrideGmp )
            farkasClause += String( "(not (" ) + ineqString + " x" + std::to_string( i ) +
                            String( " " ) + SmtLibWriter::signedValue( bound ) + ")) ";
        else
        {
            farkasClause += String( "(not " ) + boundString + std::to_string( i ) + ") ";
            farkasParticipants += boundString + std::to_string( i ) + " ";
        }

        // Add split deps of prev lemmas
        if ( isLemmaIncluded && !overrideGmp )
        {
            farkasParticipants += String( "rl" ) + _queryId + "_" + std::to_string( lemId ) + " ";
            for ( const auto dep : _idToSplits[gbEntry->id] )
                if ( !splitDeps.exists( dep ) )
                    splitDeps.append( dep );
        }
        else if ( useSplitBound )
            splitDeps.append( Tightening( i, bound, boundType ) );
    }

    for ( const auto num : explainedRow )
        mpq_clear( num );

#if BUILD_CADICAL
    if ( _cdclCore )
        return;
#endif
    if ( isLemma && _idToSplits.exists( entryId ) )
        _idToSplits[entryId] = splitDeps;
    else if ( isLemma )
        _idToSplits.insert( entryId, splitDeps );
    else
        _nodeToSplits.insert( -explainedVar, splitDeps );

    Set<unsigned> usedPlc = {};
    for ( const auto &tightening : splitDeps )
    {
        PiecewiseLinearConstraint *plc = _varToPlc[tightening._variable];
        if ( usedPlc.exists( plc->getTableauAuxVars().front() ) )
            continue;

        usedPlc.insert( plc->getTableauAuxVars().front() );

        int constraintInt = Options::get()->getBool( Options::SOLVE_WITH_CDCL )
                              ? plc->getVariableForDecision()
                              : plc->getTableauAuxVars().front();

        String identifier = std::to_string( constraintInt );

        PiecewiseLinearCaseSplit tighteningSplit;

        for ( const auto &casSplit : plc->getAllCases() )
        {
            PiecewiseLinearCaseSplit split = plc->getCaseSplit( casSplit );
            if ( split.getBoundTightenings().exists( tightening ) )
                tighteningSplit = split;
        }

        if ( isSplitActive( tighteningSplit ) )
            farkasParticipants += String( "eq" ) + identifier + "_a0 ";
        else
            farkasParticipants += String( "eq" ) + identifier + "_i1 ri1_" + identifier + " ";

        String isNegActive = isSplitActive( tighteningSplit ) ? "(not a" : "a";
        String suffix = isSplitActive( tighteningSplit ) ? ")" : " ";
        negatedSplitClause += isNegActive + identifier + suffix;
    }
}

String AletheProofWriter::convertTableauAssumptionToClause( unsigned index ) const
{
    return String( "(not e" ) + std::to_string( index ) + ")";
}

void AletheProofWriter::writeDelegatedLeaf( const UnsatCertificateNode *node )
{
    String proofHole = String( "(step r_" + std::to_string( node->getId() ) ) + " (cl " +
                       getNegatedSplitsClause( getPathSplits( node ) ) + "):rule hole)\n";

    List<Tightening> deps = {};
    for ( const auto &split : getPathSplits( node ) )
        for ( const auto tightening : split.getBoundTightenings() )
            deps.append( tightening );

    _nodeToSplits.insert( node->getId(), deps );
    _proof.append( proofHole );
}


unsigned AletheProofWriter::assignId()
{
    return _stepCounter++;
}

void AletheProofWriter::flushAssumptions()
{
    AletheProofWriter::proofFile.open( File::MODE_WRITE_TRUNCATE );
    writeBoundAssumptions();
    writePLCAssumption();
    for ( const String &s : _assumptions )
        AletheProofWriter::proofFile.write( s );

    AletheProofWriter::proofFile.close();
}

void AletheProofWriter::flushProof()
{
    AletheProofWriter::proofFileMutex.lock();
    AletheProofWriter::proofFile.open( File::MODE_WRITE_APPEND );
    for ( const String &s : _proof )
        AletheProofWriter::proofFile.write( s );

    AletheProofWriter::proofFile.close();
    AletheProofWriter::proofFileMutex.unlock();

    _proof.clear();
}

const String &AletheProofWriter::getProofFilename()
{
    return AletheProofWriter::proofFilename;
}

#if BUILD_CADICAL
void AletheProofWriter::add_derived_clause( int64_t id,
                                            bool /*redundant*/,
                                            int /*witness*/,
                                            const std::vector<int> &clause,
                                            const std::vector<int64_t> &antecedents )
{
    _proofEntries.append(
        AletheStepEntry( id, clause, antecedents, NULL, SparseUnsortedList(), 0 ) );
}


void AletheProofWriter::add_original_clause( int64_t id,
                                             bool /*redundant*/,
                                             const std::vector<int> &clause,
                                             bool /*restored*/ )
{
    std::sort(
        _lastExplainedEntries.begin(),
        _lastExplainedEntries.end(),
        []( std::shared_ptr<GroundBoundManager::GroundBoundEntry> a,
            std::shared_ptr<GroundBoundManager::GroundBoundEntry> b ) { return a->id < b->id; } );

    for ( auto entry : _lastExplainedEntries )
        writeLemma( entry );

    if ( _lastContradiction.getSize() )
    {
        if ( _lastContradiction.empty() )
            _proofEntries.append( AletheStepEntry( id, clause, {}, NULL, _lastContradiction, 0 ) );
        else
            writeContradiction( _lastContradiction, id );

        _lastContradiction.initializeToEmpty();
        _lastContradictionClause.clear();
    }
    else if ( clause.size() == 1 && _cdclCore->getSncLits().exists( clause.front() ) )
    {
        writeSncLitTrivialClause( id, abs( clause.front() ) );
        return;
    }
    else
    {
        ASSERT( !_lastExplainedEntries.empty() )
        ASSERT( _lastExplainedEntries.last()->lemma );
        _satIdToCdclVar.insert( id,
                                _varToPlc[_lastExplainedEntries.last()->lemma->getAffectedVar()]
                                    ->getVariableForDecision() );
        PiecewiseLinearConstraint *plc =
            _varToPlc[_lastExplainedEntries.last()->lemma->getAffectedVar()];
        // Add as minus, as the literal will be negated
        _proofEntries.append( AletheStepEntry( id,
                                               {},
                                               {},
                                               _lastExplainedEntries.last(),
                                               SparseUnsortedList(),
                                               -plc->propagatePhaseAsLit() ) );
    }
    _lastExplainedEntries.clear();
}

void AletheProofWriter::setLastContradiction( const SparseUnsortedList &contradiction )
{
    _lastContradiction = contradiction;
}

void AletheProofWriter::setLastContradictionClause( Set<int> &clause )
{
    _lastContradictionClause = clause;
}

void AletheProofWriter::addDummyContradiction()
{
    _lastContradiction.incrementSize();
}

void AletheProofWriter::addEntryToStack(
    const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &entry )
{
    _lastExplainedEntries.append( entry );
}

void AletheProofWriter::writeDelegatedLeaf( int64_t id, const std::vector<int> &clause )
{
    String proofHole = String( "(step r" ) + _queryId + "_" + std::to_string( id ) + " (cl ";
    for ( int lit : clause )
    {
        String isActive = lit > 0 ? "a" : "(not a";
        String plcNum = std::to_string( abs( lit ) );
        proofHole += String( " " ) + isActive + plcNum + ( lit > 0 ? "" : ")" );
    }
    proofHole += "):rule hole)\n";
    _proof.append( proofHole );
}

void AletheProofWriter::writeLemmaResolution(
    const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &entry,
    int propagatedLit,
    int64_t id )
{
    ASSERT( entry->lemma && entry->isPhaseFixing );
    unsigned lemId = entry->lemma->getId();
    String constraintId = std::to_string( abs( propagatedLit ) );

    std::vector<int> entryClause = std::vector<int>( entry->clause.begin(), entry->clause.end() );
    // Add as minus, as the literal will be negated
    entryClause.insert( entryClause.end(), propagatedLit );
    String preRule = "";
    String proofRule = String( "(step r" ) + _queryId + "_" + std::to_string( id ) + " (cl " +
                       clauseToPhases( entryClause ) + "):rule resolution :premises(";

    // Active phase since propagation of -l implies the clause includes l
    if ( propagatedLit < 0 )
        proofRule += String( "rl" ) + _queryId + "_" + std::to_string( lemId ) + " eq" +
                     constraintId + "_a1))\n";
    else
    {
        unsigned causing = entry->lemma->getCausingVars().front();
        unsigned affected = entry->var;
        if ( causing < affected )
        {
            String lemmaBound = getBoundAsClause(
                Tightening( causing, entry->lemma->getMinTargetBound(), Tightening::UB ) );
            preRule = String( "(step _rt" ) + _queryId + "_" + std::to_string( id ) +
                      " (cl (or (not " + lemmaBound + ")(not (>= x" + std::to_string( causing ) +
                      " 0.0)))):rule la_tautology)\n";
            preRule += String( "(step rt" ) + _queryId + "_" + std::to_string( id ) + " (cl (not " +
                       lemmaBound + ")(not (>= x" + std::to_string( causing ) +
                       " 0.0))):rule or :premises(_rt" + _queryId + "_" + std::to_string( id ) +
                       "))\n";
            proofRule += String( "cr" ) + _queryId + "_" + std::to_string( lemId ) + " rt" +
                         _queryId + "_" + std::to_string( id ) + "))\n";
        }
        else
        {
            const PiecewiseLinearConstraint *relu =
                _cdclCore->getConstraintFromLit( propagatedLit );
            unsigned identifierInt = relu->getTableauAuxVars().front();
            String tableauEq = "e" + std::to_string( identifierInt - ( _n - _m ) );
            String tableauLit = convertTableauAssumptionToClause( identifierInt - ( _n - _m ) );
            String counterpartAux = std::to_string( relu->getTableauAuxVars().front() );
            String lemmaBound = getBoundAsClause(
                Tightening( causing, entry->lemma->getMinTargetBound(), Tightening::LB ) );

            preRule = String( "(step rt" ) + _queryId + "_" + std::to_string( id ) + " (cl " +
                      tableauLit + "(not " + lemmaBound + ")(not a" + constraintId + ")(not (<= x" +
                      std::to_string( affected ) + " 0.0 ))(not (>= x" + counterpartAux +
                      " 0.0))):rule la_generic :args(-1 1 1 1 1))\n";

            proofRule += String( " rt" ) + _queryId + "_" + std::to_string( id ) + " " + tableauEq +
                         " cr" + _queryId + "_" + std::to_string( lemId ) + " rl" + _queryId + "_" +
                         std::to_string( lemId ) + " l" + counterpartAux + "))\n";
        }
    }

    _proof.append( { preRule, proofRule } );
}

String AletheProofWriter::clauseToPhases( const std::vector<int> &clause )
{
    String clauseString = "";
    for ( int lit : clause )
    {
        String isActive = -lit > 0 ? "a" : "(not a";
        String plcNum = std::to_string( abs( lit ) );
        clauseString += isActive + plcNum + ( -lit > 0 ? " " : ")" );
    }
    return clauseString;
}

bool AletheProofWriter::hasInfo() const
{
    return !_lastExplainedEntries.empty() || !_lastContradiction.empty();
}

const Set<int> &AletheProofWriter::getLastContradictionClause() const
{
    return _lastContradictionClause;
}

bool AletheProofWriter::lemmaExistsAsReasonClause( int64_t id ) const
{
    return _satIdToCdclVar.exists( id );
}

void AletheProofWriter::writeDerivedClauseContent( int64_t id,
                                                   const std::vector<int> &clause,
                                                   const std::vector<int64_t> &antecedents )
{
    std::vector<int> extendedClause( clause );

    // Counting begins at 1
    for ( int lit : _cdclCore->getSncLits() )
        extendedClause.insert( extendedClause.end(), -lit );

    String splitsClause = "";

    for ( int lit : extendedClause )
    {
        const PiecewiseLinearConstraint *plc = _cdclCore->getConstraintFromLit( lit );
        String isActive = lit > 0 ? "a" : "(not a";
        int constraintInt = plc->getVariableForDecision();
        splitsClause +=
            String( " " ) + isActive + std::to_string( constraintInt ) + ( lit > 0 ? "" : ")" );
    }

    String resLine = String( "(step r" ) + _queryId + "_" + std::to_string( id ) + " (cl" +
                     splitsClause + "):rule resolution :premises(";

    for ( int64_t step : antecedents )
        resLine += String( " r" ) + _queryId + "_" + std::to_string( step );

    resLine += "))\n";

    _proof.append( resLine );
}

void AletheProofWriter::setInitialTableau( const SparseMatrix *tableau )
{
    _initialTableau = tableau;
}


Vector<int> AletheProofWriter::resolution( const Vector<int> &c1, const Vector<int> &c2 )
{
    int resolutionLiteral = 0;
    for ( int lit : c1 )
        if ( c2.exists( -lit ) )
        {
            resolutionLiteral = lit;
            break;
        }

    Set<int> resolutionClause;
    for ( int lit : c1 )
        if ( lit != resolutionLiteral )
            resolutionClause.insert( lit );

    for ( int lit : c2 )
        if ( lit != -resolutionLiteral )
            resolutionClause.insert( lit );

    return { resolutionClause.begin(), resolutionClause.end() };
}

void AletheProofWriter::writeFinalStepsToProof()
{
    // TODO: expand this behavior for SNC depth > 2, where number of UNSAT jobs may not be a
    //  power of 2
    if ( AletheProofWriter::unsatJobFinalSteps.size() == 1 )
    {
        return;
    }

    AletheProofWriter::proofFile.open( File::MODE_WRITE_APPEND );

    List<Pair<String, Vector<int>>> finalClauses;
    for ( unsigned i = 1; i < AletheProofWriter::unsatJobFinalSteps.size() + 1; ++i )
        finalClauses.append( AletheProofWriter::unsatJobFinalSteps["1-" + std::to_string( i )] );

    unsigned index = 0;
    while ( finalClauses.size() > 1 )
    {
        ASSERT( finalClauses.size() % 2 == 0 );
        List<Pair<String, Vector<int>>> newFinalClauses;
        unsigned numClauses = finalClauses.size();

        for ( unsigned i = 0; i < numClauses / 2; ++i )
        {
            Pair<String, Vector<int>> step1 = finalClauses.popFront();
            Pair<String, Vector<int>> step2 = finalClauses.popFront();

            Vector<int> resolutionClause = resolution( step1.second(), step2.second() );
            String resolutionStepId = "f" + std::to_string( ++index );
            newFinalClauses.append(
                Pair<String, Vector<int>>( resolutionStepId, resolutionClause ) );

            AletheProofWriter::proofFile.write( String( "(step " ) + resolutionStepId + " (cl " +
                                                clauseToPhases( resolutionClause.getContainer() ) +
                                                "):rule resolution "
                                                ":premises (" +
                                                step1.first() + " " + step2.first() + "))\n" );
        }

        finalClauses = newFinalClauses;
    }

    AletheProofWriter::proofFile.close();
}

void AletheProofWriter::initializeProofFile( const String &filename )
{
    AletheProofWriter::proofFilename = filename;
    AletheProofWriter::proofFile = File( filename );
}

void AletheProofWriter::writeSncLitTrivialClause( int64_t id, unsigned sncVar )
{
    String trivial = String( "(step _r" ) + _queryId + "_" + std::to_string( id ) +
                     " (cl (or (not a" + std::to_string( sncVar ) + ") a" +
                     std::to_string( sncVar ) + ")):rule la_tautology)\n";
    trivial += String( "(step r" ) + _queryId + "_" + std::to_string( id ) + " (cl (not a" +
               std::to_string( sncVar ) + ") a" + std::to_string( sncVar ) +
               "):rule or :premises(_r" + _queryId + "_" + std::to_string( id ) + "))\n";
    _proof.append( trivial );
}

#endif
