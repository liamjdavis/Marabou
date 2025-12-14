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

AletheProofWriter::AletheProofWriter( unsigned explanationSize,
                                      const Vector<double> &upperBounds,
                                      const Vector<double> &lowerBounds,
                                      const GroundBoundManager &groundBoundManager,
                                      const SparseMatrix *tableau,
                                      const List<PiecewiseLinearConstraint *> &problemConstraints )
    : _initialTableau( tableau )
    , _groundBoundManager( groundBoundManager )
    , _plc( problemConstraints.begin(), problemConstraints.end() )
    , _n( upperBounds.size() )
    , _m( explanationSize )
    , _stepCounter( 1 )
    , _varToPlc()
    , _idToSplits()
    , _nodeToSplits()
{
    for ( unsigned i = 0; i < _n; ++i )
    {
        _currentLowerBounds.append( Stack<std::tuple<int, double>>() );
        _currentUpperBounds.append( Stack<std::tuple<int, double>>() );

        _currentLowerBounds[i].push( std::tuple<int, double>( 0, lowerBounds[i] ) );
        _currentUpperBounds[i].push( std::tuple<int, double>( 0, upperBounds[i] ) );
    }

    for ( const auto &plc : problemConstraints )
    {
        for ( const auto var : plc->getParticipatingVariables() )
            _varToPlc.insert( var, plc );

        _varToPlc.insert( plc->getTableauAuxVars().front(), plc );
    }

    // Write only necessary lines upon initialization
    writeTableauAssumptions();
}

void AletheProofWriter::writeTableauAssumptions()
{
    ASSERT( _assumptions.empty() );
    Vector<double> ubs;
    Vector<double> lbs;
    insertCurrentBoundsToVec( true, ubs );
    insertCurrentBoundsToVec( false, lbs );

    List<String> smtLib = SmtLibWriter::convertToSmtLib(
        _m,
        _n,
        ubs,
        lbs,
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
        ASSERT( _currentUpperBounds[i].size() == 1 );

        String s = std::to_string( i );
        String upper = String( "(assume u" ) + s + "(!(<= x" + s + " " +
                       SmtLibWriter::signedValue( std::get<1>( _currentUpperBounds[i].top() ) ) +
                       "):named u" + s + "))\n";
        String lower = String( "(assume l" ) + s + "(!(>= x" + s + " " +
                       SmtLibWriter::signedValue( std::get<1>( _currentLowerBounds[i].top() ) ) +
                       "):named l" + s + "))\n";
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
        if ( plc->getType() == RELU )
            splitsInFixedOrder.append( { plc->getCaseSplit( RELU_PHASE_ACTIVE ),
                                         plc->getCaseSplit( RELU_PHASE_INACTIVE ) } );


        String constraintNum = std::to_string( plc->getTableauAuxVars().front() );
        String plcAssumption =
            String( "(assume p" ) + constraintNum + " (or (!" +
            getSplitAsClause( splitsInFixedOrder.front() ) + ":named a" + constraintNum + ")(!" +
            getSplitAsClause( splitsInFixedOrder.back() ) + ":named i" + constraintNum + ")))\n";

        plcAssumption +=
            String( "(assume eq_a" ) + constraintNum +
            " (= " + getBoundAsClause( splitsInFixedOrder.front().getBoundTightenings().front() ) +
            getBoundAsClause( splitsInFixedOrder.front().getBoundTightenings().back() ) + "))\n";
        plcAssumption +=
            String( "(assume eq_i" ) + constraintNum +
            " (= " + getBoundAsClause( splitsInFixedOrder.back().getBoundTightenings().front() ) +
            getBoundAsClause( splitsInFixedOrder.back().getBoundTightenings().back() ) + "))\n";


        plcAssumptions.append( plcAssumption );

        String plcSplit = String( "(step s" ) + constraintNum + " (cl a" + constraintNum + " i" +
                          constraintNum + "):rule or :premises(p" + constraintNum + "))\n";
        plcSplits.append( plcSplit );

        for ( const auto &split : splitsInFixedOrder )
        {
            unsigned innerCounter = 0;
            String isActive = isSplitActive( split ) ? "a" : "i";
            for ( const auto &bound : split.getBoundTightenings() )
            {
                String line = String( "(step s" ) + constraintNum + "_" + isActive +
                              std::to_string( innerCounter ) + " (cl (not " + isActive +
                              constraintNum + ")" + getBoundAsClause( bound ) +
                              "):rule and_pos :args(" + std::to_string( innerCounter ) + "))\n";
                plcSplits.append( line );

                ++innerCounter;
            }

            if ( plc->getType() == RELU )
            {
                String line1 = String( "(step eq" ) + constraintNum + "_" + isActive +
                             +"0 (cl (not " +
                               getBoundAsClause( split.getBoundTightenings().front() ) + ")" +
                               getBoundAsClause( split.getBoundTightenings().back() ) +
                               "):rule equiv1 :premises(eq_" + isActive + constraintNum + "))\n";

                String line2 = String( "(step eq" ) + constraintNum + "_" + isActive + +"1 (cl " +
                               getBoundAsClause( split.getBoundTightenings().front() ) + "(not " +
                               getBoundAsClause( split.getBoundTightenings().back() ) +
                               " )):rule equiv2 :premises(eq_" + isActive + constraintNum + "))\n";
                plcSplits.append( line1 );
                plcSplits.append( line2 );
            }
        }
    }

    _assumptions.append( plcAssumptions );
    _assumptions.append( plcSplits );
}

void AletheProofWriter::writeContradiction( const SparseUnsortedList &contradiction,
                                            unsigned nodeId )
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
                   -(int)nodeId,
                   true );

    farkasClause += ")";
    farkasArgs += "))\n";

    String laGeneric = String( "(step t" + std::to_string( nodeId ) ) + " " + farkasClause +
                       ":rule la_generic :args" + farkasArgs;

    String res = String( "(step r" + std::to_string( nodeId ) ) + " (cl " + negatedSplitsClause +
                 "):rule resolution :premises(t" + std::to_string( nodeId ) + " " +
                 farkasParticipants + "))\n";

    _proof.append( { laGeneric, res } );
}

void AletheProofWriter::writeInstanceToFile( IFile &file )
{
    file.open( File::MODE_WRITE_TRUNCATE );
    writeBoundAssumptions();
    writePLCAssumption();
    for ( const String &s : _assumptions )
        file.write( s );

    for ( const String &s : _proof )
        file.write( s );

    file.close();
}

void AletheProofWriter::insertCurrentBoundsToVec( bool isUpper, Vector<double> &boundsVec )
{
    Vector<Stack<std::tuple<int, double>>> &temp =
        isUpper ? _currentUpperBounds : _currentLowerBounds;

    boundsVec = Vector<double>( _n, 0 );
    for ( unsigned i = 0; i < _n; ++i )
        boundsVec[i] = std::get<1>( temp[i].top() );
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
    String resLine = String( "(step r" + std::to_string( node->getId() ) + " (cl " ) +
                     getNegatedSplitsClause( splitDeps ) + "):rule resolution :premises(s" +
                     std::to_string( node->getChildren().front()->getSplitNum() ) + " r" +
                     std::to_string( childrenIndices.front() ) + " r" +
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
        String isActive = isSplitActive( split ) ? "a" : "i";
        String plcNum = std::to_string(
            _varToPlc[split.getBoundTightenings().front()._variable]->getTableauAuxVars().front() );
        clause += String( "(not " ) + isActive + plcNum + ")";
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

String AletheProofWriter::getSplitsResSteps( const List<PiecewiseLinearCaseSplit> &splits ) const
{
    if ( splits.empty() )
        return "";

    String resSteps = "";
    for ( const auto &split : splits )
    {
        unsigned splitNum =
            _varToPlc[split.getBoundTightenings().front()._variable]->getTableauAuxVars().front();
        String currentStep = " s" + std::to_string( splitNum );
        String isActive = isSplitActive( split ) ? "a" : "i";

        for ( unsigned i = 0; i < split.getBoundTightenings().size(); ++i )
            currentStep +=
                String( " s" + std::to_string( splitNum ) + "_" ) + isActive + std::to_string( i );
        // Add steps to the left
        resSteps = currentStep + resSteps;
    }

    return resSteps;
}
void AletheProofWriter::writeLemma(
    const std::shared_ptr<GroundBoundManager::GroundBoundEntry> &lemmaEntry )
{
    if ( !lemmaEntry->lemma || !lemmaEntry->lemma->getToCheck() )
        return;

    PiecewiseLinearConstraint *matchedConstraint = _varToPlc[lemmaEntry->lemma->getAffectedVar()];

    // TODO add support for all types of PLCs
    if ( matchedConstraint && matchedConstraint->getType() == RELU )
        writeReluLemma( lemmaEntry, (ReluConstraint *)matchedConstraint );
}

void AletheProofWriter::writeReluLemma(
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
    String negatedSplitClause = "";
    String causeBound = getBoundAsClause( Tightening( causingVar, targetBound, causingVarBound ) );

    farkasStrings( explanations.front(),
                   lemmaEntry->id,
                   farkasArgs,
                   farkasClause,
                   farkasParticipants,
                   negatedSplitClause,
                   causingVar,
                   causingVarBound == Tightening::UB );

    farkasClause += causeBound;

    farkasArgs += "1";
    farkasClause += ")";
    farkasArgs += "))\n";

    String laGeneric =
        String( "(step fl" ) + id + " " + farkasClause + ":rule la_generic :args" + farkasArgs;

    String res = String( "(step cr" ) + id + " (cl " + negatedSplitClause + causeBound +
                 "):rule resolution :premises(fl" + id + " " + farkasParticipants + "))\n";

    unsigned b = relu->getB();
    unsigned f = relu->getF();
    unsigned aux = relu->getAux();
    String identifier = std::to_string( relu->getTableauAuxVars().front() );
    bool matched = false;

    String proofRule = "";
    String proofRuleRes = "";
    String tempString = "";

    if ( targetBound > 0 )
        tempString += String( "(not " ) +
                      getBoundAsClause( Tightening( causingVar, 0, Tightening::UB ) ) + ")(not " +
                      causeBound + ")";
    else if ( targetBound < 0 )
        tempString += String( "(not" ) + causeBound + ")(not " +
                      getBoundAsClause( Tightening( causingVar, 0, Tightening::LB ) ) + ")";

    if ( targetBound != 0 )
    {
        proofRule =
            String( "(step taut" ) + id + " (cl (or " + tempString + ")):rule la_tautology)\n";
        proofRule += String( "(step ts" ) + id + " (cl " + tempString + "):rule or :premises(taut" +
                     id + "))\n";
    }
    String conclusion = getBoundAsClause( Tightening( affectedVar, bound, affectedVarBound ) );
    String pref = String( "(step rl" ) + id + " (cl " + negatedSplitClause + conclusion +
                  "):rule resolution :premises(cr" + id;
    if ( ( causingVar == f || causingVar == b ) && causingVarBound == Tightening::LB &&
         affectedVar == aux && affectedVarBound == Tightening::UB && targetBound > 0 )
    {
        matched = true;
        String splitrule = causingVar == f ? "_i1" : "_i0";
        proofRuleRes = pref + " ts" + id + " s" + identifier + splitrule + " s" + identifier +
                       " s" + identifier + "_a1))\n";
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

        proofRuleRes = pref + " ts" + id + " s" + identifier + "_a1 s" + identifier + " s" +
                       identifier + "_i1 ))\n";
    }

    // If ub of b is non positive, then ub of f is 0
    else if ( causingVar == b && causingVarBound == Tightening::UB && affectedVar == f &&
              affectedVarBound == Tightening::UB && targetBound < 0 )
    {
        matched = true;

        proofRuleRes = pref + " ts" + id + " s" + identifier + "_a0 s" + identifier + " s" +
                       identifier + "_i1 ))\n";
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
              affectedVarBound == Tightening::UB && FloatUtils::isNegative( targetBound ) )
    {
        matched = true;

        unsigned identifierInt = relu->getTableauAuxVars().front();
        String tautClause = String( "(not " ) +
                            getBoundAsClause( Tightening( affectedVar, 0, Tightening::UB ) ) + ")" +
                            conclusion;
        proofRule =
            String( "(step taut" ) + id + " (cl (or " + tautClause + ")):rule la_tautology)\n";

        proofRule += String( "(step ts" ) + id + " (cl " + tautClause + "):rule or :premises(taut" +
                     id + "))\n";

        String counterpartBound =
            getBoundAsClause( Tightening( identifierInt, 0, Tightening::LB ) );
        String subConclusion = getBoundAsClause( Tightening( f, 0, Tightening::UB ) );
        String tableauClause = convertTableauAssumptionToClause( identifierInt - ( _n - _m ) );
        String subFarkasClause = String( "(cl (not " ) + causeBound + ")" + conclusion + "(not " +
                                 subConclusion + ")(not " + counterpartBound + ")" + tableauClause;

        proofRule += String( "(step ifl" ) + id + " " + subFarkasClause +
                     "):rule la_generic :args(1 1 -1 1 -1))\n";

        proofRuleRes += pref + +" e" + std::to_string( identifierInt - ( _n - _m ) ) + " l" +
                        identifier + " ifl" + id + " ts" + id + " s" + identifier + "_a1 s" +
                        identifier + " s" + identifier + "_i1))\n";
    }
    // If ub of b is positive, then propagate to f
    else if ( causingVar == b && causingVarBound == Tightening::UB && affectedVar == f &&
              affectedVarBound == Tightening::UB && FloatUtils::isPositive( targetBound ) )
    {
        matched = true;

        unsigned identifierInt = relu->getTableauAuxVars().front();
        String tautClause = String( "(not " ) +
                            getBoundAsClause( Tightening( affectedVar, 0, Tightening::UB ) ) + ")" +
                            conclusion;

        proofRule =
            String( "(step taut" ) + id + " (cl (or " + tautClause + ")):rule la_tautology)\n";

        proofRule += String( "(step ts" ) + id + " (cl " + tautClause + "):rule or :premises(taut" +
                     id + "))\n";

        String counterpartBound =
            getBoundAsClause( Tightening( identifierInt, 0, Tightening::UB ) );
        String subConclusion = getBoundAsClause( Tightening( aux, 0, Tightening::UB ) );
        String tableauClause = convertTableauAssumptionToClause( identifierInt - ( _n - _m ) );
        String subFarkasClause = String( "(cl (not " ) + causeBound + ")" + conclusion + "(not " +
                                 subConclusion + ")(not " + counterpartBound + ")" + tableauClause;

        proofRule += String( "(step ifl" ) + id + " " + subFarkasClause +
                     "):rule la_generic :args(1 1 -1 1 1))\n";

        proofRuleRes += pref + " e" + std::to_string( identifierInt - ( _n - _m ) ) + " u" +
                        identifier + " ifl" + id + " ts" + id + " s" + identifier + "_a1 s" +
                        identifier + " s" + identifier + "_i1))\n";
    }

    if ( matched )
        _proof.append( { laGeneric, res, proofRule, proofRuleRes } );
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
                                       bool isUpper )
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

    farkasClause = "(cl ";
    farkasArgs = "( ";
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

        int lemId = gbEntry->lemma ? gbEntry->lemma->getId() : -1;
        double bound = gbEntry->val;
        bool isLemmaIncluded = lemId >= 0 && gbEntry->lemma->getToCheck();
        bool useSplitBound = ( lemId < 0 && gbEntry->isPhaseFixing );

        farkasArgs += temp.get_str() + " ";

        if ( isLemmaIncluded || useSplitBound )
            farkasClause += String( "(not (" ) + ineqString + " x" + std::to_string( i ) +
                            String( " " ) + SmtLibWriter::signedValue( bound ) + ")) ";
        else
        {
            farkasClause += String( "(not " ) + boundString + std::to_string( i ) + ") ";
            farkasParticipants += boundString + std::to_string( i ) + " ";
        }

        // Add split deps of prev lemmas
        if ( isLemmaIncluded )
        {
            farkasParticipants += String( "rl" + std::to_string( lemId ) ) + " ";
            for ( const auto dep : _idToSplits[gbEntry->id] )
                if ( !splitDeps.exists( dep ) )
                    splitDeps.append( dep );
        }
        else if ( useSplitBound )
            splitDeps.append( Tightening( i, bound, boundType ) );
    }

    for ( const auto num : explainedRow )
        mpq_clear( num );

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
        String identifier = std::to_string( plc->getTableauAuxVars().front() );

        PiecewiseLinearCaseSplit tighteningSplit;

        for ( const auto &casSplit : plc->getAllCases() )
        {
            PiecewiseLinearCaseSplit split = plc->getCaseSplit( casSplit );
            if ( split.getBoundTightenings().exists( tightening ) )
                tighteningSplit = split;
        }

        String isActive = isSplitActive( tighteningSplit ) ? "a" : "i";
        negatedSplitClause += String( "(not " ) + isActive + identifier + ")";
        farkasParticipants += String( "s" ) + identifier + "_" + isActive + "0 ";
        farkasParticipants += String( "s" ) + identifier + "_" + isActive + "1 ";
    }
}

String AletheProofWriter::convertTableauAssumptionToClause( unsigned index ) const
{
    return String( "(not e" ) + std::to_string( index ) + ")";
}

void AletheProofWriter::writeDelegatedLeaf( const UnsatCertificateNode *node )
{
    String proofHole = String( "(step r" + std::to_string( node->getId() ) ) + " (cl " +
                       getNegatedSplitsClause( getPathSplits( node ) ) + "):rule hole)\n";
    _proof.append( proofHole );
}

unsigned AletheProofWriter::assignId()
{
    return _stepCounter++;
}
