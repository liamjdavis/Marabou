# PR Clause Learning over PICID Conflict Clauses

A two-pass driver on top of PICID's CDCL(T) pipeline (Marabou + CaDiCaL via
IPASIR-UP). **Phase A** runs the CDCL search up to a bounded decision depth,
harvesting propagation-redundancy (PR) clauses from conditional-autarky
carves of the learned clause pool; **Phase B** restarts the solver on the
original query with the harvested clauses injected into CaDiCaL. This is the
Marabou/PICID port of the same strategy implemented for α,β-CROWN + BICCOS
(see `Verifier_Development/PR-CLAUSES.md`), which is in turn the
neural-verification analog of CAUTICAL's SAT-solver preprocessing
([Shah et al., FMCAD 2025](https://repositum.tuwien.at/bitstream/20.500.12708/219546/1/Shah%20Amar%20-%202025%20-%20Learning%20Short%20Clauses%20via%20Conditional%20Autarkies.pdf)).

PICID itself is described in [Isac, Refaeli, Wu, Barrett, Katz — arXiv:2503.12083](https://arxiv.org/abs/2503.12083).

---

## 1. Motivation

PICID's conflict clauses — whether proof-based (`ANALYZE_PROOF_DEPENDENCIES`)
or decision-based — are *entailed* facts: combinations of ReLU phases proven
empty, never to be revisited. **PR clauses** make a stronger move: they delete
a region that *might* still contain a counterexample, on the grounds that any
counterexample there has an equivalent one elsewhere that survives. The engine
for finding sound PR moves is the **conditional autarky**: once some phases
are pinned (the condition αc), the remaining fixed phases (the autarky αa)
satisfy every learned clause they touch, so each may as well be fixed the
canonical way.

The port is *simpler* than the α,β-CROWN version: PICID's pool consists of
genuine SAT clauses over CaDiCaL literals, so the coefficient arithmetic of
the BICCOS-cut predicates collapses into set membership, and injection needs
no translation layer at all.

---

## 2. The Algorithm

### 2.1 Predicates

Let the **trail** α be the set of currently assigned boolean-abstraction
literals (branch decisions + theory propagations, excluding root-fixed vars),
and let the **pool** 𝓛 be the clauses learned so far (conflict clauses via
`addExternalClause`, plus any initial NAP clauses).

```
touches(α, ℓ)   ≜  some variable of ℓ is assigned in α
satisfies(α, ℓ) ≜  some literal of ℓ is assigned true in α
is_open(α, ℓ)   ≜  touches(α, ℓ) ∧ ¬satisfies(α, ℓ)
```

Because pool clauses are pure disjunctions (±1 "coefficients"), the tight
worst-case check from the α,β-CROWN version reduces exactly to `satisfies`
above: a clause with any true literal survives every extension; a clause with
only false/free literals does not.

### 2.2 Conditional-autarky carve (lemma-first)

```
αc := ∅
for ℓ in 𝓛:
    if is_open(α, ℓ):
        αc := αc ∪ (α-literals over vars(ℓ))
αa := α \ αc
```

### 2.3 PR clause readout

```
PR-clause(a) := (¬c₁ ∨ … ∨ ¬cₖ ∨ ¬a)        for each a ∈ αa
```

One clause per autarky literal; conditions deduplicated and
subsumption-minimized per autarky bucket. **All** surviving clauses are
injected — no top-K cap (CaDiCaL handles the volume).

### 2.4 Soundness status — read this

This is deliberately the **raw, unchecked** variant, to measure what happens.
Two gaps versus the SAT setting, where CAUTICAL's Theorem 1 makes PR clauses
sound:

1. **Pool-only carve.** The autarky is checked against the learned clause
   pool, not the full formula. In CAUTICAL, Algorithm 2 iterates over the
   *entire* CNF — everything that defines solution-hood.
2. **Realizability.** Boolean assignments here are shadows of theory models.
   The autarky flip preserves satisfaction of every pool *clause*, but the
   flipped phase pattern may have no realizing input x. The theory is an
   invisible constraint set the carve never inspects.

Consequently a PR clause can, in principle, prune the region containing the
only real counterexample. SAT verdicts stay sound (`cb_check_found_model`
theory-checks every model), but an **UNSAT verdict under PR preprocessing is
not certified** — unlike α,β-CROWN, PICID derives UNSAT from the search
itself, so there is no verdict-asymmetry escape hatch. Additionally, batch
injection of clauses harvested from *different* trails is not covered by a
single application of CAUTICAL's Theorem 1 (their Algorithm 1 re-carves
against Γ including previously added PR clauses; we, like the α,β-CROWN
implementation, inject one batch).

Sound completions (not implemented, by choice): (i) learn-time entailment
filter via bound propagation on the pruned region; (ii) post-hoc discharge —
after an UNSAT, use PICID's proof-dependency analysis to find the PR clauses
the proof leans on and solve each pruned region as a case-split subquery
(UNSAT ⇒ clause was entailed, proof splices in; SAT ⇒ genuine counterexample,
answer flips to SAT); (iii) emit PR clauses as Alethe `hole` steps. Note (ii)
is *decisive* in the DNN setting — something pure SAT does not enjoy.

Alethe proof output is disabled in this build (`WRITE_ALETHE_PROOF = false`):
this line of work is not about proof generation, and the writer also crashes
in `SmtLibWriter::convertToSmtLib` on our setup. Marabou-internal proof
production stays on — `--cdcl` requires it for proof-based conflict clauses.

---

## 3. Implementation Map

```
src/cdcl/
    PrClauseLearner.h/.cpp     # pool mirror, carve, dedup, harvest
    tests/Test_PrClauseLearner.h   # 7 unit tests (cxxtest)
    CdclCore.h/.cpp            # hooks + two-pass driver (see below)
src/configuration/
    Options.h/.cpp, OptionParser.cpp   # flags
    GlobalConfiguration.cpp    # WRITE_ALETHE_PROOF = false
src/engine/Engine.cpp          # solveWithCDCL dispatch
```

### 3.1 Data contracts

- **Literal**: CaDiCaL int; `CdclCore::_satSolverVarToPlc` maps var →
  `PiecewiseLinearConstraint*`, sign = phase. No layer/neuron indexing needed.
- **Trail** (α): iteration over `_assignedLiterals` (context-dependent
  CDHashMap), skipping literals fixed at root (`isLiteralFixed`).
- **Pool clause**: `Set<int>` in disjunction form. `addExternalClause`
  receives conflicting-assignment *cubes*, so the mirror negates literals
  (`addPoolClauseFromCube`); NAP file clauses arrive already in clause form.

### 3.2 Hooks in CdclCore

- `addExternalClause()` — mirrors each learned cube into the learner pool
  (harvest mode only). The mirror is separate from `_literalToClauses`, which
  is periodically cleared for VSIDS decay and stores no clause bodies.
- `notify_new_decision_level()` — builds the trail, calls
  `PrClauseLearner::observeTrail`, and requests Phase A termination once
  `getLevel() > depth limit`.
- `terminate()` / `cb_decide()` / `cb_check_found_model()` — honor the stop
  request so CaDiCaL aborts promptly without further theory solves.

### 3.3 Two-pass driver (`CdclCore::solveWithPrPreprocessedCDCL`)

1. Snapshot root-level `_literalsToPropagate` (theory-fixed phases from
   preprocessing) for replay.
2. **Phase A**: `solveWithCDCL` with harvesting on. If it concludes
   (SAT/UNSAT/timeout) within the depth limit, return that verdict directly.
3. On depth-triggered abort: `finalizeHarvest()` (dedup + subsumption), grab
   the pool as carry (Phase A conflict clauses — entailed, sound to reuse).
4. Restart: `_shouldRestart = true; notify_backtrack(0)` (restores initial
   engine state, pops context to root) then `reset()` (fresh CaDiCaL with
   re-registered observed vars).
5. Replay root propagations; `addClause()` every carry clause and every PR
   clause into the fresh solver.
6. **Phase B**: `solveWithCDCL` to completion.

`Engine::solveWithCDCL` dispatches to the driver when
`Options::PR_CLAUSE_PREPROCESS` is set.

---

## 4. Configuration

| Flag                           | Default | Purpose                                            |
|--------------------------------|---------|----------------------------------------------------|
| `--pr-clause-preprocess`       | off     | Enable the two-pass harvest → inject driver.       |
| `--pr-clause-preprocess-depth` | `5`     | Decision level at which Phase A stops.             |

Both require `--cdcl`. Use `--lp-solver native`: the Gurobi LP path of this
branch segfaults in `Tableau::setNonBasicAssignment` during CDCL solving
(pre-existing; same crash site as the branch's `Test_VnnLibParser` failure).

---

## 5. How to Run

### 5.1 Build

```bash
mkdir -p build-picid && cd build-picid
cmake ../ -DENABLE_GUROBI=ON      # CaDiCaL auto-downloads; GMP path auto-detected
cmake --build . -j 8
```

### 5.2 Baseline (CDCL, no PR clauses)

```bash
./build-picid/Marabou resources/nnet/acasxu/ACASXU_experimental_v2a_1_1.nnet \
  resources/properties/acas_property_4.txt \
  --cdcl --lp-solver native --verbosity 1
```

### 5.3 With PR clauses (two-pass)

```bash
./build-picid/Marabou resources/nnet/acasxu/ACASXU_experimental_v2a_1_1.nnet \
  resources/properties/acas_property_4.txt \
  --cdcl --lp-solver native --pr-clause-preprocess \
  --pr-clause-preprocess-depth 5 --verbosity 1
```

Marker lines:

- `PR: phase A - harvesting up to decision level D`
- `PR: phase A done - observed N trails, harvested H candidates; carrying C
  learned clauses, injecting K PR clauses`

### 5.4 Tests

```bash
cd build-picid && ctest -R PrClauseLearner --output-on-failure
```

---

## 6. Future Work

- Conflict-budgeted (or pool-nonempty-gated) Phase A instead of pure depth:
  a conflict-free Phase A leaves the pool empty, the carve vacuous, and every
  PR clause a unit — pure phase fixing rather than conditional reasoning.
- The discharge pass of §2.4 — decisive soundness restoration, parallelizable
  over SNC workers, reusing `ANALYZE_PROOF_DEPENDENCIES`.
- Sequential re-carving against Γ ∪ {already-injected PR clauses}
  (CAUTICAL Algorithm 1 semantics) instead of batch injection.
- SNC mode: per-worker harvesting and injection.
- Benchmarking: ACASXu/MNIST sweep, verdict cross-checking against baseline
  CDCL, wall-clock and visited-states comparison.
