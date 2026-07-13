# PR Clause Learning over PICID Conflict Clauses

A two-pass driver on top of PICID's CDCL(T) pipeline (Marabou + CaDiCaL via
IPASIR-UP). **Phase A** runs the CDCL search until a bounded number of
conflict clauses has been learned (Marabou is a DFS solver, so a decision-depth
bound would trip almost immediately without any conflicts in the pool),
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
  (harvest mode only), and requests Phase A termination once the number of
  mirrored conflict clauses reaches the conflict budget. The mirror is
  separate from `_literalToClauses`, which is periodically cleared for VSIDS
  decay and stores no clause bodies.
- `notify_new_decision_level()` — builds the trail and calls
  `PrClauseLearner::observeTrail`.
- `terminate()` / `cb_decide()` / `cb_check_found_model()` — honor the stop
  request so CaDiCaL aborts promptly without further theory solves.

### 3.3 Two-pass driver (`Marabou::solveWithPrRebuild`)

Each phase runs on a **virgin engine**: the in-process engine restore
(`notify_backtrack(0)` + `reset()`) is corrupt — it deposits root-conflict
lemmas and validates models against a broken tableau (confirmed false SAT).
The driver serializes the query once (`pr_rebuild_query.ipq`) and reloads it
into a freshly constructed and processed `Engine` per phase; clauses flow
between engines via `CdclCore` statics (`prRebuildRole`, `prSeedClauses`,
`prHandoff*`).

1. **Phase A** (role HARVEST): `solveWithCDCL` with harvesting on. SAT/UNSAT
   within the conflict budget is returned directly (theory-checked / earned
   by search — sound). On budget-triggered abort: `finalizeHarvest()` (dedup
   + subsumption), hand off the PR clauses plus the conflict pool as carry
   (entailed, sound to reuse).
2. **Phase B** (role SOLVE): fresh engine; every carry clause and every PR
   clause seeded into its CaDiCaL at solve start; `solveWithCDCL` to
   completion. UNSAT here is **uncertified** (§2.4).

`Marabou::solveQuery` dispatches to the driver when
`Options::PR_CLAUSE_PREPROCESS` is set; `Engine::solveWithCDCL` routes each
engine into `CdclCore::solveWithPrPreprocessedCDCL`, which acts per role.

---

## 4. Configuration

| Flag                               | Default | Purpose                                          |
|------------------------------------|---------|--------------------------------------------------|
| `--pr-clause-preprocess`           | off     | Enable the two-pass harvest → inject driver.     |
| `--pr-clause-preprocess-conflicts` | `10`    | Learned-conflict count at which Phase A stops.   |

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
  --pr-clause-preprocess-conflicts 10 --verbosity 1
```

Marker lines:

- `PR: phase A - harvesting until N conflicts are learned`
- `PR: phase A done (t=Ts) - observed N trails, harvested H candidates;
  handing off C carry clauses and K PR clauses`
- `PR: seeded fresh core with M clauses`
- `PR: phase B unsat (PR clauses injected - uncertified)`

### 5.4 Tests

```bash
cd build-picid && ctest -R PrClauseLearner --output-on-failure
```

---

## 6. Future Work

- The discharge pass of §2.4 — decisive soundness restoration, parallelizable
  over SNC workers, reusing `ANALYZE_PROOF_DEPENDENCIES`.
- Sequential re-carving against Γ ∪ {already-injected PR clauses}
  (CAUTICAL Algorithm 1 semantics) instead of batch injection.
- SNC mode: per-worker harvesting and injection.
- Benchmarking: ACASXu/MNIST sweep, verdict cross-checking against baseline
  CDCL, wall-clock and visited-states comparison.

---

## 7. Sound Accounting Port Plan

> **Status (2026-07-12):** implemented in the **batched** form matching the
> α,β-CROWN pipeline, replacing an earlier search-based variant that was
> measured and removed (one full CDCL solve per debt cube cost more than
> the entire baseline). Phase C now runs after every phase-B UNSAT: one
> debt cube per injected clause, deduped; propositional discharge by unit
> propagation against the entailed carry pool; each survivor gets ONE bound
> propagation pass (cube phases pinned as case splits on a fresh engine,
> `CdclCore::dischargeDebtCubes`) — no search anywhere, the analog of
> α,β-CROWN's chunked debt bounding. All discharged ⇒ `unsat is SOUND`;
> otherwise the verdict stands but is reported NOT certified.
>
> Measured (ACASXU 1_2 × prop 1, budget 10): 0.1s for 144 cubes — and
> 0/144 discharged (0 propositional, matching α,β-CROWN's 0/512 support
> mismatch; 0 theory, because the harvested clauses are unit phase
> preferences whose cubes are half-spaces no bound pass can refute — the
> thin-cube/entailment-ceiling finding, again). Cheap, but on this clause
> population it certifies nothing; α,β-CROWN's batched pass pays its debt
> because BICCOS cubes carry 13–154 forced phases.

Port of the sound-accounting architecture validated on α,β-CROWN
(`Verifier_Development/PR-CLAUSES.md` §2.4/§3.3, commits `2ee5535`,
`510f061`, `4884c28`). Core idea, in the CDCL(T) vocabulary: a Phase-B
UNSAT obtained with injected PR clauses P over entailed pool Γ proves only
"no theory model whose boolean shadow satisfies Γ ∧ P." The **debt** is one
phase cube per clause — its falsifying assignment `c₁ ∧ … ∧ cₖ ∧ a` — and
once every debt cube is discharged (propositionally against Γ, or by a
theory subquery), the UNSAT verdict is retroactively sound. No PR theory is
needed anywhere in the argument; it is exact case accounting.

The stakes are *higher* here than in α,β-CROWN: PICID derives UNSAT from
the search itself (no verdict-asymmetry escape hatch), so Phase C is not an
optimization — it is what makes the pipeline's UNSAT answers meaningful at
all. SAT verdicts need no debt (`cb_check_found_model` theory-checks every
model). The DNN-decisiveness bonus stands: a debt subquery that comes back
SAT is a genuine counterexample and flips the overall answer to SAT.

### 7.0 Phase 0 — measure before building (lesson learned the hard way)

Instrumentation only, no driver changes:

1. Dump to JSON at Phase-B injection time: the carried pool (Phase-A
   conflict clauses), all injected PR clauses, and per-clause debt cubes.
2. Offline (python, mirror of `Verifier_Development` scratch analyzers):
   pool support vs cube support overlap; |condition| histogram; unit-clause
   fraction; propositional discharge rate (cube refuted ⟺ some pool clause
   has every literal falsified by the cube — pure set containment here, no
   coefficient arithmetic).
3. Expectation to test: PICID's pool is made of genuine multi-literal
   conflict clauses, so BOTH degeneracies seen in α,β-CROWN (pool collapses
   to a unit-cut cube; carve support disjoint from pool support) may simply
   not occur. If the discharge rate is high, most debt is free and the
   theory pass shrinks to a handful of subqueries. This number decides how
   much of §7.3 is worth building.

Also resolve, before any driver work, the **autarky sign question**: §2.3
emits `¬a` while CAUTICAL Theorem 1 adds the positive autarky literal
(αc → αa). The accounting is sign-agnostic (the debt cube is always the
clause's falsifying assignment) but the sign flips what Phase B prunes and
therefore how large the debt is. Decide once, document, and align both
codebases.

### 7.1 Sound credit and early stop (already mostly present)

- Phase A runs with entailed clauses only; §3.3 step 2 already returns its
  verdict directly when it concludes within budget. Keep — this is the
  α,β-CROWN early-stop, and over there it alone beat the honest baseline.
- Provenance rule 1: only Phase-A conflict clauses enter the accounting
  pool Γ. Clauses learned during Phase B are resolvents over Γ ∪ P — they
  are entailed by Γ ∧ P, not by the theory alone, and must never be used to
  discharge debt or seed subqueries.
- Provenance rule 2: audit **NAP file clauses**. If NAP clauses are
  assumptions/heuristics rather than facts entailed by this query, they
  contaminate both the carve's Γ and the discharge pool. Either prove they
  are entailed, or track them separately: carve over Γ ∪ NAP is fine
  (unsoundness is already being accounted), but *discharge* must check
  cubes against entailed clauses only.
- Provenance rule 3 (the cutter-leak analog): audit every piece of state
  that survives `notify_backtrack(0)` + `reset()` between phases and
  between debt subqueries — engine bound tightenings, `_literalsToPropagate`
  replay contents, learned-clause carryover inside CaDiCaL. The α,β-CROWN
  port lost two days to a stale-pool leak that silently fed Phase-B PR cuts
  into what claimed to be entailed-only runs; assume Marabou has an
  equivalent until proven otherwise.

### 7.2 Debt extraction and propositional discharge

New module (suggested `src/cdcl/PrDebtLedger.h/.cpp`):

- On Phase-B UNSAT, collect debt cubes from the injected clauses. Two
  scope-reduction filters, in order:
  1. **Proof-dependency filter** (Marabou-only superpower — α,β-CROWN had
     no proof object): use `ANALYZE_PROOF_DEPENDENCIES` / CaDiCaL clause
     usage to find which PR clauses the final UNSAT proof actually leans
     on. Unused clauses incur **zero debt** — they could be deleted from
     the run post-hoc without changing the proof. Expected to be the
     biggest cost reducer.
  2. **Propositional discharge**: cube refuted by the entailed pool ⇒
     clause was entailed ⇒ zero debt. Set-containment check per
     clause/cube pair; optionally a single CaDiCaL solve of Γ ∧ cube under
     assumptions for completeness beyond unit propagation.
- Ordering refinement (disjoint cover): debt cube i may assume clauses
  C₁ … Cᵢ₋₁ in addition to Γ. Free constraint tightening; keeps the union
  of discharged regions a partition.

### 7.3 Theory discharge of surviving cubes

The α,β-CROWN "one batched GPU pass" has no direct analog; candidate
mechanisms in preference order:

1. **Incremental assumptions** (if the IPASIR-UP integration supports
   `assume`): one persistent solver holding Γ (Phase-A clauses only);
   per cube, assume its literals and solve. Clauses learned under
   assumptions are assumption-free resolvents — sound to keep across
   cubes, so later cubes get faster. Closest analog to the batched pass.
2. **SNC fan-out**: each cube is a natural split-and-conquer subquery
   (phases pre-fixed); parallel workers; reuses existing SNC plumbing.
3. **Sequential restarts** (simplest first cut): the existing restart
   machinery of §3.3 steps 4–5, once per cube, cube literals injected as
   unit clauses alongside the Γ carry.

Per-cube outcomes: UNSAT ⇒ discharged; SAT ⇒ **genuine counterexample,
overall answer is SAT** (decisive flip — report immediately); timeout ⇒
verdict stays `unknown` (never report UNSAT with unpaid debt).

### 7.4 Driver changes (`CdclCore::solveWithPrPreprocessedCDCL`)

- Phase B returns UNSAT → run ledger (7.2) → theory discharge (7.3) →
  only then report UNSAT; emit marker lines mirroring the α,β-CROWN ones
  (`PR-sound: N debt cubes, P proof-pruned, Q discharged propositionally,
  R theory-solved, verdict SOUND/…`).
- Flags: `--pr-sound-discharge` (on/off), `--pr-debt-mode
  {assume,snc,restart}`, `--pr-debt-timeout-per-cube`.

### 7.5 Validation plan

- Unit tests: ledger extraction (cube = falsifying assignment, both signs),
  set-containment discharge, proof-dependency filter on a synthetic pool.
- End-to-end: ACASXu instance where baseline CDCL is UNSAT — verify
  pipeline verdict matches with debt fully discharged; deliberately inject
  a bogus clause pruning a SAT region on a SAT instance and confirm the
  debt subquery flips the answer to SAT (the soundness canary the
  α,β-CROWN setting could not express).
- Scoreboard discipline from the α,β-CROWN port: leak-fixed baseline first,
  serialized runs, and never compare against numbers produced before the
  provenance audit (7.1 rule 3).
