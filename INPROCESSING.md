# Theory-Level Inprocessing: the Implication Skeleton

SAT-style inprocessing (failed-literal probing, binary-implication mining,
vivification) lifted to the DNN-verification theory level. Before the CDCL
search starts, every unfixed ReLU is probed in both phases with the full
theory toolbox; everything the probes prove is **entailed by the query** —
sound by construction, no wagers, no debt accounting — and is delivered once,
at the root, through trail-independent channels.

## Theory

For each unfixed ReLU `r` with pre-activation `b`, and each phase pin
(`b >= 0` active / `b <= 0` inactive):

1. push a context level and apply the pin as a case split;
2. tighten to fixpoint: direct DeepPoly pass + bound propagation + valid-split
   cascade (`Engine::probeTightenToFixpoint`);
3. run a budgeted simplex feasibility check
   (`Engine::probeLpFeasible`, pivot cap
   `GlobalConfiguration::SKELETON_PROBE_SIMPLEX_PIVOT_CAP = 400`);
4. snapshot the branch bound vectors, pop the context.

One probe pass feeds two channels:

- **C1 — propositional skeleton.** A refuted pin is a *failed literal* → unit
  clause. A feasible pin that fixes another ReLU's phase (detected by the
  union of the phase CDO and the raw sign of the other `b`'s bounds) → binary
  implication `pin → phase`. Units and binaries are seeded into CaDiCaL
  before the search (`CdclCore::solveWithCDCL`, `_initialClauses`).
- **C2 — hull fold.** Every real point satisfies one of the two phases, so
  the elementwise hull of the two branch bound vectors is valid at the root
  (if one phase was refuted, the surviving branch's bounds apply outright).
  The best hull across all pins is applied to the root tableau, followed by
  one tightening cascade.

Probing runs from `Engine::solveWithCDCL` before the CDCL loop
(`Engine::computeImplicationSkeleton`). ~500 probes take ~4s on ACAS-Xu.

### Soundness invariants

- **Probe-mode guards** (`CdclCore::setProbeMode`): during probes,
  `addLiteralToPropagate` and `addDecisionBasedConflictClause` are
  suppressed. Pin-conditioned facts must never leak to the SAT solver as
  root facts. Any new engine-side callback that feeds CaDiCaL must check
  `_probeMode`.
- **Failed-literal audit** (`SKELETON_VERIFY_UNITS`, below): failed literals
  come from budgeted LP infeasibility without precision-restoration guards;
  the audit re-proves each unit by solving `Q ∧ pin` on a fresh engine with a
  fresh tableau. All 5 units across the two calibration instances verified
  entailed (2026-07-14). The audit copies a pristine post-preprocessing query
  snapshot (`Engine::_skeletonAuditQuery`) — the live query cannot be deep
  copied (its constraints carry registered CDOs and bound-manager pointers).

## Running it

Build (CaDiCaL required; note `file(GLOB)` in CMakeLists — adding/removing
.cpp files requires a cmake re-run):

```bash
mkdir build && cd build && cmake ../ -DENABLE_GUROBI=ON && make -j
```

## Benchmarking (A/B)

Control and treated commands are **identical except for
`--implication-skeleton`** — same solver mode, LP backend, timeout,
verbosity, seed defaults:

```bash
# control (baseline CDCL)
timeout -s KILL 1000 ./Marabou \
    resources/onnx/acasxu/ACASXU_experimental_v2a_3_4.onnx \
    resources/properties/acas_property_1.txt \
    --cdcl --lp-solver native --timeout 900 --verbosity 1

# treated (baseline + inprocessing) — only the one flag differs
timeout -s KILL 1000 ./Marabou \
    resources/onnx/acasxu/ACASXU_experimental_v2a_3_4.onnx \
    resources/properties/acas_property_1.txt \
    --cdcl --lp-solver native --timeout 900 --verbosity 1 \
    --implication-skeleton
```

Metric: `Total visited states` from the **final** stats block (the probe
phase prints an early block with `visited states: 1` — always take the last
occurrence). Wall time is secondary (trajectory chaos makes it noisy).
Always wrap in `timeout -s KILL` with margin over `--timeout`; kill strays
with `pkill -9 -x Marabou` (never `-f`). Calibration anchors: 3_4×prop_1
control 6,062 / treated 3,934 (both unsat); 1_2×prop_1 control 1,982 /
treated 2,106 (both unsat).

CLI flags:

| flag | effect |
|---|---|
| `--implication-skeleton` | enable probing + both channels (requires `--cdcl`) |
| `--cdcl` | CDCL search (CaDiCaL-driven); the skeleton is CDCL-only |
| `--lp-solver native` | native simplex (the configuration all numbers below use) |

Environment knobs (diagnostics/ablation):

| env var | effect |
|---|---|
| `SKELETON_NO_SEED=1` | probe but seed nothing (isolates probe-time state perturbation from clause effects) |
| `SKELETON_UNITS_ONLY=1` | seed only the failed-literal units, not the binaries |
| `SKELETON_NO_HULL=1` | disable the C2 hull fold |
| `SKELETON_VERIFY_UNITS=<sec>` | after probing, audit each failed-literal unit on a fresh engine (timeout per unit, default 120s); prints `VERIFIED entailed` / `UNSOUND UNIT` / `unresolved` per unit |
| `SKELETON_NO_VIVIFY_BOUNDS=1` | disable rung-0 numeric vivification (see graveyard — measured inert, on by default) |
| `SKELETON_VIVIFY_DELTA_FRAC=<f>` | min probe-delta size as a fraction of the root gap to store for vivification (default 0.05) |
| `SKELETON_VIVIFY_MAX_SIZE=<n>` | max clause size for the numeric vivification test (default 64) |

At verbosity ≥ 1 you get a probe-progress line every 100 probes, a summary
line (`Implication skeleton (LP probes): ...`), and a `CDCL progress` heartbeat
every 30s (stats blocks only print at solve end). The heartbeat lives at the
`cb_propagate` theory-check site — `cb_check_found_model` never fires under
Marabou-driven decisions.

## Measured results (ACAS-Xu × prop 1, visited states)

Single instances are trajectory-chaos dominated (±2×); treat these as
calibration anchors, not conclusions. The controlling variable is skeleton
density: dense skeleton = real wins, thin skeleton = pure noise.

| instance | skeleton | baseline | C1 only | C1+C2 (full) |
|---|---|---|---|---|
| 3_4 (dense: 3 units, 141 binaries) | 744 hull bounds | 6,062 | 4,241 (−30%) | **3,934 (−35%)** |
| 1_2 (thin: 2 units, 31 binaries) | 736 hull bounds | 1,982 | 3,945 (hurt) | 2,106 (mild hurt) |

Rung-1 LP vivification on top of C1+C2 (2026-07-15, see "Rung-1" below):

| instance | C1+C2 | + rung-1 (32/visit) | + hybrid (256/visit + harvest) | descent hit rate |
|---|---|---|---|---|
| 3_4 | 3,934 | 3,501 (−11%) | **3,497** | 147/149 (99%) |
| 1_2 | 2,106 | 2,023 | **2,001** (control 1,982) | 134/151 (89%) |

On the thin instance each vivification increment claws back the skeleton's
loss toward the baseline; on the dense instance it extends the win. Learned
clauses are massively compressible (2,362 literals removed on 3_4 in 1.4s),
confirming the decision-based reason coarseness diagnosis.

With the mirror oracle (2026-07-16, see "The mirror oracle" below), the
unsat-proof channel terminates both anchors at their first level-0 visit:
3_4 **3,493**, 1_2 **1,999** — small absolute gains only because the first
luby restart lands near the end of these searches (see the cadence
bottleneck note).

## The mirror oracle (one extra CaDiCaL, 2026-07-16)

The consolidated boolean side: exactly two CaDiCaL instances total. The main
solver searches; the **mirror** holds a copy of every entailed clause
(skeleton seed, learned conflicts post-vivify-attempt, harvested edges,
vivified outputs, level-0 fixed literals synced per pass) and serves five
duties at level-0 visits (`CdclCore::processVivifyLpQueue` /
`mirrorProbePass`):

1. **Vivification by assumption core**: assume the negations of a clause's
   literals, conflict-bounded solve; UNSAT ⇒ the `failed()` core IS the
   shortened clause — full conflict analysis, zero theory cost. A clause
   enters the mirror only AFTER its own attempt (no self-refutation; CaDiCaL
   cannot delete clauses, so conditionality must ride assumptions).
2. **UNSAT proofs**: mirror UNSAT below its assumptions (empty core) means
   the entailed clause set is boolean-unsat ⇒ THE QUERY IS UNSAT. Delivered
   as the root-conflict empty clause; the search ends. Measured: fires on
   the FIRST oracle solve on both anchors. Cannot misfire on satisfiable
   queries (entailed clauses of a SAT query are consistent; verified silent
   on safenlp-418).
3. **Failed-literal probing**: assume(lit) + decisions-0 solve = pure
   propagation over everything learned; conflict ⇒ free unit. Gated on
   mirror-DB growth.
4. **Learner harvest**: `connect_learner` streams the mirror's own learned
   units/binaries back as entailed facts (edge maps + main solver).
5. **Runtime soundness audit**: any unsound clause in any channel surfaces
   as a premature boolean conflict (this is how the "poison" incident
   resolved: both "contradictory" units were verified entailed by fresh
   full solves — the instance was unsat and the mirror had proven it).

**The cadence bottleneck (measured, decides the next step):** all oracle
duties run at level-0 visits, and the first luby restart (512 conflicts)
lands near the END of these ACAS searches — proofs fire at ~95% done,
vivification touches ~20% of the queue. The oracle's power is gated by
level-0 visit frequency, not by its strength. Cheapest capitalization, not
yet built: a PRE-SEARCH oracle pass right after the skeleton probe (level 0
by construction, no restart-schedule change) — catches skeletons that are
already boolean-unsat at visited-state zero. Beyond that: earlier first
restart / periodic forced level-0 visits (deferred by scope decision).

Env knobs: `SKELETON_NO_VIVIFY_MIRROR=1` (disable the oracle entirely),
`SKELETON_NO_MIRROR_PROBE=1` (disable duty 3), `MIRROR_LOG=<file>`
(streaming clause log for offline audit — the solver's own dump is
post-simplification and useless for forensics).

## The graveyard (measured dead — do not resurrect without new evidence)

- **BCP-interleaved descents** (binary-edge propagation woven into the LP
  descent pin sequence: zero-LP shortenings, implied-literal rule, extra
  pins): worked well (78–134 of ~140 shortenings at zero LP cost) but
  REMOVED 2026-07-16 — superseded, not refuted. The mirror's assumption
  cores strictly dominate the shortening duties with a real solver's
  conflict analysis, and deleting the hand-rolled prefix/blockEnd mapping
  removed the largest surface of soundness-proof-by-hand in the codebase.
  The one uncovered loss is implied extra pins for the LP descent
  (unmeasured marginal value; LP hit rate was 88–99% before extras
  existed). Resurrect only as a 10-line closure over the Learner-densified
  edge maps if LP hit rates ever sag.

- **Probe → compact → solve (theory-level neuron elimination)**: root-fixed
  ReLUs eliminated by re-preprocessing the pristine snapshot under the
  refined box, fresh engine, b-space binary transfer with parent-side
  resolution, delegate SAT solutions. Built and fully debugged 2026-07-16,
  then REMOVED the same day (user call: out if it hurts at all). Measured:
  1_2 1,488 (beat the 1,982 control — best config seen) but 3_4 7,876 vs
  3,497 uncompacted — on dense instances compaction consumes the graph (121
  of 141 binaries satisfied-absorbed into the box) and box knowledge does
  not replace boolean guidance. Machinery-only ablation was sound (5,437 vs
  6,062 control). ACAS fixation is only 1–2%; if a high-fixation family ever
  makes this attractive again, the code is in git history (look for
  "Skeleton compaction" around this doc's date). Related law, measured
  three ways now: **clauses are portable; numeric propagation results are
  not** — the child's second-pass hull fold was load-bearing (skipping it
  cost 1_2 its win, 1,488 → 2,821).
- **Cross-instance skeleton cache (SKELETON_CACHE)**: original-space facts +
  hull with a per-variable subset guard; validated end-to-end (safenlp SAT
  instance 418: control 40 visited states, probing 15,401+ — the cluster's
  0.13× SAT disaster is probe-time state perturbation, not clauses; cache
  hit restored control-identical 40). Removed with compaction: family
  instances are sibling boxes (never nested), so real hits need a union-box
  cache whose facts are near-empty, and the one real benefit (skip probing
  on easy instances) doesn't need a cache. The genuine cross-instance
  object is learned clauses under assumption-encoded boxes = incremental
  solving = a different architecture. Diagnosis retained: gate probing by
  cost/benefit, don't cache it.

- **C3, conditional-tightening table**: per feasible pin, memoize the branch
  bounds beating the refined root; apply when the literal holds mid-search.
  Dead in every delivery form on 3_4: applied per-notification **8,073**,
  applied per-theory-check **8,072** (vs 3,934 without) — the damage is the
  bounds being *visible to the LP/branching at all* mid-search, not delivery
  frequency. Trail-independent variants are neutral: root promotion on
  level-0 fixes (24 promotions, byte-identical 3,934 — trail propagation
  subsumes root-probe bounds at depth) and a read-only conflict oracle
  (never fires, 3,934). Payload classification: 4,991 entries = 14%
  phase-fixing (redundant with C1 binaries) + 86% numeric residue with no
  working delivery point. Removed from the code 2026-07-14.
- **Boolean vivification of learned clauses vs skeleton edges**: provably
  vacuous — the seeded binaries make SAT propagation pre-empt every
  first-order removal (scaffold retained in `CdclCore::vivifyClause` for a
  future theory-grade oracle).
- **Rung-0 numeric vivification (probe-vector intersection)**: for each
  learned-clause literal, intersect the stored single-pin probe bound
  vectors of the remaining literals' negations (plus BCP closure over
  skeleton edges) and drop the literal on box emptiness or a forced sign on
  its own `b`. Implemented 2026-07-15 (`CdclCore::vivifyCandidateRemovable`,
  sparse deltas vs the post-hull root box, knobs
  `SKELETON_NO_VIVIFY_BOUNDS` / `SKELETON_VIVIFY_DELTA_FRAC` /
  `SKELETON_VIVIFY_MAX_SIZE`). Measured dead on both calibration instances:
  3_4 = 48,892 numeric checks, 0 removals, 4.4s (visited states byte-identical
  3,934); 1_2 = 0 removals (2,106); unfiltered deltas (frac=0, 9,128 deltas)
  change nothing. Diagnosis: probe tightenings are sparse (~7–18 deltas/pin)
  and single-pin root boxes never jointly cross — the conflicts CDCL learns
  needed LP-grade *joint* reasoning to find, so box-grade joint reasoning
  cannot refute their subsets. Same signature as the C3 autopsy (86% numeric
  residue) and abcrown's BICCOS-edge vivification (0 removals). Conclusion:
  theory vivification needs the rung-1 oracle (joint-pin fixpoint + budgeted
  LP per candidate, one incremental descent per clause); rung 0 cannot
  pre-filter for it — it fires on nothing.
- **SCC phase merging**: forward-only probes make the implication graph a
  layered DAG; 2-cycles essentially impossible.

**Design law (measured repeatedly, both here and in α,β-CROWN):** the value
of probe information is dwarfed by sensitivity to *where* it is injected.
High-frequency, trail-conditioned delivery points inject trajectory chaos
that swamps the information's value; amortized, trail-independent points
(SAT clause database, root tableau) are safe. Deliver once, at the root, or
not at all.

## Next steps

1. Cluster sweep over 10–15 instances for the density-vs-benefit curve
   (single instances are chaos-dominated).
2. Root promotion v2: on a level-0 fix, fold the literal's probe bounds into
   the *root tableau* with a cascade so new root facts feed the SAT solver
   globally (the per-check form was neutral; the global form is untested).
3. DONE 2026-07-15 — **Rung-1 LP vivification + clausalized graph** (the
   "let CaDiCaL maintain the implication graph" architecture). Learned
   clauses queue (`CdclCore::processVivifyLpQueue`); at boolean level 0
   (post-restart, engine at root state) each gets ONE incremental descent
   (`Engine::probePinDescent`): pins of the negated literals applied
   most-tightening-first, fixpoint + budgeted LP after each; infeasibility
   after j pins ⇒ the j-literal prefix replaces the clause. The pin-1
   fixpoint's phase fixes are harvested as binary edges (free graph growth
   in the current root box). Probe mode held throughout; injected clauses
   are excluded from the restart schedule. ~14ms/descent. Knobs:
   `SKELETON_NO_VIVIFY_LP`, `SKELETON_VIVIFY_LP_MAX_SIZE` (32),
   `SKELETON_VIVIFY_LP_CLAUSES` (256/visit), `SKELETON_VIVIFY_LP_BUDGET`
   (60s). Rung 0 (probe-vector intersection) is measured dead (graveyard).
   2026-07-16 update: the boolean half now belongs to the mirror oracle
   (see its section); the LP descent runs plain original-literal pins.
4. **Pre-search oracle pass** (next): run the mirror's first solve right
   after the skeleton probe, before any search — catches boolean-unsat
   skeletons at visited-state zero, no restart-schedule change needed.
5. **Level-0 cadence** (deferred by scope decision): earlier first restart
   or periodic forced level-0 visits would multiply every oracle duty —
   the measured bottleneck for proofs, vivification coverage, and probing.
4. Conditioned re-probing at depth: density grows as boxes shrink (17 vs 144
   root clauses across instances).
