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

## The graveyard (measured dead — do not resurrect without new evidence)

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
3. Theory-grade vivification: upgrade the vivification oracle to the probe
   stack (pin the negations of `C∖{l}`, fixpoint + budgeted LP, infeasible ⇒
   drop `l`), selectively — short clauses / on-reuse. Expensive; gate on
   sweep evidence.
4. Conditioned re-probing at depth: density grows as boxes shrink (17 vs 144
   root clauses across instances).
