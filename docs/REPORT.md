# Technical Report: Learning-Augmented Euclidean TSP

## 1. Scope and research question

This framework studies how exact optimization, classical heuristics, supervised
signals, and online adaptation interact on the symmetric Euclidean traveling
salesperson problem (TSP). Its purpose is controlled experimentation and clear
implementation. No empirical result is embedded in this report; results must be
generated from a recorded command and code revision.

The central experimental question is whether a learned edge signal, trained on
small exact instances, can improve constructive decisions or complement 2-opt
on held-out Euclidean instances. A separate online experiment asks whether an
expert-weighting rule can adapt among fixed algorithms as instances arrive.

## 2. Problem formulation

An instance contains points \(V=\{0,\ldots,n-1\}\) with coordinates
\(x_i\in\mathbb{R}^2\). The complete undirected graph has edge costs

\[
d_{ij}=\|x_i-x_j\|_2=d_{ji}.
\]

For a permutation \(\pi\), the framework minimizes the closed-tour length

\[
L(\pi)=\sum_{k=0}^{n-1}d_{\pi_k,\pi_{(k+1)\bmod n}}.
\]

The stored tour lists each vertex once. The final return to \(\pi_0\) is
implicit. This convention is validated at module boundaries.

## 3. Exact reference: Held--Karp

Fix a start vertex \(s\). For \(S\subseteq V\setminus\{s\}\) and \(j\in S\),
define

\[
C(S,j)=\min\{\text{cost of a path from }s\text{ through exactly }S
\text{ and ending at }j\}.
\]

The base case and recurrence are

\[
C(\{j\},j)=d_{sj},
\qquad
C(S,j)=\min_{i\in S\setminus\{j\}}
\left[C(S\setminus\{j\},i)+d_{ij}\right].
\]

The optimum is

\[
L^*=\min_{j\ne s}\left[C(V\setminus\{s\},j)+d_{js}\right].
\]

Predecessors stored during the recurrence reconstruct an optimal permutation.
The method uses \(O(n^2 2^n)\) time and \(O(n2^n)\) space. The implementation
therefore has a configurable maximum size and raises an error beyond it. It is
a source of ground truth and training labels, not a scalable solver.

## 4. Classical baselines

### 4.1 Nearest neighbor

From a chosen start, nearest neighbor repeatedly appends the closest unvisited
city. Equal distances are broken by city index, making the solver deterministic.
An optional variant evaluates every starting city and keeps the shortest tour.
Its construction takes \(O(n^2)\) time with the dense distance matrix.

Nearest neighbor is fast and interpretable but makes irrevocable local choices.
It has no optimality claim here.

### 4.2 2-opt

Given tour edges \((a,b)\) and \((c,d)\), a 2-opt move removes them and reverses
the segment between \(b\) and \(c\), producing edges \((a,c)\) and \((b,d)\).
The cost change is

\[
\Delta=d_{ac}+d_{bd}-d_{ab}-d_{cd}.
\]

The implementation accepts a move only when \(\Delta\) is below a numerical
tolerance, uses first improvement, and repeats passes until no improving move
remains. It verifies the returned tour is no worse than the initial tour. The
primary classical pipeline is nearest-neighbor construction followed by 2-opt.

## 5. Supervised edge-scoring heuristic

### 5.1 Labels and features

Training instances are solved exactly. Every undirected pair receives label 1
when it occurs in the optimal cycle and 0 otherwise. The five pair features are:

1. absolute horizontal coordinate difference;
2. absolute vertical coordinate difference;
3. Euclidean pair distance;
4. the smaller endpoint radius from the instance centroid;
5. the larger endpoint radius from the instance centroid.

Coordinates are centered and divided by the root-mean-square radius. The
features are consequently invariant to translation and uniform scaling, and
sorting endpoint radii preserves undirected symmetry.

### 5.2 Model and training

A two-hidden-layer multilayer perceptron produces one edge logit. Training uses
weighted binary cross-entropy and Adam in a deterministic, full-batch CPU loop.
Positive weighting addresses the imbalance between \(n\) tour edges and
\(n(n-1)/2\) possible edges. The architecture is intentionally small so the
experiment focuses on the value and limitations of the signal.

### 5.3 Decoding and leakage control

Starting from a chosen city, the decoder repeatedly chooses the unvisited city
with the highest predicted edge logit. Ties fall back to Euclidean distance and
city index. Optional 2-opt separates the learned construction from classical
post-processing in result metadata.

Training and evaluation are separate CLI operations. Model artifacts record
training-instance names, and the learned solver rejects any generated instance
whose name appears in that set. Experimental protocols must also use disjoint
seed sets, as shown in the README.

This model estimates edge utility; it neither represents the complete tour
state nor guarantees feasibility before decoding. Feasibility comes from the
decoder, and optimality is never claimed.

## 6. Online algorithm selection with Hedge

A round is one newly observed TSP instance. Each expert is a complete TSP
solver. Before observing round costs, the selector chooses the expert with the
largest current probability, using declaration order for ties. It then runs all
experts, so the feedback model is full information.

For costs \(L_{t,i}\), the bounded round loss is

\[
\ell_{t,i}=\begin{cases}
0, & \max_jL_{t,j}=\min_jL_{t,j},\\
\dfrac{L_{t,i}-\min_jL_{t,j}}{\max_jL_{t,j}-\min_jL_{t,j}}, & \text{otherwise}.
\end{cases}
\]

With learning rate \(\eta>0\), Hedge updates

\[
w_{t+1,i}=w_{t,i}\exp(-\eta\ell_{t,i}),
\qquad
p_{t+1,i}=\frac{w_{t+1,i}}{\sum_j w_{t+1,j}}.
\]

The learned state is a distribution over algorithms, reflecting their observed
relative performance. The implementation follows the multiplicative-weights
idea; this project does not claim a new regret theorem for the chosen
instance-dependent normalization or deterministic action rule.

## 7. Experimental protocol and metrics

Uniform instances use NumPy's modern generator in \([0,1)^2\). A benchmark is
the Cartesian product of requested sizes and seeds. All solvers receive the
same instance object. Records contain size, seed, identifier, solver, tour
length, runtime, exact optimum when available, and

\[
\operatorname{gap}(L,L^*)=\frac{L-L^*}{L^*}.
\]

The gap is absent when no exact result exists. A negative gap beyond tolerance
is treated as inconsistent data. Solver-reported lengths are independently
recomputed from their tours before a record is accepted.

A defensible learned-method study should predeclare disjoint training and test
seeds, report distributions rather than a single favorable instance, compare
both constructive and post-2-opt variants, and record hardware and software
versions. Runtime plots use a logarithmic axis; gap plots aggregate means by
size. Raw CSV should accompany any later interpretation.

## 8. Verification strategy

Tests cover geometric calculations, route invariants, seeded generation,
heuristic determinism, non-worsening 2-opt behavior, common solver conformance,
CSV semantics, online updates, feature invariance, training reproducibility,
and train/test overlap rejection. Held--Karp is compared with exhaustive
permutation search on independent tiny instances.

Static checks use Ruff and strict mypy. CI runs on Python 3.12 and requires no
GPU. Coverage is diagnostic: missing important branches should guide tests,
while the project does not optimize for an arbitrary percentage.

## 9. Limitations

- Dense distances require \(O(n^2)\) memory.
- Held--Karp and exact-label generation scale exponentially.
- The learned model sees pair geometry rather than the partial-tour state.
- Greedy decoding can create globally poor choices even with useful edge scores.
- Synthetic uniform instances do not represent all Euclidean distributions.
- Benchmark runtimes include Python implementation effects and machine noise.
- Full-information Hedge pays the cost of running every expert each round.

These limits motivate later work on stronger baselines, broader distributions,
calibrated evaluation, learned state-aware decisions, and bandit-feedback
selection. Such work should be reported only after reproducible measurement.

## References

- R. Bellman, “Dynamic Programming Treatment of the Travelling Salesman
  Problem,” *Journal of the ACM*, 9(1), 61–63, 1962.
- M. Held and R. M. Karp, “A Dynamic Programming Approach to Sequencing
  Problems,” *Journal of the Society for Industrial and Applied Mathematics*,
  10(1), 196–210, 1962.
- G. A. Croes, “A Method for Solving Traveling-Salesman Problems,” *Operations
  Research*, 6(6), 791–812, 1958.
- Y. Freund and R. E. Schapire, “A Decision-Theoretic Generalization of On-Line
  Learning and an Application to Boosting,” *Journal of Computer and System
  Sciences*, 55(1), 119–139, 1997.
