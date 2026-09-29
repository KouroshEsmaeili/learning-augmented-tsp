# Learning-Augmented TSP

Learning-Augmented TSP is a reproducible experimental framework for asking a
focused question: **when can learned signals improve transparent algorithmic
decisions for the symmetric Euclidean traveling salesperson problem?** It puts
exact optimization, classical heuristics, a small supervised model, and online
algorithm selection behind compatible solver interfaces so they can be evaluated
on the same instances.

The project is educational and research-oriented. It does not claim a new
algorithm or state-of-the-art performance. Exact solutions are used only at
sizes where their exponential cost is practical, and learned results are always
reported as heuristic results.

## Supported problem

For city coordinates \(x_0,\ldots,x_{n-1}\in\mathbb{R}^2\), the project uses the
symmetric distance \(d(i,j)=\lVert x_i-x_j\rVert_2\). A tour is a permutation
\(\pi\) of the city indices. Its closing edge is implicit, and its objective is

\[
L(\pi)=\sum_{k=0}^{n-1}d\bigl(\pi_k,\pi_{(k+1)\bmod n}\bigr).
\]

Coordinates must be finite two-dimensional values. Asymmetric costs, time
windows, capacities, and other TSP variants are outside the current scope.

## Implemented methods

- **Held--Karp dynamic programming** computes and reconstructs exact tours for
  small instances. Its \(O(n^2 2^n)\) time and \(O(n2^n)\) memory requirements
  are enforced with a configurable size limit.
- **Nearest neighbor** provides deterministic construction from a chosen city,
  with an optional best-of-all-starts mode.
- **2-opt** performs deterministic first-improvement local search. The standard
  classical baseline is nearest neighbor followed by 2-opt.
- **Learned edge scoring** trains a compact PyTorch multilayer perceptron on
  optimal-tour edge labels from separate small training instances. Greedy
  construction can be followed by 2-opt. Saved models retain training-instance
  identifiers so known train/evaluation overlap is rejected.
- **Hedge algorithm selection** treats solvers as experts. After each fully
  observed instance, it updates their probabilities using normalized tour-cost
  losses and a multiplicative-weights update.

## Architecture

```text
src/tsp_learning/
├── problem.py, distance.py, route.py, result.py
├── solvers/
│   ├── exact/held_karp.py
│   ├── heuristics/{nearest_neighbor,two_opt}.py
│   └── learning/{features,model,solver,training}.py
├── instances/synthetic.py
├── experiments/benchmark.py
├── online/hedge.py
├── visualization.py
└── cli.py
```

`TSPInstance`, `SolveResult`, and the structural `Solver` protocol form the
small domain layer. Experiment code depends only on that interface, so exact,
classical, and learned solvers produce directly comparable records.

## Installation

Python 3.12 is the primary target.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Core dependencies are NumPy and PyTorch. Matplotlib is included in the `dev`
extra and can otherwise be installed with `.[plot]`. Pandas, Gymnasium, and
tqdm are deliberately omitted because the current experiments do not need them.

## Quick start

Solve one deterministic instance:

```bash
tsp-learning solve --cities 9 --seed 7 --solver nearest-neighbor-2opt
```

Or compare the small baselines in Python:

```bash
python examples/quickstart.py
```

Generate a CSV benchmark. Held--Karp supplies the optimum only up to the stated
limit; larger rows leave optimum-dependent fields empty.

```bash
tsp-learning benchmark \
  --sizes 5,8,10,14 \
  --seeds 0,1,2 \
  --exact-max-cities 10 \
  --output artifacts/benchmark.csv
```

Train on explicitly separate seeds, then evaluate on new seeds:

```bash
tsp-learning train \
  --sizes 5,6,7 \
  --seeds 100,101,102 \
  --epochs 100 \
  --output artifacts/edge-model.pt

tsp-learning benchmark \
  --sizes 6,8,10 \
  --seeds 200,201,202 \
  --model artifacts/edge-model.pt \
  --output artifacts/learned-benchmark.csv
```

Additional commands run the online selector and create plots:

```bash
tsp-learning online --cities 12 --seeds 0,1,2,3,4
tsp-learning plot --input artifacts/benchmark.csv --output-dir artifacts/plots
```

Run `tsp-learning <command> --help` for all options.

## Experiment methodology

Synthetic coordinates use `numpy.random.default_rng(seed)` in the unit square.
Every solver in a benchmark receives the same generated instance. Each CSV row
contains the size, seed, instance identifier, solver, tour length, runtime,
available optimum, and relative gap

\[
\text{gap}=\frac{L_{\text{solver}}-L^*}{L^*}.
\]

The gap is computed only when Held--Karp has supplied a true optimum. Runtime is
wall-clock time and should be interpreted as an implementation-level measurement,
not as a hardware-independent comparison. Generated CSV files, plots, and model
artifacts belong in the ignored `artifacts/` directory.

For reproducibility, record the command, Python environment, code revision,
training seeds, evaluation seeds, and machine. The model artifact stores its
architecture, optimizer settings, training seed, and training-instance names.

## Quality checks

```bash
ruff check .
mypy src
pytest --cov=tsp_learning --cov-report=term-missing
```

GitHub Actions runs these checks on Python 3.12 without a GPU. Tests use small
instances and compare Held--Karp against exhaustive search, rather than assuming
the exact implementation is correct.

## Limitations and planned work

The learned scorer uses local pair geometry and greedy decoding; it is a compact
experimental baseline, not a modern neural combinatorial optimizer. Labels from
Held--Karp restrict supervised training to small instances. Hedge uses full
information because it evaluates every expert after each round, and its current
loss normalization is relative to the experts present in that round.

Useful extensions include stronger constructive baselines, repeated train/test
splits across instance distributions, uncertainty-aware learned guidance,
partial-information online selection, and statistical summaries over larger
benchmark suites. Those extensions should be driven by measured experiments,
not added as unsupported claims.

More mathematical detail is in [docs/REPORT.md](docs/REPORT.md).

## License

This project is available under the MIT License.
