# Ternary Viterbi

The **Viterbi algorithm** for ternary hidden Markov models — finds the most likely sequence of hidden states from a sequence of observations, where both states and observations are drawn from {−1, 0, +1}. Uses log-space computation to avoid floating-point underflow on long sequences.

## Why It Matters

Hidden Markov Models (HMMs) are the workhorse of sequence inference: speech recognition, bioinformatics (gene prediction), NLP (POS tagging), and time-series analysis. The Viterbi algorithm finds the **maximum a posteriori** state sequence — the single most likely explanation for the observations. Standard Viterbi assumes discrete states; this crate specializes to the ternary domain where the three states carry semantic meaning: Positive (chosen), Neutral (undecided), Negative (avoided). This is directly useful in the SuperInstance framework for reconstructing an agent's internal state trajectory from its observable actions. If we see a sequence of choices and avoidances, what was the agent's actual decision state at each timestep?

## How It Works

### HMM Definition

A ternary HMM has three components:

```rust
struct TernaryHMM {
    initial: [f64; 3],         // P(state at t=0): [P(-1), P(0), P(+1)]
    transition: [[f64; 3]; 3], // P(next | current): [from][to]
    emission: [[f64; 3]; 3],   // P(obs | state): [state][obs]
}
```

### Viterbi Recursion (Log-Space)

All probabilities are converted to log-space to prevent underflow:

```
δ₀(s) = log P(s) + log P(o₀ | s)
δₜ(s) = max_s' [δₜ₋₁(s') + log P(s | s')] + log P(oₜ | s)
ψₜ(s) = argmax_s' [δₜ₋₁(s') + log P(s | s')]   (backtrace pointer)
```

After processing all T observations, the best final state is selected and the path is reconstructed by following backtrace pointers:

```
s*_T = argmax_s δ_T(s)
s*_{t-1} = ψ_t(s*_t)
```

### Sticky Model

The `sticky(stay_prob)` constructor creates an HMM where self-transitions are more likely:

```
P(stay) = stay_prob
P(transition) = (1 - stay_prob) / 2
```

This models agents that tend to maintain their state — a common assumption in behavioral modeling.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Decoding (T observations) | O(T · K²) | O(T · K) |
| Path log-likelihood | O(T · K) | O(1) |
| Path agreement | O(T) | O(1) |

K = 3 (ternary states), so each step is just 9 comparisons — extremely fast. For T = 10,000 observations: 90,000 comparisons, sub-millisecond on modern hardware.

### Comparison with Forward-Backward

Viterbi finds the single best path; Forward-Backward computes marginal probabilities. Viterbi is appropriate when you need *the* explanation, not a distribution over explanations.

## Quick Start

```rust
use ternary_viterbi::{TernaryHMM, decode};

fn main() {
    // Sticky HMM: 80% chance of staying in same state
    let hmm = TernaryHMM::sticky(0.8);

    // Observation sequence: +1, +1, 0, -1, -1
    let observations = vec![1, 0, 0, -1, -1];

    let result = decode(&hmm, &observations);

    println!("Most likely states: {:?}", result.path);
    println!("Log probability: {:.4}", result.log_probability);
    println!("Path agreement with observations: {:.1}%",
        result.path.iter()
            .zip(&observations)
            .filter(|(p, o)| p == o)
            .count() as f64 / observations.len() as f64 * 100.0);
}
```

```bash
cargo build
cargo test
```

## API

| Type/Function | Description |
|---------------|-------------|
| `TernaryHMM` | HMM with initial, transition, emission matrices |
| `TernaryHMM::uniform()` | All probabilities equal (1/3) |
| `TernaryHMM::sticky(stay_prob)` | Self-transition bias |
| `decode(hmm, observations)` | Run Viterbi → `ViterbiResult` |
| `ViterbiResult` | `.path`, `.log_probability`, `.backtrace` |
| `path_log_likelihood(hmm, obs, path)` | Log-likelihood of a given path |
| `path_agreement(a, b)` | Fraction of matching positions |

## Architecture Notes

Ternary Viterbi reconstructs hidden γ/η states from observable actions. The observations are the agent's external behavior; the hidden states are the internal ternary decision-making. In the γ + η = C framework, Viterbi answers: "given what we saw, what was the agent's γ (constructive) vs η (avoidant) state at each moment?" The sticky model captures the empirical observation that agents don't flip between constructive and avoidant states randomly — they have momentum. The log-probability of the best path measures the agent's internal consistency. See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Viterbi, A. J. (1967). "Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm." *IEEE Transactions on Information Theory*, 13(2), 260–269.
2. Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2), 257–286.
3. Forney, G. D. (1973). "The Viterbi Algorithm." *Proceedings of the IEEE*, 61(3), 268–278.

## License

MIT
