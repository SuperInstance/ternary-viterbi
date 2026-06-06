# ternary-viterbi

*The Viterbi algorithm for ternary HMMs. Find the most likely {-1, 0, +1} state sequence using log-space computation.*

## Why This Exists

Hidden Markov Models are everywhere — speech recognition, bioinformatics, time-series analysis. But when your state space is ternary {-1, 0, +1}, the standard Viterbi implementation does unnecessary work. With only 3 states and 3 possible observations, the trellis has exactly 9 transitions per timestep. This crate exploits that structure for a minimal, correct, numerically stable decoder.

## The Algorithm

The Viterbi algorithm finds the most likely state sequence given observations and an HMM. For ternary HMMs:
- 3 states: -1, 0, +1
- 3 observation symbols: -1, 0, +1
- 9 transition probabilities (3×3)
- 9 emission probabilities (3×3)

All computation happens in log-space to prevent floating-point underflow on long sequences.

## Usage

```rust
use ternary_viterbi::*;

let hmm = TernaryHMM {
    initial: [-1.0_f64.ln(), (1.0/3.0_f64).ln(), (1.0/3.0_f64).ln()],
    transition: [[...your 3x3 transition log-probs...]],
    emission: [[...your 3x3 emission log-probs...]],
};

let observations: &[i8] = &[1, 1, -1, 0, 1];
let result = decode(&hmm, observations);

println!("Most likely states: {:?}", result.path);
println!("Log-likelihood: {}", result.log_likelihood);

// Check agreement between two paths
let agreement = path_agreement(&result.path, &ground_truth);
```

## Related Crates

- `ternary-hmm` — HMM training (Baum-Welch)
- `ternary-baum-welch` — EM training for ternary HMMs
- `ternary-knn` — K-nearest neighbors (another classification approach)
