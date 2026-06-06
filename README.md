# ternary-viterbi

*The most likely story the trits could tell.*

---

Viterbi decoder for ternary state sequences. Given a sequence of observations in {-1, 0, +1} and a ternary HMM (initial, transition, emission probabilities over 3 states), find the most likely state path that produced those observations.

Works in log-space to avoid floating-point underflow on long sequences. Includes a `sticky` constructor for HMMs where staying in the same state is more likely (common in real ternary systems — a component that's healthy tends to stay healthy).

Also provides path log-likelihood computation and path agreement measurement for comparing decoded paths.

9 tests: uniform and sticky HMMs, state switching on strong evidence, empty/single observations, likelihood consistency, path agreement.

Part of [SuperInstance](https://github.com/SuperInstance/SuperInstance).

License: MIT
