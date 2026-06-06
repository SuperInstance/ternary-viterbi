//! # ternary-viterbi
//!
//! Viterbi algorithm for finding the most likely sequence of ternary states.
//! States and observations are {-1, 0, +1}. Uses log-space computation to
//! avoid floating-point underflow on long sequences.

/// A single trit value.
pub type Trit = i8;

/// Ternary HMM parameters for Viterbi decoding.
#[derive(Debug, Clone)]
pub struct TernaryHMM {
    /// Initial state probabilities: P(state at t=0). Index: trit_to_idx.
    pub initial: [f64; 3],
    /// Transition matrix: P(next_state | current_state). [from][to].
    pub transition: [[f64; 3]; 3],
    /// Emission matrix: P(observation | state). [state][observation].
    pub emission: [[f64; 3]; 3],
}

impl TernaryHMM {
    pub fn uniform() -> Self {
        Self {
            initial: [1.0/3.0; 3],
            transition: [[1.0/3.0; 3]; 3],
            emission: [[1.0/3.0; 3]; 3],
        }
    }

    /// Create with self-transition bias (staying in same state is more likely).
    pub fn sticky(stay_prob: f64) -> Self {
        let other = (1.0 - stay_prob) / 2.0;
        Self {
            initial: [1.0/3.0; 3],
            transition: [
                [stay_prob, other, other],
                [other, stay_prob, other],
                [other, other, stay_prob],
            ],
            emission: [[1.0/3.0; 3]; 3],
        }
    }
}

fn trit_to_idx(t: Trit) -> usize {
    match t {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => panic!("Invalid trit"),
    }
}

fn idx_to_trit(i: usize) -> Trit {
    match i {
        0 => -1,
        1 => 0,
        2 => 1,
        _ => panic!("Invalid index"),
    }
}

/// Result of Viterbi decoding.
#[derive(Debug, Clone)]
pub struct ViterbiResult {
    /// Most likely state sequence.
    pub path: Vec<Trit>,
    /// Log probability of the best path.
    pub log_probability: f64,
    /// Backtrace matrix for inspection.
    pub backtrace: Vec<[usize; 3]>,
}

/// Run Viterbi decoding on a sequence of ternary observations.
pub fn decode(hmm: &TernaryHMM, observations: &[Trit]) -> ViterbiResult {
    let timesteps = observations.len();
    if timesteps == 0 {
        return ViterbiResult { path: vec![], log_probability: 0.0, backtrace: vec![] };
    }

    // Viterbi tables in log space
    let neg_inf = f64::NEG_INFINITY;
    let mut viterbi = vec![[neg_inf; 3]; timesteps];
    let mut backtrace = vec![[0usize; 3]; timesteps];

    // Initialize t=0
    let obs0 = trit_to_idx(observations[0]);
    for s in 0..3 {
        viterbi[0][s] = hmm.initial[s].ln() + hmm.emission[s][obs0].ln();
    }

    // Recursion
    for t in 1..timesteps {
        let obs = trit_to_idx(observations[t]);
        for s in 0..3 {
            let mut best_score = neg_inf;
            let mut best_prev = 0;
            for ps in 0..3 {
                let score = viterbi[t-1][ps] + hmm.transition[ps][s].ln() + hmm.emission[s][obs].ln();
                if score > best_score {
                    best_score = score;
                    best_prev = ps;
                }
            }
            viterbi[t][s] = best_score;
            backtrace[t][s] = best_prev;
        }
    }

    // Find best final state
    let (best_final, best_score) = viterbi[timesteps-1].iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &s)| (i, s))
        .unwrap_or((0, neg_inf));

    // Backtrace
    let mut path = vec![0usize; timesteps];
    path[timesteps-1] = best_final;
    for t in (1..timesteps).rev() {
        path[t-1] = backtrace[t][path[t]];
    }

    ViterbiResult {
        path: path.iter().map(|&i| idx_to_trit(i)).collect(),
        log_probability: best_score,
        backtrace,
    }
}

/// Compute log-likelihood of an observation sequence given state path.
pub fn path_log_likelihood(hmm: &TernaryHMM, observations: &[Trit], path: &[Trit]) -> f64 {
    if observations.is_empty() { return 0.0; }
    let obs0 = trit_to_idx(observations[0]);
    let s0 = trit_to_idx(path[0]);
    let mut ll = hmm.initial[s0].ln() + hmm.emission[s0][obs0].ln();

    for t in 1..observations.len() {
        let prev_s = trit_to_idx(path[t-1]);
        let curr_s = trit_to_idx(path[t]);
        let obs = trit_to_idx(observations[t]);
        ll += hmm.transition[prev_s][curr_s].ln() + hmm.emission[curr_s][obs].ln();
    }
    ll
}

/// Compare two paths: how many positions agree.
pub fn path_agreement(a: &[Trit], b: &[Trit]) -> f64 {
    if a.len().min(b.len()) == 0 { return 0.0; }
    let min_len = a.len().min(b.len());
    let matches = (0..min_len).filter(|&i| a[i] == b[i]).count();
    matches as f64 / min_len as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_short_sequence() {
        let hmm = TernaryHMM::uniform();
        let obs = vec![1, 1, 1];
        let result = decode(&hmm, &obs);
        assert_eq!(result.path.len(), 3);
        assert!(result.log_probability.is_finite());
    }

    #[test]
    fn test_sticky_prefers_same_state() {
        let hmm = TernaryHMM::sticky(0.9);
        let obs = vec![1, 1, 1, 1, 1];
        let result = decode(&hmm, &obs);
        // With high self-transition and uniform emission, should stay in same state
        assert_eq!(result.path, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_sticky_switches_on_evidence() {
        let mut hmm = TernaryHMM::sticky(0.7);
        // Strong emission: each state always emits itself
        hmm.emission = [
            [0.9, 0.05, 0.05], // state -1 mostly emits -1
            [0.05, 0.9, 0.05], // state 0 mostly emits 0
            [0.05, 0.05, 0.9], // state 1 mostly emits 1
        ];
        let obs = vec![-1, -1, 0, 0, 1, 1];
        let result = decode(&hmm, &obs);
        // Should follow the observations when emission is strong
        assert_eq!(result.path, obs);
    }

    #[test]
    fn test_empty_observation() {
        let hmm = TernaryHMM::uniform();
        let result = decode(&hmm, &[]);
        assert!(result.path.is_empty());
    }

    #[test]
    fn test_single_observation() {
        let hmm = TernaryHMM::uniform();
        let result = decode(&hmm, &[0]);
        assert_eq!(result.path.len(), 1);
    }

    #[test]
    fn test_path_log_likelihood() {
        let hmm = TernaryHMM::sticky(0.8);
        let obs = vec![1, 1, 1];
        let path = vec![1, 1, 1];
        let ll = path_log_likelihood(&hmm, &obs, &path);
        assert!(ll.is_finite());
        assert!(ll < 0.0); // log of probability < 1
    }

    #[test]
    fn test_viterbi_likelihood_matches_manual() {
        let mut hmm = TernaryHMM::sticky(0.8);
        hmm.emission = [[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]];
        let obs = vec![1, 1];
        let result = decode(&hmm, &obs);
        let manual_ll = path_log_likelihood(&hmm, &obs, &result.path);
        assert!((result.log_probability - manual_ll).abs() < 1e-10);
    }

    #[test]
    fn test_path_agreement_identical() {
        assert!((path_agreement(&[1,-1,0], &[1,-1,0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_path_agreement_different() {
        assert!((path_agreement(&[1,-1,0], &[-1,1,0]) - 1.0/3.0).abs() < 1e-10);
    }
}
