//! Chunk-score -> document-score aggregation.
//!
//! Retrieval returns per-chunk scores; ranking *documents* needs an
//! aggregation. The two classical ones are both flawed:
//!   max  (== old min-distance)  best single chunk; rewards any long doc
//!                                that happens to contain one lucky chunk.
//!   mean                        punishes long docs (irrelevant chunks dilute
//!                                a genuinely relevant one toward the floor).
//! Default here is top-K mean: mean of the best K chunks. Rewards docs with
//! *several* relevant passages, is length-neutral once len>=K, and reduces
//! to mean for short docs. noisy_or is a probabilistic union of chunk
//! relevances: P(doc relevant) = 1 - prod(1 - P(chunk relevant)); saturating
//! but well-calibrated if scores are in [0,1].
//!
//! Scores are always "higher = more relevant" (callers convert cosine
//! distance to similarity = 1 - distance before aggregating).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocScoreMethod {
    Max,
    Mean,
    TopK,
    NoisyOr,
}

impl DocScoreMethod {
    pub fn parse(s: &str) -> Option<DocScoreMethod> {
        match s.to_ascii_lowercase().as_str() {
            "max" => Some(DocScoreMethod::Max),
            "mean" => Some(DocScoreMethod::Mean),
            "topk" | "top_k" | "topk_mean" => Some(DocScoreMethod::TopK),
            "noisy_or" | "noisy-or" => Some(DocScoreMethod::NoisyOr),
            _ => None,
        }
    }
}

/// Aggregate a document's chunk scores (higher=better) into one score.
pub fn doc_score(scores: &[f64], method: DocScoreMethod, k: usize) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    match method {
        DocScoreMethod::Max => scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        DocScoreMethod::Mean => scores.iter().sum::<f64>() / scores.len() as f64,
        DocScoreMethod::TopK => {
            let mut v = scores.to_vec();
            v.sort_by(|a, b| b.total_cmp(a));
            let kk = k.max(1).min(v.len());
            v[..kk].iter().sum::<f64>() / kk as f64
        }
        DocScoreMethod::NoisyOr => {
            let one_minus = scores
                .iter()
                .map(|s| 1.0 - s.clamp(0.0, 1.0))
                .fold(1.0, |a, b| a * b);
            1.0 - one_minus
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_picks_best_single_chunk() {
        let s = [0.2, 0.9, 0.3];
        assert_eq!(doc_score(&s, DocScoreMethod::Max, 5), 0.9);
    }

    #[test]
    fn mean_dilutes_a_buried_hit() {
        // one great chunk in a long doc: mean buries it
        let long: Vec<f64> = std::iter::once(0.9)
            .chain(std::iter::repeat_n(0.1, 19))
            .collect();
        assert!(doc_score(&long, DocScoreMethod::Mean, 5) < 0.15);
    }

    #[test]
    fn topk_recognizes_multi_hit_docs_without_length_penalty() {
        // 3 strong chunks buried in 17 weak ones -> topk sees the hits
        let mut multi: Vec<f64> = vec![0.85, 0.8, 0.75];
        multi.extend(std::iter::repeat_n(0.1, 17));
        // single-hit long doc
        let mut single: Vec<f64> = vec![0.86];
        single.extend(std::iter::repeat_n(0.1, 19));
        let m = DocScoreMethod::TopK;
        assert!(doc_score(&multi, m, 5) > doc_score(&single, m, 5));
    }

    #[test]
    fn topk_reduces_to_mean_for_short_docs() {
        let s = [0.4, 0.6, 0.8];
        assert!((doc_score(&s, DocScoreMethod::TopK, 5) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn noisy_or_saturates_and_beats_single_hit() {
        let single = [0.8];
        let two = [0.6, 0.6];
        assert!((doc_score(&single, DocScoreMethod::NoisyOr, 5) - 0.8).abs() < 1e-9);
        // 0.6 + 0.6 independent hits -> 1 - 0.4*0.4 = 0.84
        assert!((doc_score(&two, DocScoreMethod::NoisyOr, 5) - 0.84).abs() < 1e-9);
        assert!(
            doc_score(&two, DocScoreMethod::NoisyOr, 5)
                > doc_score(&single, DocScoreMethod::NoisyOr, 5)
        );
    }

    #[test]
    fn empty_scores_is_zero() {
        for m in [
            DocScoreMethod::Max,
            DocScoreMethod::Mean,
            DocScoreMethod::TopK,
            DocScoreMethod::NoisyOr,
        ] {
            assert_eq!(doc_score(&[], m, 5), 0.0);
        }
    }
}
