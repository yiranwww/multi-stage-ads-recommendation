# Interview Preparation — Multi-Stage Recommendation System

## 30-second pitch (straightforward spoken English)

I built an end-to-end recommendation portfolio project using RetailRocket behavior data. The system combines several retrieval methods, then uses LightGBM for pre-ranking, DeepFM for fine-ranking, and MMR for diversity. I also deployed a local FastAPI endpoint and checked that its output matched the offline pipeline. My main focus was the trade-off between recommendation quality, latency, and engineering complexity.

## 2-minute project walkthrough

**Problem:** Rank relevant products for users from sparse implicit actions. The dataset has user events, not impression or click-through logs, so this is an offline next-interaction ranking task rather than true ad CTR prediction.

**Data:** I used a time-based train/validation/test split and built user, item, and category features using DuckDB. I was careful about data leakage and separated historical features from future labels.

**Retrieval:** I combined popularity, item-based collaborative filtering, SVD/FAISS, and category-aware candidates with weighted reciprocal rank fusion. The fused stage returns 500 candidates per user.

**Ranking:** LightGBM scores all 500 and keeps 100; DeepFM scores the 100 survivors; MMR reorders the results using a frozen λ of 0.9. The keep count was selected using validation data only.

**Evaluation:** On the test split, Recall@20 moved from 7.414% for the full-500 MMR reference to 7.572% for the selected cascade, while NDCG@20 moved from 0.05565 to 0.05609. Category diversity and coverage decreased, so the change is a trade-off, not a universal improvement.

**Serving:** I exposed the cascade with FastAPI. Models run inference in the server, but retrieval and feature values are precomputed. On 55 local sequential timed HTTP requests, P50 was 21.23 ms. Serving-to-offline cascade parity was 100% on 60 sampled users.

**What I would do next:** Add live feature freshness, a more realistic impression-based dataset, concurrent load tests, diversity constraints, and online A/B testing if production traffic were available.

## Questions and concise answers

**Q1. Why use multi-stage ranking instead of one model?**  
A: Expensive models should not score the whole catalog. Retrieval reduces millions of possible items to hundreds, LightGBM filters them, and DeepFM spends computation on the most promising candidates.

**Q2. Why weighted RRF?**  
A: Retrieval sources produce scores on different scales. RRF combines their ranks without needing perfectly calibrated scores.

**Q3. Why LightGBM before DeepFM?**  
A: LightGBM is a strong tabular baseline and a useful pruning stage. DeepFM models feature interactions, but running it on fewer candidates reduces work.

**Q4. Why did pruning slightly improve Recall@20?**  
A: Removing low-quality candidates can change the final competition for the Top-20. The measured gain is small and could reflect sample variation; it does not prove pruning is intrinsically better.

**Q5. What is the biggest quality trade-off?**  
A: Test category coverage@20 dropped from 4.781 to 4.320 and ILD@20 from 0.3115 to 0.2964. I would add a diversity guardrail before using this in a product.

**Q6. Is 7.57% Recall@20 good?**  
A: It depends on the candidate pool, label definition, sparsity, and dataset. I would compare against baselines on the same split, not against unrelated commercial CTR numbers.

**Q7. Why not report CTR or conversion rate?**  
A: The data does not contain impression exposure logs. Unobserved interactions are not known negatives, so I report offline ranking metrics rather than calibrated online CTR.

**Q8. How did you avoid leakage?**  
A: I used time cutoffs and computed historical features from data available before the label period. I selected the cascade size using validation only. I would still audit all earlier modeling decisions before claiming the final test was fully untouched.

**Q9. How do you measure inference consistency?**  
A: I replay the same cached user candidates and features through the service and compare ordered Top-20 IDs with the offline cascade output.

**Q10. Is your HTTP latency production-ready?**  
A: No. It is a localhost, single-worker, sequential warm benchmark with 55 measured requests. It excludes live retrieval, feature generation, network conditions, cold starts, and concurrency.

**Q11. Why use MMR?**  
A: Pure relevance ranking can show near-duplicate categories. MMR balances relevance with item dissimilarity, but it may reduce Recall or NDCG.

**Q12. What would you optimize first?**  
A: I would profile the real end-to-end service under concurrent traffic, then optimize feature conversion and MMR. I would also improve candidate retrieval, because downstream ranking cannot recover missed items.

**Q13. What was your hardest engineering problem?**  
A: Maintaining feature schema and score parity between offline training and online model inference, especially category vocabularies and preprocessing.

**Q14. Why does DeepFM not always outperform LightGBM?**  
A: Tree models can be very strong on engineered tabular features. Deep models need enough signal, carefully tuned sampling, and regularization. More complex is not automatically better.

**Q15. What is the difference between recall and NDCG?**  
A: Recall measures how many relevant future interactions appear in the Top-K. NDCG also rewards placing relevant items higher.

## Honest resume bullets (adapt to your actual ownership)

- Built a reproducible multi-stage recommendation pipeline on RetailRocket implicit events using DuckDB, FAISS, LightGBM, DeepFM, and MMR, with temporal evaluation and offline ranking metrics.
- Implemented a validation-selected Top-500 → Top-100 cascade, reducing DeepFM candidate volume by 80% while achieving 7.57% test Recall@20 and 0.0561 NDCG@20.
- Served precomputed-candidate recommendations through FastAPI with real LightGBM/DeepFM inference, 100% sampled offline-to-serving Top-20 parity, and 21.23 ms local sequential HTTP P50 latency.

## What not to claim

- Do not claim online CTR lift, A/B-test improvement, production QPS, live retrieval, real-time features, causal business impact, or broad generalization from 55 timed requests.
