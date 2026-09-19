# llm-leakage-research

## Investigating how much of an LLM's forecasting accuracy on resolved prediction market questions is recall of training data rather than reasoning.

### Question
Forecasting accuracy on resolved prediction-market questions is used to benchmark LLM world model quality. Datasets (AutoCast, ForecastBench) make this concrete: if the model gives calibrated probabilities on resolved questions, then it must have internalized how the world works.

This argument only holds if the model hasn't already seen the outcome. A model that is trained through July 2025 and scored on questions that resolved in 2023 would have had access to the ground truth as training data (resolution note, news coverage, Wikipedia text). The probability might reflect recalling the outcome rather than reasoning toward it.

This question isn't hypothetical. Training data is retrospective and often assembled months after events actually occur. The practical stakes are also high. If a considerable share of measured forecasting accuracy is contaminated, then improvements on forecasting benchmarks might be tracking training data coverage rather than an increase in reasoning ability.

### Method

The model we use is `claude-haiku-4-5-20251001`. The training cutoff for this model is July 2025. The reliable knowledge cutoff is February 2025. The cutoff split uses the training cutoff as the splitting date.

The data we use is 800 resolved binary markets pulled from the Polymarket Gamma API (400 pre-cutoff, 400 post-cutoff). This is then filtered to n=19 pre-cutoff and n=20 post-cutoff questions (the asymmetry comes from the prompt-building step that will drop a question if `resolve_outcome` returns `None`) matched on category and duration bucket. There is a 90-day buffer before and 60-day buffer after the closing date when selecting markets. Filters that we applied are: binary/resolved, volume >= $10K, duration >= 7 days, cohort membership, deduplication.

The metrics we use are Brier score (lower = better) and log loss, computed on the mean probability over 5 independent runs per question. Standard deviation across the runs is reported separately as the noise floor.

We use three controls:
1. Cohort split: pre-cutoff vs post-cutoff questions
2. Stripped-question baseline: for each question, a variant strips the proper nouns, specific dates, and identifying numbers while maintaining grammatical structure and time horizon
3. Repeat-sampling noise floor: 5 runs per question to reduce sampling variance

The elicitation prompt asks the model to forecast as of the market open date using only the information available then, and return a JSON object with a `probability` field.

### Resources
1. Polymarket Gamma API (`gamma-api.polymarket.com/markets`) for market metadata
2. Polymarket Data API v2 (`data-api.polymarket.com/v2/prices-history`) tested but returned empty data for resolved markets
3. Anthropic Python SDK 1.6.0 (`claude-haiku-4-5-20251001`) for LLM calls
4. Anthropic model documentation for training cutoff verification

Computation proceeded locally

### Results
#### H1 (leakage gap)
For the leakage gap, the pre-cutoff Brier = 0.208 and the post-cutoff Brier = 0.054. The gap = -0.154 (95% CI: -0.281 to -0.037), well above the noise floor of 0.0013. The gap is in the opposite direction from H1's prediction: the model performs better on post-cutoff questions.

The reversal is most likely due to selection. The post-cutoff questions are dominated by low-probability tail-risk questions (nuclear detonation, Tether insolvency, USDT depeg, Putin removed, Khamenei removed, Ukraine joins NATO). These are easy questions: the base rate is low and the model's structural prior is well-calibrated. The pre-cutoff questions have more competitive, balanced questions (election margins, box office cutoffs, leadership races) where the model made more errors.

The errors in the top 20 misses confirm this: 11 of the 20 are `world_model` errors (the model's factual priors were wrong), 7 are `overconfident` (Fed rate-cuts, where the model distributed probability uniformly across mutually exclusive buckets), and 2 are `resolution_criteria` errors (NATO by March 31 and Avatar by January 31, where the model's pre-cutoff knowledge of the likely outcome dominated over the specific deadline in the resolution criteria). There were no `ambiguous` cases.

#### H2 (structural baseline)
The pre-cutoff full Brier = 0.208, stripped = 0.135, gap = -0.055 (CI: -0.195 to 0.059) and not significant. The post-cutoff full Brier = 0.054, stripped = 0.139, gap = 0.080 (CI: 0.004 to 0.166) and significant. This is again reversed, as removing specifics degrades performance on post-cutoff questions, consistent with the model relying on structural priors rather than recalled outcomes.

#### H3 (crowd comparison)
The model Brier = 0.054 vs uninformed 0.5-prior baseline = 0.250, gap = -0.196 (CI: -0.222 to -0.162). The model beats the baseline. This is limited by the absence of opening crowd prices.
