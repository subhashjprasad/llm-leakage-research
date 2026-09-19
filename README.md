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
