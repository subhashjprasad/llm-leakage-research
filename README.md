# llm-leakage-research

## Investigating how much of an LLM's forecasting accuracy on resolved prediction market questions is recall of training data rather than reasoning.

### Question
Forecasting accuracy on resolved prediction-market questions is used to benchmark LLM world model quality. Datasets (AutoCast, ForecastBench) make this concrete: if the model gives calibrated probabilities on resolved questions, then it must have internalized how the world works.

This argument only holds if the model hasn't already seen the outcome. A model that is trained through July 2025 and scored on questions that resolved in 2023 would have had access to the ground truth as training data (resolution note, news coverage, Wikipedia text). The probability might reflect recalling the outcome rather than reasoning toward it.

This question isn't hypothetical. Training data is retrospective and often assembled months after events actually occur. The practical stakes are also high. If a considerable share of measured forecasting accuracy is contaminated, then improvements on forecasting benchmarks might be tracking training data coverage rather than an increase in reasoning ability.
