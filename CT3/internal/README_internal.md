## Analyzer (internal)

### Overview

`TTTModelAnalyzer` is an extension of the `TTTModel` class designed to provide detailed analysis and logging of the test-time training (TTT) process. This class captures various metrics and timings during the TTT process, which can be useful for debugging, performance evaluation, and research purposes.

### Features

- Query Analysis: Logs the input query and whether cached parameters were used (and the reused query raw text).
- Retrieval Information: Records the indices and scores of the retrieved samples from the knowledge base.
- Loss Tracking: Captures the loss values during the fine-tuning process.
- Timing Metrics: Measures the time taken for embedding generation, knowledge base retrieval, fine-tuning, and forward pass.
- JSON Logging: Saves the analysis data in a JSON file for further inspection.

### Usage

To use TTTModelAnalyzer, you need to enable the analysis flag in the configuration settings. Simply add the `--analysis` flag when running:

```bash
export PYTHONPATH=./
python eval/ct3_lm_eval.py \
  --model_path <path to model> \
  --task <task name> \
  --output_path <path to results> \
  --ttt \
  --knowledge_base_path <path to knowledge base> \
  --num_ttt_samples <number of ttt training samples> \
  --local_adapter_managing \
  --local_adapter_manager_size 4 \
  --local_adapter_manager_threshold 0.4 \
  --local_adapter_manager_strategy lfu \
  --analysis
```

### Example Analysis Data

Below is an example of the JSON file saved by TTTModelAnalyzer:

```json

[
    {
        "query": "What is the capital of France?",
        "used_cache": false,
        "cache_query": null,
        "retrieved_indices": [
            12345,
            67890,
            23456,
            78901
        ],
        "retrieved_scores": [
            0.95,
            0.85,
            0.75,
            0.65
        ],
        "losses": [
            2.123
            1.890,
            1.567,
            1.234,
        ],
        "timing": {
            "embedding": 0.05,
            "kb_retrieval": 0.10,
            "fine_tuning": 8.50,
            "forward_pass": 3.20,
            "total": ...
        }
    }
]

```

Explanation of Analysis Data:

- **query**: The input query (raw text).
- **used_cache**: A boolean indicating whether cached parameters were used (local adapter manager).
- **cache_query**: The query that matched the cache entry, if any.
- **retrieved_indices**: The indices of the retrieved samples from the knowledge base.
- **retrieved_scores**: The similarity scores of the retrieved samples.
- **losses**: The loss values recorded during the fine-tuning process.
- **timing**: The time taken for different stages of the process:
    - **embedding**: Time taken to generate the query embedding.
    - **kb_retrieval**: Time taken to retrieve samples from the knowledge base.
    - **fine_tuning**: Time taken for the fine-tuning process.
    - **forward_pass**: Time taken for the forward pass or generation.
    - **total**: Total time taken for the entire process.

This detailed analysis helps in understanding the performance and efficiency of the TTT process.
