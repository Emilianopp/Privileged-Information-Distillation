# Multi-Agent Examples for GEM

This directory contains multi-agent environment examples using GEM's MultiAgentEnv framework.

## TAU-BENCH Retail Integration

The `tau_bench_retail/` directory contains the official integration of TAU-BENCH Retail benchmark into GEM. TAU-BENCH evaluates tool-augmented LLM agents on realistic customer service tasks in a retail environment.

### Setup

1. Install the tau-bench fork (at `<repo-root>/tau-bench`):
```bash
cd tau-bench
git checkout dheeraj/local-sync
pip install -e .
```

2. Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

3. Run the evaluation:
```bash
cd tau_bench_retail
python run_eval.py --model gpt-4o --model-provider openai
```

### Directory Structure

```
multiagent/
└── tau_bench_retail/
    ├── tau_bench_env.py       # GEM MultiAgentEnv adapter (delegates to fork)
    ├── tau_bench_agent.py     # Agent wrapper (delegates to fork's agent_factory)
    ├── run_eval.py            # CLI runner (delegates to fork's run pipeline)
    └── README.md              # Full usage instructions
```

## Performance

TAU-bench Retail: **78/115 (67.8%)**

## Available Tools

16 customer service tools including order management, user identification, information retrieval, and support functions.