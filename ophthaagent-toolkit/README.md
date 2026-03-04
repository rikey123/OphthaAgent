# ophthaagent-toolkit

Core released code for the OphthaAgent project.

## Included Components

- `agents/`: ophthalmic tool modules and RAG/tool integration components
- `generator/`: trajectory generation and multi-turn orchestration
- `fundus_decision*.py`: core diagnostic decision logic variants
- `run_vqa.py`, `batch_run_vqa.py`: evaluation/inference entrypoints
- `calculate_metrics.py`: scoring utility

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_vqa.py --help
```

## Typical Entry Points

```bash
python run_vqa.py
python batch_run_vqa.py
python calculate_metrics.py
```

For project-level overview and method summary, see:

- `../README.md`
