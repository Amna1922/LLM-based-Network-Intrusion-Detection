# IDS Robustness

This project studies a TCP SYN flood evasion against a small LLM network intrusion detector.

## Hardware
The baseline work assumes a laptop CPU and a GTX 1650 Ti GPU.

## Python version
Python 3.11 is required.

## Create the venv
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## GPU note
If you use CUDA, install the CUDA build of PyTorch. For a GTX 1650 Ti, the cu121 wheel is a common choice. Check `nvidia-smi` before installing.

## Run commands
```bash
python -m src.prepare_data
python -m src.pcap_features
python -m pytest -q
python -m src.train_llm --quick
python -m src.predict_llm
python -m src.run_eval
```

Results are written under `results/` and docs under `docs/`.

## Regenerate docs
```bash
python tools/make_docs.py
```
