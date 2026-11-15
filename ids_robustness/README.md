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
python -m pytest -q
python -m src.run_eval --quick
python -m src.run_eval
```

The quick run reuses an existing LoRA adapter when available, evaluates 100 flows, and uses two perturbation variants. Use `python -m src.run_eval --quick --retrain` to retrain for the smoke test. The full run uses the configured defaults and reuses the adapter unless `--retrain` is supplied.

Results are written under `results/` and docs under `docs/`. Place benign `.pcap` or `.pcapng` files in `../data/benign_pcaps/` to train and evaluate using PCAP flows. If that directory has no captures, the project uses the CSV-only fallback and reports the source mismatch.

## Regenerate docs
```bash
python tools/make_docs.py
```
