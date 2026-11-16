# IDS Robustness

This project evaluates an offline TCP SYN-flood intrusion detector using PCAP flow
features, benign PCAP captures, and controlled packet perturbations. It does not
scan networks or send traffic. A missing Hugging Face model may be downloaded
when training or inference first loads it.

## Requirements and virtual environment

Use Python 3.11. From this project directory, open PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In Command Prompt, activate the environment with:

```cmd
.venv\Scripts\activate.bat
```

## Verify GPU support

Install a CUDA-enabled PyTorch build compatible with the installed NVIDIA
driver, then verify it from the activated environment:

```powershell
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The trainer automatically selects CUDA when PyTorch reports it available;
otherwise it uses CPU. GPU runs use FP16, micro-batch 4, and gradient
accumulation 8 by default.

## Inputs

Place the input data relative to the project as configured in `config.yaml`:

- `../data/FlowStatsfile_modified_eval.csv`
- `../data/CIC-DDoS-2019-SynFlood_modified.pcap`
- Optional benign `.pcap` or `.pcapng` files in `../data/benign_pcaps/`

When benign captures are present, PCAP flows are the primary training source.
CSV rows remain a separate baseline and are never mixed into PCAP training.

## Run

Run the quick end-to-end smoke test:

```powershell
python -m pytest -q
python -m src.run_eval --quick
```

Run the configured evaluation:

```powershell
python -m src.run_eval
```

Run or resume all ordered project tasks, including the ablations:

```powershell
python -m src.run_experiments
```

The experiment runner prepares data, performs the delay audit, runs baselines
and multi-seed statistics, completes Task 4 ablations, generates Tasks 5-6
analyses and Task 7 figures/tables, compares with the supplied paper values,
and writes the setup report. Existing stage outputs are reused. The three
pending Task 4 LoRA ablations require GPU training and may take several hours.
If Task 4 is interrupted while writing its summary, rerun the same command:
completed A1-A3 run caches are reused, and missing ablation rows are reported
without crashing.
To start at a later stage, for example after completing the experiments:

```powershell
python -m src.run_experiments --start-at task7-plots-tables
```

To run individual existing analyses:

```powershell
python -m src.delay_audit
python -m src.baselines
python -m src.feature_space
python -m src.analysis
python -m src.plots
python -m src.make_tables
python tools/make_setup_doc.py
```

## Outputs and documentation

Results are written under `results/`, including `results/figures/` and
`results/tables/`. Every generated table has CSV and LaTeX formats. Figures
are saved as 300 dpi PNG and PDF, with captions in
`results/figures/captions.md`. Missing experiments are marked as unavailable;
the project does not fabricate ablation results or historical loss curves.
New training runs save per-step `training_loss.csv` files inside their adapter
run folders; Fig. 10 uses the saved per-seed histories.

Generate the source guide and reproduction setup after code changes:

```powershell
python tools/make_docs.py
python tools/make_setup_doc.py
```

See [docs/SETUP.md](docs/SETUP.md) for reproduction details and
[docs/DECISIONS.md](docs/DECISIONS.md) for recorded design choices.
