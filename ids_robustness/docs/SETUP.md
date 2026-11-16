# Reproduction setup

Run these commands from the `ids_robustness` project directory in Windows PowerShell.

## Create and activate the environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In Command Prompt, activate the same environment with:

```cmd
.venv\Scripts\activate.bat
```

## Confirm CUDA

```powershell
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The CUDA-enabled PyTorch wheel is required for GPU training. The training code selects CUDA when PyTorch reports it available; otherwise it runs on CPU. No separate CUDA toolkit is used by the Python code.

## Run and resume the experiment pipeline

```powershell
python -m pytest -q
python -m src.run_experiments
```

The runner executes preparation, delay audit, baselines and multi-seed statistics, Task 4 ablations, Tasks 5-6 analyses, plots/tables, the base-paper comparison, and this setup document in order. Existing complete stage outputs are reused. Task 4 trains three separate LoRA ablations if their results are missing and can take several hours on a laptop GPU. It was not run while preparing these docs.

Resume from a particular stage:

```powershell
python -m src.run_experiments --start-at task4-ablation
python -m src.run_experiments --start-at task7-plots-tables
```

To run the main evaluation separately, use `python -m src.run_eval --quick` for the smoke test or `python -m src.run_eval` for the configured full evaluation.

## Inputs and outputs

- CSV baseline: `../data/FlowStatsfile_modified_eval.csv`
- Attack capture: `../data/CIC-DDoS-2019-SynFlood_modified.pcap`
- Benign captures: `../data/benign_pcaps`
- Result directory: `results`
- Model files may be downloaded from Hugging Face when absent. After model preparation, evaluation reads local PCAPs and writes local outputs; it does not scan or send network traffic.

## Recorded environment

- Python: `3.11.9`
- PyTorch: `2.5.1+cu121`
- CUDA runtime: `12.1`
- CUDA available during preparation: `True`
- GPUs: `NVIDIA GeForce GTX 1650 Ti`
- Git revision: `ef59f215b6009d86ccda3b5a3cca8c83c682b282`

LoRA runs with recorded training durations: 2; measured training duration median=5272.1 seconds, minimum=5256.4 seconds, maximum=5287.7 seconds. These measurements do not include the pending ablation runs.

The project seed and configured model/training settings are in `config.yaml`. `results/splits/` records split membership when data preparation has completed. `results/environment.json` records package and hardware metadata for that run.
Each new training run writes `training_loss.csv` inside its adapter run folder. Fig. 10 reads those per-seed histories; older runs without per-step logs remain unavailable.
