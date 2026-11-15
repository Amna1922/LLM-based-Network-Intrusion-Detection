# Decisions

- Qwen2.5-0.5B-Instruct was selected instead of larger models such as Qwen3-8B or LLaMA3-8B to keep the project lightweight and runnable on a laptop CPU.
- FP16 mixed precision was chosen for GPU runs rather than BF16 to match the GTX 1650 Ti profile and remain compatible with the broader tooling.
- The training configuration uses micro-batch 4 with gradient accumulation 8 because it is a practical CPU/GPU compromise for the smoke-test path.
- The project does not include certification or randomized smoothing because the task scope is a single TCP SYN flood attack with a binary DDoS target.
- The task is framed as binary DDoS classification rather than a multi-class network intrusion problem.
- The model uses the 7-feature set specified by the project instructions, excluding identifiers, constants, and invalid rows.
- The project uses its own PCAP flow extractor instead of CICFlowMeter so the feature pipeline remains fully offline and self-contained.
- The prompt template is a custom text-only classifier format with normalised feature values rounded to four decimals.
- The data uses a 70/10/20 split grouped by flow id and a single seed value for reproducibility.
- The default training schedule uses 3 epochs and one seed, while the smoke-test path is intentionally shorter.
- The CSV and PCAP sources differ and therefore the project tracks the domain gap explicitly instead of pretending the domains are identical.
