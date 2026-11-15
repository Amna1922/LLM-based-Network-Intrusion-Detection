# Decisions

- Qwen2.5-0.5B-Instruct was selected instead of larger models such as Qwen3-8B or LLaMA3-8B to keep the project lightweight and runnable on a laptop CPU.
- FP16 mixed precision was chosen for GPU runs rather than BF16 to match the GTX 1650 Ti profile and remain compatible with the broader tooling.
- Full GPU runs use micro-batch 4 with gradient accumulation 8; CPU runs use FP32, micro-batch 2, and accumulation 16.
- The project does not include certification or randomized smoothing because the task scope is a single TCP SYN flood attack with a binary DDoS target.
- The task is framed as binary DDoS classification rather than a multi-class network intrusion problem.
- The model uses the 7-feature set specified by the project instructions, excluding identifiers, constants, and invalid rows.
- The project uses its own PCAP flow extractor instead of CICFlowMeter so the feature pipeline remains fully offline and self-contained.
- The prompt template is a custom text-only classifier format with normalised feature values rounded to four decimals.
- The data uses a 70/10/20 split grouped by bidirectional session for PCAP data and flow id for CSV data, with a single seed value for reproducibility.
- The shortcut audit measures one-feature accuracy on the unscaled training partition and reports its source and row count. PCAP-primary preparation also reports each feature's raw median, minimum, and maximum by class.
- The full-run defaults use 4000 rows, 2 epochs, micro-batch 4, and accumulation 8 for the GTX 1650 Ti. The quick path uses 200 rows and two optimizer steps.
- One fixed seed is used. Training logs show per-step loss and an ETA.
- The CSV and PCAP sources differ and therefore the project tracks the domain gap explicitly instead of pretending the domains are identical.
