# Experiment plan and reproducibility contract

## Research question

Can a 500M-parameter vision-language model provide strong multi-vector visual document retrieval, and how far can its model/index computation be reduced before quality degrades?

## Fixed rules

- Seed is 42 unless the experiment is explicitly a seed ablation.
- Train, validation, and test IDs are persisted before fine-tuning.
- All compared methods use the same queries, qrels, and candidate pages.
- Every result is generated as JSON from code and records config, model revision, seed, Git commit, precision, image resolution, quality, latency, VRAM, and index size.
- Evaluation queries never enter training.
- CUDA latency measurements include warm-up and `torch.cuda.synchronize()` around timed regions.

## Milestones

1. Environment and correctness: run the diagnostic; encode 5-20 pages and 3-5 queries; test manual MaxSim; save/reload embeddings.
2. Small evaluation: freeze 100 queries and 500-1000 pages; compare BM25, dense text, global visual, and ColSmol MaxSim.
3. Scale and capacity: cache embeddings; compare 256M and 500M on quality, latency, VRAM, throughput, and index size.
4. Fine-tuning: begin with a 100-example one-epoch gradient check, then 2k and 5k-20k LoRA runs; select on validation nDCG, not training loss.
5. Compression: evaluate 100%, 75%, 50%, 25%, and 12.5% retention using random, uniform, mean pooling, and k-means. Repeat random compression over three seeds.
6. Analysis: MaxSim heatmaps, retained-token visualizations, Pareto fronts, and 20-30 categorized failures.
7. Optional RAG: hold the generator constant and compare retrieval contexts. No frontend is required.

## Training objective

The current official `ColModelTrainingConfig` defaults to `ColbertLoss`, implemented in `colpali_engine.loss.late_interaction_losses`, and `ContrastiveTrainer` applies it to query/document outputs. Training code must import that official loss rather than reimplementing an assumed contrastive formula. Before the first GPU training run, record the installed source/version and confirm its batch-negative behavior in the run metadata.

## Exit criteria for milestone 1

- Diagnostic JSON records actual RTX/CUDA/model measurements.
- Manual MaxSim tests pass on known tensors.
- A matching query ranks a sensible page above distractors.
- Cache round-trip is exact.
- No result contains guessed GPU or embedding dimensions.
