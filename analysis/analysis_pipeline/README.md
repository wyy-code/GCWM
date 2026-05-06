# Analysis Pipeline

This folder contains a complete pipeline for turning benchmark results + mechanism tables
(step/pair/family/layer outputs from your bridge/grad analyzers) into:

- derived CSV tables for analysis
- joined step-level and pair-level datasets
- summary tables (explanation ranking, top harmful pairs)
- publication-style figures

## Files

- `common.py`: shared helpers
- `prepare_downstream.py`: derive downstream step-level and pair-level targets from benchmark results
- `merge_mechanism_tables.py`: normalize and merge bridge/grad mechanism tables
- `build_analysis_dataset.py`: join downstream and mechanism tables, build summary CSVs
- `plot_analysis_report.py`: generate figures
- `run_all.py`: run the full pipeline end-to-end
- `config_example.yaml`: example config template

## Expected inputs

You should provide these files in the config:

- `step_bridge_summary.csv`
- `pair_bridge_agg.csv`
- `family_agg.csv`
- `top_geometry_conflict_layers.csv`
- `step_grad_summary.csv`
- `pair_grad_agg.csv`
- `top_negative_grad_cos_layers.csv`
- benchmark results CSV (for example `mmlu_pro_all_results.csv`)

## Run

```bash
python run_all.py --config /path/to/config.yaml
```

## Main outputs

Under `output_root` the pipeline writes:

- `raw/`: copies of the raw input CSVs
- `derived/step_downstream_metrics.csv`
- `derived/pair_downstream_drop.csv`
- `derived/step_mechanism.csv`
- `derived/pair_mechanism.csv`
- `derived/family_metric_by_step.csv`
- `derived/family_metric_matrix.csv`
- `derived/step_mechanism_join.csv`
- `derived/pair_mechanism_join.csv`
- `derived/step_explanation_ranking.csv`
- `derived/top_harmful_pairs.csv`
- `figs/fig1_step_master.png`
- `figs/fig2_selective_forgetting_vs_compatibility.png`
- `figs/fig3_taskpair_heatmaps.png`
- `figs/fig4_family_metric_heatmap.png`
- `figs/fig5_global_vs_local_geometry.png`
- `figs/fig6_top_layer_cases.png`
- `figs/fig7_step_explanation_heatmap.png`
- `figs/fig8_top_harmful_pairs.png`

## Notes

- If benchmark task names and mechanism table task names do not match exactly, use
  `task_alias_map` and/or `benchmark.task_column_map` in the config.
- If `task_order` is omitted, it is inferred from the benchmark `model` column.
- The pipeline expects benchmark rows to be in continual step order.
