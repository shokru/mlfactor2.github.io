# Chapter 11 (Genetic Algorithms) — figure sources

**All 15 chapter figures are matplotlib data-plots, not hand-drawn diagrams.**
To actually change a figure (data, axes, colours), edit the generating script/code
below and re-run it; the PNG is regenerated. `Chapter_11_figures.pptx` embeds each
PNG on its own slide (with caption + source) so the images can be annotated, laid
out, or handed to the publisher, but the raster image itself is not vector-editable.

| Figure | PNG file | Editable source |
|--------|----------|-----------------|
| 11.1  | fig_11_1_mini_ga.png           | `ga_mini_toy.py` |
| 11.2  | fig_11_2_selection.png         | `Chapter_11_Draft.md` §11.3.7 code block |
| 11.3  | fig_11_3_crossover.png         | `Chapter_11_Draft.md` §11.3.8 code block |
| 11.4  | fig_11_4_param_sensitivity.png | `ga_mini_toy.py` |
| 11.5  | fig_11_5_portfolio.png         | `ga_book_demos.py` (`demo_portfolio`) |
| 11.6  | fig_11_6_gp_alpha.png          | `ga_book_demos.py` (`demo_gp`) |
| 11.7  | fig_11_7_nsga2.png             | `ga_book_demos.py` (`demo_nsga2`) |
| 11.8  | fig_11_8_cumulative_ic.png     | `ga_real_data_plots_v3.py` |
| 11.9  | fig_11_9_feature_frequency.png | `ga_real_data_plots_v3.py` |
| 11.10 | fig_11_10_n_selected.png       | `ga_real_data_plots_v3.py` |
| 11.11 | fig_11_11_pareto_front.png     | `ga_multiobjective_fs.py` |
| 11.12 | fig_11_12_convergence.png      | `ga_xgb_tuning.py` |
| 11.13 | fig_11_13_walkforward.png      | `ga_repro_walkforward.py` |
| 11.14 | fig_11_14_decile.png           | `ga_reproductions.py` (repro2) |
| 11.15 | fig_11_15_pareto.png           | `ga_reproductions.py` (repro3) |

Note: PNG filenames keep the `fig_11_*` prefix (from before the 19->11 renumber);
the figure *labels/captions* are 11.x. The scripts read the book DB and the
walk-forward result CSVs in the parent `19_GA/` folder.

Not used in the chapter (brainstorm leftovers, no PPT): `ideas_chromosome_ex.png`,
`GA_algo_param_feature_Selection*.png`, `Ideas_charts_GA_for_book*.png`,
`Ideas_table_hyperparameters_for_book.png`, `fig_11_hp_convergence.png`,
`repro1_warmstart.png`.
