# Combined regret history

50 + 450 = 500 BO evaluations, excluding 100 initial Sobol evaluations.

`regret_history.csv` contains the 249 original saved estimates with global counts and provenance.
`regret_history_all_iterations.csv` contains one row per evaluation, with blank values where regret was not recorded.

Missing even evaluations: [50]. No interpolation or GP refitting was performed. The resume boundary is marked in the plot; a model refit there can change the diagnostic abruptly. The estimates are grid diagnostics, not measured true regret or an optimality certificate.

SHA-256 hashes confirm both source folders were unchanged. See `recovery_report.json`.
