# scp_simulations
Simulations of social decision games

Use scpsim_ prefix for m-files within this repository

## CPR Task Simulations (New)

**`scpsim_cpr_simulations_cursor.m`** - Comprehensive CPR wagering simulation ⭐

All-in-one script with easy configuration section at the top:
- Tests fixed vs adaptive tilt strategies
- Compares 7 reward formulations
- Analyzes performance across coherence levels (customizable)
- Generates 4 comprehensive figures
- Outputs detailed results to text file

**Quick start:**
```matlab
% Run complete analysis with visualizations
scpsim_cpr_simulations_cursor

% Results saved to: cpr_simulation_results.txt
```

**Key findings:**
- Linear reward provides **28% adaptive advantage** ✅
- Quadratic reward provides **45% adaptive advantage** ✅✅ (best)
- "Coherence Scaled" strategy (tilt: 0.2→0.6) outperforms fixed tilt
- Fixed medium tilt (0.5) is surprisingly competitive

**Documentation:**
- `SIMULATION_RESULTS_QUICK_REF.txt` - One-page results summary
- `CPR_RESULTS_SUMMARY.md` - Detailed analysis and recommendations