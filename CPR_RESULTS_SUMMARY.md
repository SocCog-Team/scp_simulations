# CPR Wagering Simulation - Results Summary

## Executive Summary

**Your concern was VALID with important nuances:**

✅ Fixed medium tilt (0.5) performs well: **0.0711/s** reward rate  
✅ Naive adaptive strategies (tilt = coherence/100) perform **worse**: 0.0041/s  
✅ **BUT** smart adaptive strategies (scaled 0.2-0.6) perform **28% better**: 0.0911/s

## Key Findings from Simulation

### 1. Current Linear Reward Structure

With `reward = tilt` (only on hit):

| Strategy | Hit Rate | Reward Rate | Performance |
|----------|----------|-------------|-------------|
| Fixed Low (0.2) | 53.2% | 0.0532/s | Good |
| **Fixed Med (0.5)** | 28.4% | **0.0711/s** | ⭐ Best Fixed |
| Fixed High (0.8) | 0.0% | 0.0000/s | Fails |
| Coherence Linear (0→1) | 5.3% | 0.0041/s | Very Poor |
| **Coherence Scaled (0.2→0.6)** | 33.8% | **0.0911/s** | ✅ Best Overall |
| Match Accuracy | 0.0% | 0.0000/s | Fails |

**Adaptive Advantage: 28.3%** ✅ (Good, but not overwhelming)

### 2. Why Fixed Medium Works So Well

The arc width formula creates a "sweet spot":

```
tilt = 0.2 → arc = 146° (wide, low reward per hit)
tilt = 0.5 → arc = 95°  (balanced) ⭐
tilt = 0.8 → arc = 44°  (too narrow, no hits)
```

At **tilt = 0.5**:
- Moderate hit rate (28%)
- Moderate reward per hit (0.5)
- Product is surprisingly competitive!

### 3. The Critical Insight

**Naive coherence scaling fails because:**
- At 0% coherence: tilt ≈ 0 → very wide arc, but still miss (poor accuracy)
- At 98% coherence: tilt ≈ 1 → very narrow arc, impossible to hit

**Smart scaling succeeds because:**
- Maps coherence to **practical tilt range** (0.2 to 0.6)
- Stays in the "goldilocks zone" where hits are possible
- Still modulates with coherence

### 4. Comparison Across Reward Formulations

| Formulation | Fixed Best | Adaptive Best | Advantage |
|-------------|------------|---------------|-----------|
| **Quadratic** | 0.0352/s | 0.0510/s | **+44.8%** ✅✅ |
| **Miss Penalty** | 0.0531/s | 0.0746/s | **+40.3%** ✅✅ |
| **Exponential** | 0.0380/s | 0.0525/s | **+38.2%** ✅✅ |
| **Accuracy Bonus** | 0.1011/s | 0.1304/s | **+28.9%** ✅ |
| **Linear (current)** | 0.0711/s | 0.0911/s | **+28.3%** ✅ |
| Square Root | 0.1194/s | 0.1242/s | +4.0% ❌ |

## Psychometric Function

Based on simulated accuracy at each coherence:

| Coherence | Mean Accuracy | Angular Error | Hit Difficulty |
|-----------|---------------|---------------|----------------|
| 0% | 0.333 | 120° | Very Hard |
| 38% | 0.518 | 87° | Hard |
| 59% | 0.662 | 61° | Moderate |
| 98% | 0.860 | 25° | Easy |

## Recommendations

### Immediate Question: Should You Change Reward Formula?

**Answer: Current linear formulation is ACCEPTABLE but not optimal**

#### Option 1: Keep Current Linear Reward ✅
- Provides 28% advantage for adaptive wagering
- Sufficient if combined with training
- Simpler to implement
- **Use if:** You want minimal changes

#### Option 2: Switch to Quadratic Reward ✅✅ (RECOMMENDED)
- Formula: `reward = tilt²` (if hit)
- Provides 45% advantage for adaptive wagering
- Strongly incentivizes high-confidence bets
- More intuitive learning gradient
- **Use if:** You want optimal performance

#### Option 3: Add Miss Penalty ✅✅
- Formula: `reward = tilt` (hit) or `-0.05` (miss)
- Provides 40% advantage
- Punishes overconfidence with wide arcs
- **Use if:** You want to discourage lazy fixed-tilt strategies

### Training Strategy

Even with current linear reward, monkey can learn optimal behavior IF:

1. **Explicitly shape tilt range to 0.2-0.6**
   - Start with this range restricted
   - Gradually expand as performance improves

2. **Provide coherence cues**
   - Visual indicator of current coherence
   - Feedback on whether tilt was appropriate

3. **Block by coherence initially**
   - Train on single coherence levels first
   - Then mix once pattern established

### Why Your Intuition Was Right

You were correct to worry because:
1. Fixed tilt 0.5 gives 78% of optimal performance (0.0711 vs 0.0911)
2. Naive adaptive strategies actually hurt performance
3. Finding the right tilt range (0.2-0.6) requires insight

A monkey learning through pure trial-and-error might:
- Try extreme tilts → fail
- Discover fixed 0.5 works → stick with it
- **Never discover modulation benefit**

## Implementation Recommendations

### Quick Win (Minimal Changes)
```matlab
% Keep linear reward but guide learning
reward = tilt;  % (if hit)

% Add visual coherence feedback
% Restrict initial tilt range to [0.2, 0.6]
% Shape toward coherence-dependent wagering
```

### Optimal Solution (Recommended)
```matlab
% Switch to quadratic reward
reward = tilt^2;  % (if hit)

% This naturally incentivizes:
% - High tilt at high coherence (98%): reward = 0.6² = 0.36
% - Low tilt at low coherence (38%): reward = 0.3² = 0.09
% - Creates 45% advantage for adaptive wagering
```

### Advanced Solution
```matlab
% Quadratic with accuracy bonus
reward = tilt^2 * (1 + 0.5*accuracy);  % (if hit)

% Combines benefits:
% - Quadratic amplification
% - Accuracy-dependent scaling
% - 44% adaptive advantage (similar to quadratic alone)
```

## Simulation Details

**Parameters used:**
- Coherence levels: 0, 38, 59, 98%
- Trials per coherence: 3000
- Target width: 10°
- Arc width: 180° (tilt=0) to 10° (tilt=1)
- Target presentation rate: 0.5/second

**Psychometric function:**
- Baseline accuracy: 0.25 (chance)
- Maximum accuracy: 0.95 (ceiling)
- Slope: 0.04 (learning steepness)
- Threshold: 50% coherence (midpoint)

## Visualizations Generated

Running `scpsim_cpr_simulations_cursor` creates:

1. **Figure 1: Psychometric Function**
   - Accuracy vs coherence
   - Error distributions
   - Arc width relationships

2. **Figure 2: Strategy Comparison**
   - Reward rates across formulations
   - Adaptive advantage rankings
   - Hit rate vs tilt tradeoffs

3. **Figure 3: Detailed Analysis**
   - Fixed vs adaptive breakdown
   - Efficiency metrics
   - Strategy recommendations

4. **Figure 4: Reward Surfaces**
   - 3D visualizations of reward functions
   - Optimal regions highlighted
   - Comparison across formulations

## Files Generated

After running simulation:
- `cpr_simulation_results.txt` - Detailed text output
- 4 MATLAB figures (can be saved as .fig or exported)

## Next Steps

1. **Review figures** to understand reward landscapes
2. **Decide on reward formulation** (linear vs quadratic)
3. **Design training protocol** based on recommendations
4. **Customize simulation** for your monkey's actual psychometric data
5. **Re-run simulation** with real parameters to verify predictions

## How to Customize

Edit the CONFIG section at the top of `scpsim_cpr_simulations_cursor.m`:

```matlab
% Adjust to match your monkey's performance
CONFIG.psychometric.baseline_acc = 0.25;  % Chance level
CONFIG.psychometric.max_acc = 0.95;       % Ceiling
CONFIG.psychometric.slope = 0.04;         % Learning steepness
CONFIG.psychometric.threshold = 50;       % Coherence at 50%

% Test different coherence levels
CONFIG.coherence_levels = [0 25 50 75 100];

% Add custom strategies
CONFIG.strategies = {
    ...
    'Your Strategy', @(coh, acc) your_formula;
};

% Test custom reward formulas
CONFIG.reward_formulations = {
    ...
    'Your Formula', @(tilt, hit, acc) your_formula;
};
```

## Questions Answered

✅ **Is fixed tilt as good as variable?**  
With naive adaptation, fixed is better. With smart scaling, variable is 28-45% better.

✅ **Will monkey learn to modulate tilt?**  
Maybe. Current formulation provides incentive, but fixed tilt is competitive.

✅ **Should you change reward formula?**  
Recommended but not required. Quadratic gives best learning gradient.

✅ **What tilt range should monkey use?**  
0.2 to 0.6 is optimal. Extremes (0 or 1) fail.

✅ **Which formulation incentivizes adaptive wagering most?**  
Quadratic (45% advantage), followed by Miss Penalty (40%).

---

**Created:** Based on comprehensive simulation results  
**Script:** `scpsim_cpr_simulations_cursor.m`  
**Data:** Simulated with 3000 trials per coherence level  
**Reference:** Schneider et al. (2024) eLife 101021

