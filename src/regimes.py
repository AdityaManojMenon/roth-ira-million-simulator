import os
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

print("""
    S&P 500 Returns for approximations for accurate model: https://www.slickcharts.com/sp500/returns
    Inflation for approximations for accurate model based on my economics knowledge and research from various sources
    There have been roughly 15 years of recession in past 100 years, 7 years of stagflation, and 3 years of 
    true depression cause by deflation and rest are normal economic growth.
    Based on research on past 100 years I have decided to split economic cycles into 4 regimes with the following 
    distributions:
    - Normal Economic Growth (75%) Inflation 1-4% std: +-15% Returns average: 10%
    - Recession (15%) Inflation -2-2% std: +-25% Returns average: 3%
    - Stagflation (7%) Inflation 5-12% std: +-15% Returns average: 1% 
    - Depression (3%) Inflation -10% to -2% std: +-25% Returns average: -35%
""")

REGIME_NAMES = ["Normal", "Recession", "Stagflation", "Depression"]
len_regimes = len(REGIME_NAMES)

print(f"""
We construct a 4×4 transition matrix based on historical economic patterns:

States: {REGIME_NAMES}

TRANSITION LOGIC (from economic history):

FROM NORMAL:
- 85% stay in Normal (expansions last ~8-10 years on average)
- 10% transition to Recession (business cycle downturn)
- 4% transition to Stagflation (supply shock, monetary policy failure)
- 1% transition to Depression (rare catastrophic event)

FROM RECESSION:
- 50% recover to Normal (recessions typically end in 1-2 years)
- 35% persist in Recession (deeper/longer recessions)
- 10% worsen to Stagflation (policy response causes inflation)
- 5% deteriorate to Depression (financial contagion, rare)

FROM STAGFLATION:
- 40% recover to Normal (inflation brought under control)
- 30% slip into Recession (tight monetary policy to fight inflation)
- 25% persist in Stagflation (inflation expectations anchored high)
- 5% collapse to Depression (economic policy failures compound)

FROM DEPRESSION:
- 10% recover directly to Normal (rare, requires massive intervention)
- 60% improve to Recession (gradual recovery begins)
- 5% shift to Stagflation (policy overcorrection)
- 25% persist in Depression (debt deflation spiral continues)

KEY INSIGHTS:
1. Diagonal elements (persistence): Regimes tend to continue
2. Normal is "sticky" (0.85 probability of staying)
3. Depression rarely goes straight to Normal (only 0.10)
4. Bad regimes tend to cluster (high persistence in Recession/Stagflation/Depression)
""")

TRANSITION_MATRIX = np.array([
#To: Normal Recession Stagflation Depression
    [0.85,    0.10,     0.04,       0.01],      # From: Normal
    [0.50,    0.35,     0.10,       0.05],      # From: Recession
    [0.40,    0.30,     0.25,       0.05],      # From: Stagflation
    [0.10,    0.60,     0.05,       0.25]       # From: Depression
])
print(TRANSITION_MATRIX)

for i, regime in enumerate(REGIME_NAMES):
    row_sum = TRANSITION_MATRIX[i].sum()
    checker = "True" if abs(row_sum - 1.0) < 1 else "False"
    print(f"{regime:>12}: {row_sum:.15f} {checker}")

# Method: Find eigenvector corresponding to eigenvalue 1
eigenvalues, eigenvectors = eig(TRANSITION_MATRIX.T)

# Find index of eigenvalue closest to 1
steady_state_idx = np.argmin(np.abs(eigenvalues - 1.0))
steady_state = np.real(eigenvectors[:, steady_state_idx])
steady_state = steady_state / steady_state.sum()  # Normalize

print("\nEigenvalues of P^T:")
for i, val in enumerate(eigenvalues):
    marker = " ← λ=1 (steady-state)" if i == steady_state_idx else ""
    print(f"  λ_{i+1} = {val.real:.6f} + {val.imag:.6f}i{marker}")

print("\nSteady-State Distribution π:")
print("\n  Regime         π_i    Expected %  Actual (Historical)")
print("  " + "-" * 60)
expected = [0.75, 0.15, 0.07, 0.03]  # Our historical estimates

print("\nVerification: π P = π")
verification = steady_state @ TRANSITION_MATRIX
print(f"  π                = {steady_state}")
print(f"  π P              = {verification}")
print(f"  Difference       = {np.abs(steady_state - verification)}")
print(f"  Max difference   = {np.max(np.abs(steady_state - verification)):.2e}")

# Method: Simulate Markov Chain
def simulate_markov_chain(initial_state, n_steps, transition_matrix):
    """
    Simulate a Markov chain for n_steps
    """
    sequence = [initial_state]
    current_state = initial_state

    
    for _ in range(n_steps - 1):
        current_idx = REGIME_NAMES.index(current_state)
        probs = transition_matrix[current_idx]
        next_idx = np.random.choice(len(REGIME_NAMES), p=probs)
        current_state = REGIME_NAMES[next_idx]
        sequence.append(current_state)
    
    return sequence

# Simulate one path
print("\nEXAMPLE: SIMULATING 40 YEARS")
print("--------------------------------")

n_years = 40
sequence = simulate_markov_chain('Normal', n_years, TRANSITION_MATRIX)

print(f"\nStarting state: Normal")
print(f"Simulating {n_years} years...\n")

# Print first 40 years
print("First 40 years:")
for year, state in enumerate(sequence[:40], start=1):
    print(f"  Year {year:2}: {state}")

# Count regime frequencies
regime_counts = {regime: sequence.count(regime) for regime in REGIME_NAMES}

for i, regime in enumerate(REGIME_NAMES):
    count = regime_counts[regime]
    freq = count / n_years
    ss = steady_state[i]
    print(f"  {regime:12}   {count:3}     {freq:5.1%}       {ss:5.1%}")


print("\nDEMONSTRATING CONVERGENCE")
print("-----------------------------")
print("\nComputing P^t for increasing powers t:")

powers = [1, 2, 5, 10, 20, 50, 100]
print("\nP^t[0,:] (first row, starting from Normal):")
print(f"{'t':<6} " + "  ".join([f"{regime:>10}" for regime in REGIME_NAMES]))
print("-" * 60)
# Compute P^t for increasing powers t
for t in powers:
    P_t = np.linalg.matrix_power(TRANSITION_MATRIX, t)
    row = P_t[0, :]
    print(f"{t:<6} " + "  ".join([f"{p:>10.6f}" for p in row]))

print(f"\nπ:     " + "  ".join([f"{p:>10.6f}" for p in steady_state]))

# Visualize output of the Transition Matrix, Steady-State Distribution, and a sample path
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Markov Regime Switching: Three Key Concepts', 
             fontsize=15, fontweight='bold')

# Plot 1: Transition Matrix
ax1 = axes[0]
im = ax1.imshow(TRANSITION_MATRIX, cmap='YlOrRd', vmin=0, vmax=1)
ax1.set_xticks(range(4))
ax1.set_yticks(range(4))
ax1.set_xticklabels(['Norm', 'Rec', 'Stag', 'Depr'], fontsize=9)
ax1.set_yticklabels(REGIME_NAMES, fontsize=9)
ax1.set_title('Transition\nProbabilities', fontweight='bold')

for i in range(4):
    for j in range(4):
        ax1.text(j, i, f'{TRANSITION_MATRIX[i,j]:.2f}',
                ha="center", va="center", 
                color='white' if TRANSITION_MATRIX[i,j] > 0.5 else 'black',
                fontsize=9, fontweight='bold')

plt.colorbar(im, ax=ax1, fraction=0.046)

# Plot 2: Steady-State
ax2 = axes[1]
colors = ['green', 'blue', 'orange', 'red']
bars = ax2.bar(range(4), steady_state, color=colors, alpha=0.7)
ax2.set_xticks(range(4))
ax2.set_xticklabels(REGIME_NAMES, rotation=45, ha='right')
ax2.set_ylabel('Probability')
ax2.set_title('Long-Run\nFrequencies', fontweight='bold')
ax2.set_ylim(0, 1)

for bar, val in zip(bars, steady_state):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val*100:.0f}%', ha='center', fontweight='bold')

# Plot 3: Sample Path
ax3 = axes[2]
seq = simulate_markov_chain('Normal', 40, TRANSITION_MATRIX)
regime_colors = {'Normal': 'green', 'Recession': 'blue',
                'Stagflation': 'orange', 'Depression': 'red'}

for year, regime in enumerate(seq):
    ax3.barh(0, 1, left=year, color=regime_colors[regime])

ax3.set_xlim(0, 40)
ax3.set_ylim(-0.5, 0.5)
ax3.set_xlabel('Year')
ax3.set_yticks([])
ax3.set_title('Sample 40-Year\nSequence', fontweight='bold')


legend = [Patch(facecolor=regime_colors[r], label=r[:3]) for r in REGIME_NAMES]
ax3.legend(handles=legend, loc='upper right', fontsize=8, ncol=2)


plt.tight_layout()

# Ensure the output directory exists and build a path relative to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(project_root, "data", "generated")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "markov_essential.png")

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n Essential visualization saved: {output_path}")
plt.show()
