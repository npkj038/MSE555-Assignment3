"""
Q2 Pipeline: Trajectories, Optimization & Selecting K
Westfield Children's Centre – Assignment 3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── styling ──────────────────────────────────────────────────────────────────
COLORS = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0']
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 130,
})

# =============================================================================
# STEP 1 – Load & build cumulative trajectories
# =============================================================================
df = pd.read_csv('output\\scored_notes.csv')
df.columns = df.columns.str.strip()

# Each row is a session-transition score (session 1 = transition 1→2, etc.)
# Pivot to wide: clients × 11 transitions
wide = (df.sort_values(['client_id', 'session'])
          .pivot(index='client_id', columns='session', values='score'))
wide.columns = [f's{c}' for c in wide.columns]
wide = wide.fillna(0)

# Cumulative progress trajectory (sum over transitions 1..t)
cum = wide.cumsum(axis=1)
cum.columns = [f'cum_{i+1}' for i in range(cum.shape[1])]

clients = wide.index.tolist()
n_clients = len(clients)
T_MAX = 12
N_TRANS = 11  # transitions (session pairs)

print(f"Clients: {n_clients}, Transitions per client: {N_TRANS}")
print("Score distribution:\n", df['score'].value_counts().sort_index())


# =============================================================================
# STEP 2 – Compute t* for every client
# t*_i = earliest transition t where cumulative progress ≥ 90% of total
# Note: t* maps to a session where the plateau is confirmed, so we treat
# t* as the session number after which care can end (1-indexed, max = T_MAX=12)
# We define t* over sessions 1..12 by treating the cumulative at transition t
# as the cumulative sum up to session t+1.
# =============================================================================
def compute_t_star(cum_row, threshold=0.90):
    """
    Given a cumulative progress array (length 11, indexed 0-10),
    find the earliest index t (0-based) where cum[t] >= threshold * cum[-1].
    Return the session number = t + 2 (since transition t means between session t+1 and t+2).
    Capped at T_MAX=12.
    """
    total = cum_row[-1]
    if total == 0:
        return T_MAX  # no progress at all → runs to end
    for t, val in enumerate(cum_row):
        if val >= threshold * total:
            return min(t + 2, T_MAX)  # session where plateau confirmed
    return T_MAX

cum_arr = cum.values  # shape (80, 11)
t_stars = np.array([compute_t_star(cum_arr[i]) for i in range(n_clients)])
print(f"\nt* distribution:\n{pd.Series(t_stars).value_counts().sort_index()}")


# =============================================================================
# STEP 3 – K-means clustering on cumulative trajectories
# We evaluate K = 2, 3, 4 and select based on policy distinctiveness
# =============================================================================
scaler = StandardScaler()
X = scaler.fit_transform(cum_arr)

def run_kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
    labels = km.fit_predict(X)
    return labels

# Try K = 2, 3, 4
k_labels = {}
for k in [2, 3, 4]:
    k_labels[k] = run_kmeans(X, k)


# =============================================================================
# STEP 4 – Newsvendor optimization for each K
# For each cluster c, Q*_c = argmax_Q  F_c(Q) * (T_max - Q)
# where F_c(Q) = P(t* <= Q | cluster c)
# =============================================================================
def newsvendor_analysis(labels, t_stars, T_max=12):
    """
    Returns:
      results: dict cluster_id -> {Q_star, savings_curve, t_star_vals, Fc}
    """
    k = len(np.unique(labels))
    Q_range = np.arange(1, T_max + 1)
    results = {}
    for c in range(k):
        mask = labels == c
        t_c = t_stars[mask]
        n_c = len(t_c)
        savings_curve = []
        for Q in Q_range:
            Fc_Q = np.mean(t_c <= Q)          # empirical CDF at Q
            exp_savings = Fc_Q * (T_max - Q)
            savings_curve.append(exp_savings)
        savings_curve = np.array(savings_curve)
        Q_star_idx = np.argmax(savings_curve)
        Q_star = Q_range[Q_star_idx]
        results[c] = {
            'Q_star': Q_star,
            'savings_curve': savings_curve,
            't_star_vals': t_c,
            'n': n_c,
            'E_savings_at_Q_star': savings_curve[Q_star_idx],
        }
    return results


policy_results = {}
for k in [2, 3, 4]:
    policy_results[k] = newsvendor_analysis(k_labels[k], t_stars)


# =============================================================================
# STEP 5 – Select K
# Print policy summary for each K to justify selection
# =============================================================================
print("\n" + "="*60)
print("POLICY SUMMARY FOR K SELECTION")
print("="*60)
for k in [2, 3, 4]:
    res = policy_results[k]
    q_stars = [res[c]['Q_star'] for c in range(k)]
    savings = [res[c]['E_savings_at_Q_star'] for c in range(k)]
    sizes = [res[c]['n'] for c in range(k)]
    print(f"\nK={k}: Q* = {q_stars}, E[savings] = {[round(s,2) for s in savings]}, sizes = {sizes}")
    print(f"  Q* range = {max(q_stars)-min(q_stars)} sessions apart")

# ── CHOSEN K ─────────────────────────────────────────────────────────────────
K = 3
labels = k_labels[K]
res = policy_results[K]

# Assign readable cluster labels after inspecting trajectory shapes
# (will refine after plotting)
print(f"\n>>> Final K = {K}")


# =============================================================================
# STEP 6 – PLOTS for final K
# =============================================================================

# ── Helper: cluster sizes and colors ─────────────────────────────────────────
cluster_ids = sorted(np.unique(labels))
cluster_sizes = {c: np.sum(labels == c) for c in cluster_ids}

# --- PLOT A: Spaghetti plots -------------------------------------------------
fig, axes = plt.subplots(1, K, figsize=(5*K, 4.5), sharey=True)
session_axis = np.arange(1, T_MAX + 1)  # sessions 1–12

for c in cluster_ids:
    ax = axes[c]
    mask = labels == c
    c_cum = cum_arr[mask]         # shape (n_c, 11)
    c_cum_full = np.hstack([np.zeros((c_cum.shape[0], 1)), c_cum])  # prepend 0 at session 1

    # Individual trajectories
    for i in range(c_cum_full.shape[0]):
        ax.plot(session_axis, c_cum_full[i], color=COLORS[c], alpha=0.25, lw=1)

    # Cluster mean
    mean_traj = c_cum_full.mean(axis=0)
    ax.plot(session_axis, mean_traj, color=COLORS[c], lw=2.5,
            label=f'Cluster {c+1} mean', zorder=5)

    ax.set_title(f'Cluster {c+1}  (n={cluster_sizes[c]})', color=COLORS[c], fontweight='bold')
    ax.set_xlabel('Session')
    ax.set_xticks(session_axis)

axes[0].set_ylabel('Cumulative Progress Score')
fig.suptitle('Spaghetti Plot: Cumulative Progress Trajectories by Cluster', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output\\spaghetti_plots.png', bbox_inches='tight')
plt.close()
print("Saved: spaghetti_plots.png")


# --- PLOT 1: t* Distributions ------------------------------------------------
fig, axes = plt.subplots(1, K, figsize=(5*K, 4), sharey=False)
bins = np.arange(0.5, T_MAX + 1.5, 1)

for c in cluster_ids:
    ax = axes[c]
    t_c = res[c]['t_star_vals']
    ax.hist(t_c, bins=bins, color=COLORS[c], edgecolor='white', linewidth=0.8, alpha=0.85)
    ax.axvline(res[c]['Q_star'], color='black', linestyle='--', lw=1.5,
               label=f"Q*={res[c]['Q_star']}")
    ax.set_title(f'Cluster {c+1}  (n={cluster_sizes[c]})', color=COLORS[c], fontweight='bold')
    ax.set_xlabel('Stopping Session (t*)')
    ax.set_xticks(range(1, T_MAX + 1))
    ax.legend(fontsize=9)

axes[0].set_ylabel('Count of Clients')
fig.suptitle('Plot 1 — Distribution of t* (Stopping Points) by Cluster', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output\\plot1_t_star_distributions.png', bbox_inches='tight')
plt.close()
print("Saved: plot1_t_star_distributions.png")


# --- PLOT 2: Expected Savings vs Q ------------------------------------------
Q_range = np.arange(1, T_MAX + 1)
fig, ax = plt.subplots(figsize=(8, 5))

for c in cluster_ids:
    curve = res[c]['savings_curve']
    q_star = res[c]['Q_star']
    ax.plot(Q_range, curve, color=COLORS[c], lw=2.5, marker='o', markersize=4,
            label=f'Cluster {c+1} (Q*={q_star})')
    ax.axvline(q_star, color=COLORS[c], linestyle='--', lw=1.2, alpha=0.7)

ax.set_xlabel('Reassessment Session Q')
ax.set_ylabel('Expected Sessions Saved per Client')
ax.set_xticks(Q_range)
ax.set_title('Plot 2 — Expected Sessions Saved vs. Q by Cluster')
ax.legend()
plt.tight_layout()
plt.savefig('output\\plot2_expected_savings.png', bbox_inches='tight')
plt.close()
print("Saved: plot2_expected_savings.png")


# --- PLOT 3: Optimized Q* vs Mean Baseline ----------------------------------
# Baseline: round(mean t*) per cluster
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(K)
width = 0.35

opt_savings = []
base_savings = []
baseline_Qs = []

for c in cluster_ids:
    t_c = res[c]['t_star_vals']
    Q_star = res[c]['Q_star']
    Q_base = int(np.round(np.mean(t_c)))
    Q_base = max(1, min(Q_base, T_MAX))
    baseline_Qs.append(Q_base)

    Fc_opt = np.mean(t_c <= Q_star)
    opt_sav = Fc_opt * (T_MAX - Q_star)
    opt_savings.append(opt_sav)

    Fc_base = np.mean(t_c <= Q_base)
    base_sav = Fc_base * (T_MAX - Q_base)
    base_savings.append(base_sav)

bars1 = ax.bar(x - width/2, opt_savings, width, label='Optimized Q*',
               color=[COLORS[c] for c in cluster_ids], edgecolor='white')
bars2 = ax.bar(x + width/2, base_savings, width, label='Mean Baseline',
               color=[COLORS[c] for c in cluster_ids], alpha=0.45, edgecolor='white',
               hatch='//')

ax.set_xticks(x)
ax.set_xticklabels([f'Cluster {c+1}\n(Q*={res[c]["Q_star"]}, base={baseline_Qs[c]})'
                    for c in cluster_ids])
ax.set_ylabel('Expected Sessions Saved per Client')
ax.set_title('Plot 3 — Optimized Q* vs. Mean Heuristic Baseline')
ax.legend()

# Annotate bars
for bar in bars1:
    ax.annotate(f'{bar.get_height():.2f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.2f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('output\\plot3_optimized_vs_baseline.png', bbox_inches='tight')
plt.close()
print("Saved: plot3_optimized_vs_baseline.png")


# =============================================================================
# STEP 7 – Summary table (Q2e)
# =============================================================================
print("\n" + "="*60)
print("SUMMARY TABLE (Q2e)")
print("="*60)

total_clients = n_clients
total_opt_savings = 0
total_base_savings = 0

rows = []
for c in cluster_ids:
    t_c = res[c]['t_star_vals']
    n_c = res[c]['n']
    Q_star = res[c]['Q_star']
    E_saved = res[c]['E_savings_at_Q_star']
    pct_saved = E_saved / T_MAX * 100
    rows.append({
        'Cluster': c + 1,
        'Size': n_c,
        'Q*': Q_star,
        'E[saved/child]': round(E_saved, 2),
        '% of sessions saved': f"{pct_saved:.1f}%"
    })
    total_opt_savings += E_saved * n_c
    total_base_savings += base_savings[c] * n_c

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
print(f"\nOverall E[saved/child] = {total_opt_savings/total_clients:.2f} sessions")
print(f"Baseline E[saved/child] = {total_base_savings/total_clients:.2f} sessions")
print(f"Total gain from optimized policy: {total_opt_savings - total_base_savings:.1f} sessions across all clients")

# =============================================================================
# STEP 8 – Print key stats for answering written questions
# =============================================================================
print("\n" + "="*60)
print("CLUSTER PROFILE STATS (for Q2f writeup)")
print("="*60)
for c in cluster_ids:
    mask = labels == c
    c_wide = wide.values[mask]
    t_c = res[c]['t_star_vals']
    print(f"\nCluster {c+1} (n={cluster_sizes[c]}):")
    print(f"  Mean total score:  {c_wide.sum(axis=1).mean():.2f}")
    print(f"  Mean t*:           {t_c.mean():.2f}")
    print(f"  t* std:            {t_c.std():.2f}")
    print(f"  Q*:                {res[c]['Q_star']}")
    print(f"  E[saved/child]:    {res[c]['E_savings_at_Q_star']:.2f}")

    # Where does most progress happen? First half vs second half
    first_half = c_wide[:, :5].sum(axis=1).mean()
    second_half = c_wide[:, 5:].sum(axis=1).mean()
    print(f"  Avg score sessions 1-5:  {first_half:.2f}")
    print(f"  Avg score sessions 6-11: {second_half:.2f}")
    
    # Gen-AI was used in the completion of this code
