"""
Q3 – Intake characteristics by trajectory cluster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from q2 import clients, k_labels, policy_results

# ====================================================
# Q3a - Exploring Intake Features by Trajectory Type
# ====================================================

# =============================================================================
# SETUP – cluster assignments
# =============================================================================
K = 3
labels = k_labels[K]

features = pd.read_csv("data/client_features.csv")
features['cluster'] = pd.Series(dict(zip(clients, labels + 1))).reindex(features['client_id'].values).values
features = features.dropna(subset=['cluster'])
features['cluster'] = features['cluster'].astype(int)


COLORS = {1: '#2196F3', 2: '#E91E63', 3: '#4CAF50'}
CLABELS = {1: 'Cluster 1 (Q*=6)', 2: 'Cluster 2 (Q*=8)', 3: 'Cluster 3 (Q*=9)'}

# =============================================================================
# SUMMARY TABLES
# =============================================================================
print("INTAKE CHARACTERISTICS BY CLUSTER")
print("="*55)
print(features.groupby('cluster').agg(
    n=('client_id', 'count'),
    age_mean=('age_years', 'mean'),
    age_std=('age_years', 'std'),
    complexity_mean=('complexity_score', 'mean'),
    complexity_std=('complexity_score', 'std')
).round(2).to_string())

print("\nReferral reason proportions:")
print(pd.crosstab(features['cluster'], features['referral_reason'], normalize='index').round(2).to_string())

print("\nGender proportions:")
print(pd.crosstab(features['cluster'], features['gender'], normalize='index').round(2).to_string())


# =============================================================================
# PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Intake characteristics by trajectory cluster", fontsize=13, fontweight='normal')

cluster_ids = [1, 2, 3]
positions = [1, 2, 3]

# --- Shared jittered dotplot helper --------------------------------------
def jitter_plot(ax, feature, ylabel, title):
    for i, c in enumerate(cluster_ids):
        vals = features.loc[features['cluster'] == c, feature].dropna()
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(positions[i] + jitter, vals, color=COLORS[c],
                   alpha=0.6, s=28, zorder=3, linewidths=0)
        ax.hlines(vals.median(), positions[i] - 0.25, positions[i] + 0.25,
                  colors=COLORS[c], linewidth=2.5, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([CLABELS[c] for c in cluster_ids], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight='normal')
    ax.spines[['top', 'right']].set_visible(False)

jitter_plot(axes[0, 0], 'complexity_score', 'Complexity score',      'Complexity distribution')
jitter_plot(axes[0, 1], 'age_years',        'Age at intake (years)', 'Age distribution')

# --- Referral reason: grouped counts bar ---------------------------------
ax = axes[1, 0]
reasons = sorted(features['referral_reason'].dropna().unique())
reason_colors = ['#B5D4F4', '#9FE1CB', '#F5C4B3', '#D3D1C7']
x = np.arange(len(cluster_ids))
n_reasons = len(reasons)
width = 0.8 / n_reasons
for j, reason in enumerate(reasons):
    counts = [features[(features['cluster'] == c) &
                        (features['referral_reason'] == reason)].shape[0]
              for c in cluster_ids]
    offset = (j - (n_reasons - 1) / 2) * width
    ax.bar(x + offset, counts, width=width * 0.9,
           color=reason_colors[j % len(reason_colors)],
           label=reason, linewidth=0.5, edgecolor='#aaa')
ax.set_xticks(x)
ax.set_xticklabels([CLABELS[c] for c in cluster_ids], fontsize=9)
ax.set_ylabel("Count")
ax.set_title("Referral reason", fontsize=11, fontweight='normal')
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(fontsize=8, frameon=False)
ax.spines[['top', 'right']].set_visible(False)

# --- Gender: grouped counts bar ------------------------------------------
ax = axes[1, 1]
genders = ['M', 'F']
gender_colors = ['#B5D4F4', '#F4C0D1']
x = np.arange(len(cluster_ids))
width = 0.35
for j, g in enumerate(genders):
    counts = [(features[features['cluster'] == c]['gender'] == g).sum()
               for c in cluster_ids]
    ax.bar(x + (j - 0.5) * width, counts, width=width,
           color=gender_colors[j], label=g,
           linewidth=0.5, edgecolor='#aaa')
ax.set_xticks(x)
ax.set_xticklabels([CLABELS[c] for c in cluster_ids], fontsize=9)
ax.set_ylabel("Count")
ax.set_title("Gender", fontsize=11, fontweight='normal')
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(fontsize=8, frameon=False)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("output\\q3a_cluster_intake_features.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/q3a_cluster_intake_features.png")

# --- Pairplots ------------------------------------------

import seaborn as sns

plot_df = features[['age_years', 'complexity_score', 'cluster']].copy()
plot_df['cluster'] = plot_df['cluster'].map(CLABELS)
palette = {v: COLORS[k] for k, v in CLABELS.items()}

g = sns.JointGrid(data=plot_df, x='age_years', y='complexity_score', height=7)

# Main scatterplot
sns.scatterplot(
    data=plot_df, x='age_years', y='complexity_score',
    hue='cluster', palette=palette,
    alpha=0.7, s=50, ax=g.ax_joint, linewidth=0
)

# Marginal KDEs
sns.kdeplot(
    data=plot_df, x='age_years',
    hue='cluster', palette=palette,
    fill=True, alpha=0.3, linewidth=1,
    ax=g.ax_marg_x, legend=False
)
sns.kdeplot(
    data=plot_df, y='complexity_score',
    hue='cluster', palette=palette,
    fill=True, alpha=0.3, linewidth=1,
    ax=g.ax_marg_y, legend=False
)

g.ax_joint.set_xlabel("Age at intake (years)", fontsize=11)
g.ax_joint.set_ylabel("Complexity score", fontsize=11)
g.ax_joint.legend(title='Cluster', fontsize=9, title_fontsize=9, frameon=False)
g.figure.suptitle("Age vs complexity by cluster", fontsize=13,
                   fontweight='normal', y=1.01)

g.figure.savefig("output\\q3a_joint_age_complexity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/q3a_joint_age_complexity.png")


# ====================================================
# Q3b - Train a Model to Predict Trajectory Group
# ====================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SEED = 42

# =============================================================================
# SETUP
# =============================================================================
# Encode referral reason
le_referral = LabelEncoder()
features['referral_enc'] = le_referral.fit_transform(features['referral_reason'])

FEATURE_COLS = ['complexity_score', 'referral_enc']
X = features[FEATURE_COLS].values
y = features['cluster'].values

# =============================================================================
# TRAIN / TEST SPLIT  (stratified 80/20)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# =============================================================================
# MODEL 1 – Multinomial Logistic Regression
# =============================================================================
lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=SEED)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
acc_lr    = accuracy_score(y_test, y_pred_lr)
cm_lr     = confusion_matrix(y_test, y_pred_lr, labels=[1, 2, 3])

# =============================================================================
# MODEL 2 – Random Forest
# =============================================================================
rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf    = accuracy_score(y_test, y_pred_rf)
cm_rf     = confusion_matrix(y_test, y_pred_rf, labels=[1, 2, 3])

# =============================================================================
# CROSS-VALIDATED ACCURACY (5-fold stratified)
# =============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_lr = cross_val_score(lr, scaler.fit_transform(X), y, cv=cv, scoring='accuracy')
cv_rf = cross_val_score(rf, X,                       y, cv=cv, scoring='accuracy')

# =============================================================================
# PRINT RESULTS
# =============================================================================
print()
print("=" * 55)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 55)
for name, acc, cv_scores, y_pred in [
    ("Logistic Regression", acc_lr, cv_lr, y_pred_lr),
    ("Random Forest",       acc_rf, cv_rf, y_pred_rf),
]:
    print(f"\n{name}")
    print(f"  Test accuracy      : {acc:.3f}")
    print(f"  5-fold CV accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(classification_report(y_test, y_pred,
                                target_names=['C1 (Q*=6)', 'C2 (Q*=8)', 'C3 (Q*=9)'],
                                digits=2))

# =============================================================================
# PLOTS – confusion matrices + feature importance
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Trajectory classifier evaluation", fontsize=13, fontweight='normal')

cluster_tick_labels = ['C1 (Q*=6)', 'C2 (Q*=8)', 'C3 (Q*=9)']

def plot_cm(ax, cm, title, acc):
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(cluster_tick_labels, fontsize=9)
    ax.set_yticklabels(cluster_tick_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"{title}\nTest acc = {acc:.2f}", fontsize=11, fontweight='normal')
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > thresh else '#2C2C2A')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plot_cm(axes[0], cm_lr, "Logistic regression", acc_lr)
plot_cm(axes[1], cm_rf, "Random forest",       acc_rf)

# Feature importance
ax = axes[2]
feat_labels = ['complexity', 'referral reason']
importances = rf.feature_importances_
order = np.argsort(importances)
bar_colors = ['#185FA5' if importances[i] == importances.max() else '#B5D4F4' for i in order]
ax.barh([feat_labels[i] for i in order], importances[order],
        color=bar_colors, linewidth=0.5, edgecolor='#888780', height=0.4)
ax.set_xlabel("Feature importance (RF)", fontsize=10)
ax.set_title("Feature importance", fontsize=11, fontweight='normal')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("output\\q3b_classifiers.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/q3b_classifiers.png")


# ====================================================
# Q3c - Predict Trajectory Mix for the Waitlist
# ====================================================

res = policy_results[K]
T_MAX   = 12

# =============================================================================
# STEP 1 – Rebuild classifier on full historical data
# =============================================================================
X = features[FEATURE_COLS].values
y = features['cluster'].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=SEED)
lr.fit(X_sc, y)

# =============================================================================
# STEP 2 – Load waitlist and predict trajectory group
# =============================================================================
waitlist = pd.read_csv("data/waitlist.csv")
waitlist['referral_enc'] = le_referral.transform(waitlist['referral_reason'])

X_wait    = waitlist[FEATURE_COLS].values
X_wait_sc = scaler.transform(X_wait)

waitlist['predicted_cluster'] = lr.predict(X_wait_sc)
waitlist['cluster_label']     = waitlist['predicted_cluster'].map(CLABELS)

print("Waitlist predicted trajectory mix:")
print(waitlist['cluster_label'].value_counts().to_string())
print(f"\nTotal waitlist clients: {len(waitlist)}")

# =============================================================================
# STEP 3 – Expected sessions delivered per cluster under policy
# E[delivered | Q*_c] = Q*_c * (1 - F_c(Q*_c)) + E[t* | t* <= Q*_c] * F_c(Q*_c)
# which simplifies to: T_MAX - E[savings | Q*_c]
# pull directly from res which already has this computed
# =============================================================================
# E[delivered] = T_MAX - E[savings at Q*]
e_delivered = {}
q_stars     = {}
for c in range(K):
    q_stars[c + 1]     = res[c]['Q_star']
    e_delivered[c + 1] = T_MAX - res[c]['E_savings_at_Q_star']

print("\nCluster policy parameters:")
for c in [1, 2, 3]:
    print(f"  {CLABELS[c]}: Q*={q_stars[c]}, "
          f"E[delivered]={e_delivered[c]:.2f}, "
          f"E[saved]={T_MAX - e_delivered[c]:.2f}")

# =============================================================================
# STEP 4 – Total capacity estimate
# =============================================================================
waitlist['e_sessions_policy']   = waitlist['predicted_cluster'].map(e_delivered)
waitlist['e_sessions_baseline'] = T_MAX

total_policy   = waitlist['e_sessions_policy'].sum()
total_baseline = waitlist['e_sessions_baseline'].sum()
total_saved    = total_baseline - total_policy
n_wait         = len(waitlist)

print("\n" + "="*55)
print("WAITLIST CAPACITY ESTIMATE")
print("="*55)
print(f"  Waitlist size                  : {n_wait} clients")
print(f"  Baseline total sessions (Tmax) : {total_baseline:.0f}")
print(f"  Policy total sessions          : {total_policy:.1f}")
print(f"  Total sessions saved           : {total_saved:.1f}")
print(f"  Savings per client             : {total_saved/n_wait:.2f}")
print(f"  Capacity reduction             : {total_saved/total_baseline*100:.1f}%")

# Per-cluster contribution to savings
print("\nSavings breakdown by predicted cluster:")
cluster_summary = (waitlist.groupby('cluster_label')
                            .agg(n=('client_id', 'count'),
                                 total_policy_sessions=('e_sessions_policy', 'sum'))
                            .reset_index())
cluster_summary['baseline_sessions'] = cluster_summary['n'] * T_MAX
cluster_summary['sessions_saved']    = (cluster_summary['baseline_sessions']
                                        - cluster_summary['total_policy_sessions'])
cluster_summary['saved_per_client']  = (cluster_summary['sessions_saved']
                                        / cluster_summary['n']).round(2)
print(cluster_summary.to_string(index=False))

# =============================================================================
# STEP 5 – Plots
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Waitlist capacity under differentiated reassessment policy",
             fontsize=13, fontweight='normal')

# --- Predicted cluster mix (bar) -----------------------------------------
ax = axes[0]
counts = waitlist['predicted_cluster'].value_counts().sort_index()
bar_colors = [COLORS[c] for c in counts.index]
ax.bar([CLABELS[c] for c in counts.index], counts.values,
       color=bar_colors, width=0.5, linewidth=0.5, edgecolor='#aaa')
ax.set_ylabel("Number of clients")
ax.set_title("Predicted trajectory mix\n(waitlist)", fontsize=11, fontweight='normal')
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_xticklabels([CLABELS[c] for c in counts.index], fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

# --- Expected sessions per cluster (grouped: baseline vs policy) ----------
ax = axes[1]
x = np.arange(K)
width = 0.35
baseline_vals = [T_MAX] * K
policy_vals   = [e_delivered[c] for c in [1, 2, 3]]
ax.bar(x - width/2, baseline_vals, width, label='Baseline (Tmax)',
       color='#D3D1C7', linewidth=0.5, edgecolor='#aaa')
ax.bar(x + width/2, policy_vals,   width, label='Policy (E[delivered])',
       color=[COLORS[c] for c in [1, 2, 3]], linewidth=0.5, edgecolor='#aaa')
ax.set_xticks(x)
ax.set_xticklabels([CLABELS[c] for c in [1, 2, 3]], fontsize=8)
ax.set_ylabel("Expected sessions per client")
ax.set_title("Baseline vs policy\nsessions per cluster", fontsize=11, fontweight='normal')
ax.legend(fontsize=8, frameon=False)
ax.spines[['top', 'right']].set_visible(False)

# --- Total sessions saved by cluster -------------------------------------
ax = axes[2]
saved_by_cluster = cluster_summary.set_index('cluster_label')['sessions_saved']
bar_colors2 = [COLORS[c] for c in [1, 2, 3]
               if CLABELS[c] in saved_by_cluster.index]
ax.bar(saved_by_cluster.index, saved_by_cluster.values,
       color=bar_colors2, width=0.5, linewidth=0.5, edgecolor='#aaa')
ax.set_ylabel("Total sessions saved")
ax.set_title("Capacity savings by\ntrajectory group", fontsize=11, fontweight='normal')
ax.set_xticklabels(saved_by_cluster.index, fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("output\\q3c_waitlist_capacity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/q3c_waitlist_capacity.png")
