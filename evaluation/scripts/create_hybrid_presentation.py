#!/usr/bin/env python3
"""
Create presentation-ready visualizations for the 7-model hybrid evaluation.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Load results
results_path = Path(__file__).parent.parent / "results" / "hybrid_evaluation_results.json"
with open(results_path) as f:
    data = json.load(f)

results = data["results"]

# Sort by weighted score descending
results = sorted(results, key=lambda x: x["weighted_score"], reverse=True)

# Extract data
models = [r["model"] for r in results]
scores = [r["weighted_score"] * 100 for r in results]
use_case_acc = [r["field_accuracy"]["use_case"] * 100 for r in results]
user_count_acc = [r["field_accuracy"]["user_count"] * 100 for r in results]
priority_acc = [r["field_accuracy"]["priority"] * 100 for r in results]
hardware_acc = [r["field_accuracy"]["hardware"] * 100 for r in results]
json_validity = [r["json_validity"] * 100 for r in results]
latencies = [r["latency_ms"]["avg"] for r in results]

# Model sizes (GB)
model_sizes = {
    "tinyllama": 0.64,
    "phi3:mini": 2.2,
    "mistral:7b": 4.4,
    "qwen2.5:7b": 4.7,
    "llama3.1:8b": 4.9,
    "gemma2:9b": 5.4,
    "phi3:medium": 7.9,
}
sizes = [model_sizes.get(m, 0) for m in models]

# Colors
colors = ['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#e74c3c', '#c0392b']
bar_colors = ['#2ecc71' if s > 90 else '#3498db' if s > 80 else '#f39c12' if s > 70 else '#e74c3c' for s in scores]

output_dir = Path(__file__).parent.parent / "results" / "presentation"
output_dir.mkdir(exist_ok=True)

# ============================================================================
# PLOT 1: Executive Summary - Main Ranking
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🏆 Compass LLM Evaluation: 7 Models × 400 Test Cases', fontsize=18, fontweight='bold', y=0.98)

# 1a. Overall Weighted Score
ax1 = axes[0, 0]
bars = ax1.barh(models[::-1], scores[::-1], color=bar_colors[::-1], edgecolor='white', linewidth=2)
ax1.set_xlabel('Weighted Score (%)', fontweight='bold')
ax1.set_title('📊 Overall Weighted Score (Hybrid Scoring)', fontweight='bold', fontsize=13)
ax1.set_xlim(0, 100)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, scores[::-1])):
    ax1.text(score + 1, bar.get_y() + bar.get_height()/2, f'{score:.1f}%', 
             va='center', fontweight='bold', fontsize=11)

# Add medals
medals = ['🥇', '🥈', '🥉', '4', '5', '6', '7']
for i, medal in enumerate(medals[::-1]):
    ax1.text(-8, i, medal, va='center', ha='center', fontsize=14)

ax1.axvline(x=90, color='green', linestyle='--', alpha=0.5, label='Excellent (90%)')
ax1.axvline(x=80, color='blue', linestyle='--', alpha=0.5, label='Good (80%)')
ax1.legend(loc='lower right', fontsize=9)

# 1b. Field Accuracy Comparison
ax2 = axes[0, 1]
x = np.arange(len(models))
width = 0.2

bars1 = ax2.bar(x - 1.5*width, use_case_acc, width, label='Use Case (50%)', color='#3498db')
bars2 = ax2.bar(x - 0.5*width, user_count_acc, width, label='User Count (25%)', color='#2ecc71')
bars3 = ax2.bar(x + 0.5*width, priority_acc, width, label='Priority (15%)', color='#f39c12')
bars4 = ax2.bar(x + 1.5*width, hardware_acc, width, label='Hardware (10%)', color='#9b59b6')

ax2.set_ylabel('Accuracy (%)', fontweight='bold')
ax2.set_title('📈 Field-Level Accuracy by Model', fontweight='bold', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_ylim(0, 105)
ax2.legend(loc='upper right', fontsize=9)
ax2.axhline(y=90, color='green', linestyle='--', alpha=0.3)

# 1c. Speed vs Quality
ax3 = axes[1, 0]
scatter = ax3.scatter(latencies, scores, s=[s*50 for s in sizes], c=scores, cmap='RdYlGn', 
                       edgecolors='black', linewidth=1.5, alpha=0.8)

for i, model in enumerate(models):
    ax3.annotate(model, (latencies[i], scores[i]), textcoords="offset points", 
                  xytext=(5, 5), fontsize=10, fontweight='bold')

ax3.set_xlabel('Average Latency (ms)', fontweight='bold')
ax3.set_ylabel('Weighted Score (%)', fontweight='bold')
ax3.set_title('⚡ Speed vs Quality (bubble size = model size)', fontweight='bold', fontsize=13)
ax3.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='90% threshold')
ax3.axvline(x=1000, color='red', linestyle='--', alpha=0.3, label='1s threshold')
ax3.legend(loc='lower left', fontsize=9)

# Add color bar
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Score (%)', fontweight='bold')

# 1d. JSON Validity
ax4 = axes[1, 1]
bars = ax4.barh(models[::-1], json_validity[::-1], color=['#2ecc71' if v == 100 else '#f39c12' if v > 95 else '#e74c3c' for v in json_validity[::-1]], 
                edgecolor='white', linewidth=2)
ax4.set_xlabel('JSON Validity Rate (%)', fontweight='bold')
ax4.set_title('✅ JSON Generation Validity', fontweight='bold', fontsize=13)
ax4.set_xlim(85, 101)

for i, (bar, val) in enumerate(zip(bars, json_validity[::-1])):
    ax4.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
             va='center', fontweight='bold', fontsize=11)

ax4.axvline(x=100, color='green', linestyle='--', alpha=0.5)
ax4.axvline(x=95, color='orange', linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_dir / 'hybrid_executive_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: hybrid_executive_summary.png")

# ============================================================================
# PLOT 2: Detailed Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Create data matrix
metrics = ['Weighted\nScore', 'Use Case\n(50%)', 'User Count\n(25%)', 'Priority\n(15%)', 'Hardware\n(10%)', 'JSON\nValidity']
data_matrix = np.array([
    scores,
    use_case_acc,
    user_count_acc,
    priority_acc,
    hardware_acc,
    json_validity
]).T

im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=25, vmax=100)

# Add labels
ax.set_xticks(np.arange(len(metrics)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(metrics, fontweight='bold')
ax.set_yticklabels(models, fontweight='bold')

# Add text annotations
for i in range(len(models)):
    for j in range(len(metrics)):
        val = data_matrix[i, j]
        text_color = 'white' if val < 50 else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=text_color, fontweight='bold', fontsize=11)

# Add color bar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Accuracy / Score (%)', fontweight='bold')

ax.set_title('📊 Detailed Performance Heatmap: 7 Models × 6 Metrics\n(Green = Better, Red = Worse)', 
             fontweight='bold', fontsize=14, pad=20)

# Add rank column
for i, (model, score) in enumerate(zip(models, scores)):
    rank = i + 1
    medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else str(rank)
    ax.text(-0.7, i, medal, ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.savefig(output_dir / 'hybrid_heatmap_detailed.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: hybrid_heatmap_detailed.png")

# ============================================================================
# PLOT 3: Comparison Table (visual)
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

# Table data
table_data = []
headers = ['Rank', 'Model', 'Score', 'Use Case', 'User Count', 'Priority', 'Hardware', 'JSON', 'Latency', 'Size']

for i, r in enumerate(results):
    rank = '🥇' if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else str(i + 1)
    table_data.append([
        rank,
        r["model"],
        f'{r["weighted_score"]*100:.1f}%',
        f'{r["field_accuracy"]["use_case"]*100:.1f}%',
        f'{r["field_accuracy"]["user_count"]*100:.1f}%',
        f'{r["field_accuracy"]["priority"]*100:.1f}%',
        f'{r["field_accuracy"]["hardware"]*100:.1f}%',
        f'{r["json_validity"]*100:.1f}%',
        f'{r["latency_ms"]["avg"]:.0f}ms',
        f'{model_sizes.get(r["model"], 0):.1f}GB'
    ])

table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.0)

# Style header
for i, header in enumerate(headers):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

# Color code cells based on value
for i in range(len(results)):
    # Score column
    score = results[i]["weighted_score"] * 100
    color = '#2ecc71' if score > 90 else '#3498db' if score > 85 else '#f39c12' if score > 75 else '#e74c3c'
    table[(i+1, 2)].set_facecolor(color)
    table[(i+1, 2)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    if i % 2 == 0:
        for j in range(len(headers)):
            if j != 2:  # Skip score column
                table[(i+1, j)].set_facecolor('#ecf0f1')

ax.set_title('📋 Complete Model Comparison Table\nCompass Business Context Extraction Evaluation (400 test cases)', 
             fontweight='bold', fontsize=16, pad=20)

# Add footer text
fig.text(0.5, 0.02, 
         'Scoring: Use Case (50%) + User Count (25%) + Priority (15%) + Hardware (10%) | Dataset: 400 cases across 9 use cases',
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(output_dir / 'hybrid_comparison_table.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: hybrid_comparison_table.png")

# ============================================================================
# PLOT 4: Winner Spotlight
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('🏆 TOP 3 MODELS FOR COMPASS BUSINESS CONTEXT EXTRACTION', fontsize=16, fontweight='bold', y=1.02)

for idx, (ax, r) in enumerate(zip(axes, results[:3])):
    medal = '🥇' if idx == 0 else '🥈' if idx == 1 else '🥉'
    color = '#FFD700' if idx == 0 else '#C0C0C0' if idx == 1 else '#CD7F32'
    
    # Radar chart data
    categories = ['Use Case', 'User Count', 'Priority', 'Hardware', 'JSON']
    values = [
        r["field_accuracy"]["use_case"] * 100,
        r["field_accuracy"]["user_count"] * 100,
        r["field_accuracy"]["priority"] * 100,
        r["field_accuracy"]["hardware"] * 100,
        r["json_validity"] * 100
    ]
    values += values[:1]  # Close the polygon
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax = plt.subplot(1, 3, idx + 1, polar=True)
    
    ax.plot(angles, values, 'o-', linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    
    ax.set_title(f'{medal} {r["model"]}\nScore: {r["weighted_score"]*100:.1f}% | {r["latency_ms"]["avg"]:.0f}ms', 
                 fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'hybrid_top3_spotlight.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: hybrid_top3_spotlight.png")

print(f"\n✅ All presentation images saved to: {output_dir}")

