#!/usr/bin/env python3
"""
Create presentation images documenting the accuracy progression:
v1: 91.6% (baseline)
v2: 92.5% (few-shot examples)  
v3: 95.1% (enhanced prompts + post-processing)
v4: 95.0% (use case disambiguation)

For PowerPoint presentation of improvements.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load latest results
results_path = Path(__file__).parent.parent / "results" / "top3_evaluation_600cases.json"
with open(results_path) as f:
    data = json.load(f)

results = data["results"]
output_dir = Path(__file__).parent.parent / "results" / "presentation"
output_dir.mkdir(exist_ok=True)

# Current Qwen results
qwen = results[0]

print("Generating progression presentations...")

# =============================================================================
# 1. ACCURACY PROGRESSION CHART
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

versions = ['v1\nBaseline', 'v2\n+Few-Shot\nExamples', 'v3\n+Enhanced\nPrompts', 'v4\n+Use Case\nDisambiguation']
scores = [91.6, 92.5, 95.1, 95.0]
colors = ['#E74C3C', '#F39C12', '#27AE60', '#2ECC71']

bars = ax.bar(versions, scores, color=colors, edgecolor='white', linewidth=2, width=0.6)

# Add value labels
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{score:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')

# Add improvement arrows
for i in range(1, len(scores)):
    improvement = scores[i] - scores[i-1]
    sign = '+' if improvement > 0 else ''
    color = '#27AE60' if improvement > 0 else '#E74C3C'
    ax.annotate(f'{sign}{improvement:.1f}%', 
                xy=(i, scores[i] - 2),
                fontsize=12, fontweight='bold', color=color, ha='center')

ax.set_ylim(85, 100)
ax.set_ylabel('Weighted Score (%)', fontsize=14)
ax.axhline(y=97, color='green', linestyle='--', alpha=0.5, label='Target: 97%')
ax.axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='Excellent: 95%')
ax.axhline(y=90, color='gray', linestyle='--', alpha=0.3, label='Good: 90%')
ax.legend(loc='lower right')

ax.set_title('Qwen 2.5 7B Accuracy Progression\nCompass Business Context Extraction (600 Test Cases)', 
             fontsize=16, fontweight='bold', pad=20)

# Add improvement summary box
summary = """IMPROVEMENTS MADE:
v1→v2: Added 5 few-shot examples (+0.9%)
v2→v3: Enhanced prompts with 10 examples,
       better disambiguation rules (+2.6%)
v3→v4: Use case disambiguation for
       RAG vs chatbot, short vs long summary (-0.1%)

TOTAL IMPROVEMENT: +3.4%"""
ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_progression.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ accuracy_progression.png")

# =============================================================================
# 2. FIELD-BY-FIELD PROGRESSION
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Field-by-Field Accuracy Progression\nQwen 2.5 7B on 600 Test Cases', fontsize=16, fontweight='bold')

# Data for each field across versions
fields_data = {
    'Use Case (50%)': [88.8, 89.8, 93.3, 93.0],
    'User Count (25%)': [94.0, 96.0, 95.8, 95.7],
    'Priority (15%)': [90.1, 85.8, 95.7, 95.8],
    'Hardware (10%)': [91.8, 99.5, 99.7, 99.7],
}
versions_short = ['v1', 'v2', 'v3', 'v4']

for ax, (field, values) in zip(axes.flat, fields_data.items()):
    colors = ['#3498DB', '#9B59B6', '#27AE60', '#2ECC71']
    bars = ax.bar(versions_short, values, color=colors, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylim(80, 105)
    ax.set_title(field, fontsize=12, fontweight='bold')
    ax.axhline(y=97, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_dir / 'field_progression.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ field_progression.png")

# =============================================================================
# 3. WHAT WE DID - DETAILED BREAKDOWN
# =============================================================================
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

improvement_details = """
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                        ACCURACY IMPROVEMENT JOURNEY: 91.6% → 95.0%                           ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║  VERSION 1: BASELINE (91.6%)                                                                 ║
║  ─────────────────────────────────────────────────────────────────────────────────────────   ║
║  • Basic prompt with schema definition                                                       ║
║  • No examples provided to model                                                             ║
║  • Simple keyword-based post-processing                                                      ║
║                                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║  VERSION 2: +FEW-SHOT EXAMPLES (92.5%, +0.9%)                                               ║
║  ─────────────────────────────────────────────────────────────────────────────────────────   ║
║  • Added 5 few-shot examples to prompt                                                       ║
║  • Examples cover: chatbot, code, RAG, summarization, translation                            ║
║  • Model learns expected output format from examples                                         ║
║                                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║  VERSION 3: +ENHANCED PROMPTS (95.1%, +2.6%)                                                ║
║  ─────────────────────────────────────────────────────────────────────────────────────────   ║
║  • Expanded to 12 few-shot examples covering all 9 use cases                                ║
║  • Added explicit disambiguation tables in prompt:                                           ║
║    - CHATBOT vs RAG distinction rules                                                        ║
║    - SUMMARIZATION length distinction rules                                                  ║
║  • Enhanced priority keyword detection (30+ keywords)                                        ║
║  • Improved post-processor with priority order checking                                      ║
║                                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║  VERSION 4: +USE CASE DISAMBIGUATION (95.0%, -0.1%)                                         ║
║  ─────────────────────────────────────────────────────────────────────────────────────────   ║
║  • Fixed over-aggressive long_document_summarization detection                               ║
║  • Added paraphrasing → translation mapping                                                  ║
║  • Priority-ordered keyword checking in post-processor                                       ║
║  • Note: Slight decrease due to fixing over-fitting on specific patterns                     ║
║                                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║  REMAINING GAP TO 97%: ~2%                                                                   ║
║  ─────────────────────────────────────────────────────────────────────────────────────────   ║
║  • Main issues: summarization_short ↔ long confusion, edge cases                            ║
║  • Solutions: Fine-tuning (LoRA) or ensemble voting                                          ║
║                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
"""

ax.text(0.5, 0.5, improvement_details, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

plt.savefig(output_dir / 'improvement_details.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ improvement_details.png")

# =============================================================================
# 4. FINAL RESULTS SUMMARY (Updated)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LLM Evaluation Results: Qwen 2.5 7B with Enhanced Prompts\n600 Test Cases | 12 Few-Shot Examples | Use Case Disambiguation', 
             fontsize=16, fontweight='bold', y=0.98)

# Extract data
models = [r["model"] for r in results]
scores = [r["weighted_score"] * 100 for r in results]
use_case = [r["field_accuracy"]["use_case"] * 100 for r in results]
user_count = [r["field_accuracy"]["user_count"] * 100 for r in results]
priority = [r["field_accuracy"]["priority"] * 100 for r in results]
hardware = [r["field_accuracy"]["hardware"] * 100 for r in results]
latency = [r["latency_ms"]["avg"] for r in results]
json_valid = [r["json_validity"] * 100 for r in results]

model_labels = [m.replace(':7b', ' 7B').replace(':8b', ' 8B').replace('qwen2.5', 'Qwen 2.5').replace('mistral', 'Mistral').replace('llama3.1', 'Llama 3.1') for m in models]
colors = ['#2ECC71', '#3498DB', '#9B59B6']

# Panel 1: Overall Score
ax1 = axes[0, 0]
bars = ax1.barh(model_labels, scores, color=colors, height=0.5, edgecolor='white', linewidth=2)
ax1.set_xlim(0, 100)
ax1.set_xlabel('Weighted Score (%)', fontsize=12)
ax1.set_title('Overall Weighted Score', fontsize=14, fontweight='bold')
ax1.axvline(x=97, color='green', linestyle='--', alpha=0.5, label='Target (97%)')
ax1.axvline(x=95, color='orange', linestyle='--', alpha=0.5, label='Excellent (95%)')
for bar, score in zip(bars, scores):
    ax1.text(score + 1, bar.get_y() + bar.get_height()/2, f'{score:.1f}%', 
             va='center', fontsize=12, fontweight='bold')
ax1.text(scores[0] - 12, 0, 'WINNER', fontsize=9, fontweight='bold', 
         color='white', ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='#27AE60', edgecolor='none'))

# Panel 2: Field Accuracy
ax2 = axes[0, 1]
fields = ['Use Case\n(50%)', 'User Count\n(25%)', 'Priority\n(15%)', 'Hardware\n(10%)']
x = np.arange(len(fields))
width = 0.25
ax2.bar(x - width, [use_case[0], user_count[0], priority[0], hardware[0]], width, label=model_labels[0], color=colors[0])
ax2.bar(x, [use_case[1], user_count[1], priority[1], hardware[1]], width, label=model_labels[1], color=colors[1])
ax2.bar(x + width, [use_case[2], user_count[2], priority[2], hardware[2]], width, label=model_labels[2], color=colors[2])
ax2.set_xticks(x)
ax2.set_xticklabels(fields)
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(0, 110)
ax2.axhline(y=97, color='green', linestyle='--', alpha=0.3)
ax2.set_title('Field-Level Accuracy', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)

# Panel 3: Speed vs Quality
ax3 = axes[1, 0]
sizes = [180, 140, 160]
scatter = ax3.scatter(latency, scores, c=scores, s=sizes, cmap='RdYlGn', 
                      edgecolors='black', linewidths=2, vmin=80, vmax=100)
for i, label in enumerate(model_labels):
    ax3.annotate(label, (latency[i], scores[i]), textcoords="offset points", 
                 xytext=(0, 15), ha='center', fontsize=10, fontweight='bold')
ax3.set_xlabel('Average Latency (ms)', fontsize=12)
ax3.set_ylabel('Weighted Score (%)', fontsize=12)
ax3.set_title('Speed vs Quality', fontsize=14, fontweight='bold')
ax3.axhline(y=95, color='green', linestyle='--', alpha=0.3)

# Panel 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary = f"""
╔════════════════════════════════════════════════════════════╗
║           FINAL RESULTS (v4 - Use Case Disambiguation)     ║
╠════════════════════════════════════════════════════════════╣
║  Test Cases:     600 (unified dataset)                     ║
║  Improvements:   12 few-shot + enhanced post-processing    ║
╠════════════════════════════════════════════════════════════╣
║                     RANKINGS                               ║
╠════════════════════════════════════════════════════════════╣
║  1. Qwen 2.5 7B:   {scores[0]:.1f}%  (WINNER)                     ║
║  2. Mistral 7B:    {scores[1]:.1f}%                               ║
║  3. Llama 3.1 8B:  {scores[2]:.1f}%                               ║
╠════════════════════════════════════════════════════════════╣
║              QWEN 2.5 7B DETAILED SCORES                   ║
╠════════════════════════════════════════════════════════════╣
║  Use Case:       {use_case[0]:.1f}%                               ║
║  User Count:     {user_count[0]:.1f}%                              ║
║  Priority:       {priority[0]:.1f}%                               ║
║  Hardware:       {hardware[0]:.1f}%                               ║
║  JSON Validity:  {json_valid[0]:.0f}%                                ║
║  Avg Latency:    {latency[0]:.0f}ms                               ║
╠════════════════════════════════════════════════════════════╣
║  JOURNEY: 91.6% → 92.5% → 95.1% → {scores[0]:.1f}%               ║
║  TOTAL IMPROVEMENT: +{scores[0]-91.6:.1f}%                            ║
╚════════════════════════════════════════════════════════════╝
"""
ax4.text(0.5, 0.5, summary, transform=ax4.transAxes, fontsize=10,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_dir / 'final_results_v4.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ final_results_v4.png")

# Also update the main files
plt.savefig(output_dir / 'executive_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(output_dir / 'hybrid_executive_summary.png', dpi=150, bbox_inches='tight', facecolor='white')

print(f"\n✅ All progression presentations saved to: {output_dir}")
print(f"\nFiles created for PowerPoint:")
print(f"  1. accuracy_progression.png    - Bar chart showing 91.6% → 95.0%")
print(f"  2. field_progression.png       - Field-by-field improvement")
print(f"  3. improvement_details.png     - What we did at each step")
print(f"  4. final_results_v4.png        - Current results summary")

