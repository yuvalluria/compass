#!/usr/bin/env python3
"""
Create presentation comparing small models vs the 7B winner.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11


def load_results():
    results_path = Path(__file__).parent.parent / "results" / "small_model_evaluation.json"
    with open(results_path) as f:
        return json.load(f)


def create_comparison():
    data = load_results()
    results = data["results"]
    
    # Sort by weighted score
    results = sorted(results, key=lambda x: x["weighted_score"], reverse=True)
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Small Model Evaluation: Can We Reduce Size Without Losing Quality?', 
                 fontsize=18, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94, 
             'Comparing Qwen2.5:1.5b, Qwen2.5:3b, Gemma2:2b vs Winner (Qwen2.5:7b)',
             ha='center', fontsize=12, style='italic', color='gray')
    
    # Extract data
    models = [r["model"] for r in results]
    short_names = [m.replace("qwen2.5:", "Qwen ").replace("gemma2:", "Gemma ") for m in models]
    scores = [r["weighted_score"] * 100 for r in results]
    use_case = [r["field_accuracy"]["use_case"] * 100 for r in results]
    user_count = [r["field_accuracy"]["user_count"] * 100 for r in results]
    priority = [r["field_accuracy"]["priority"] * 100 for r in results]
    json_valid = [r["json_validity"] * 100 for r in results]
    latencies = [r["latency_ms"]["avg"] for r in results]
    
    # Model sizes (approximate)
    sizes = {"qwen2.5:7b": 4.7, "qwen2.5:3b": 1.9, "qwen2.5:1.5b": 1.0, "gemma2:2b": 1.6}
    model_sizes = [sizes.get(m, 2.0) for m in models]
    
    # Colors - green for winner, blue for others
    colors = ['#2ecc71' if s == max(scores) else '#3498db' if 'qwen' in m else '#e74c3c' 
              for m, s in zip(models, scores)]
    
    # ===== 1. Overall Weighted Score (Top Left) =====
    ax1 = fig.add_subplot(2, 2, 1)
    bars = ax1.barh(short_names, scores, color=colors, height=0.6, edgecolor='white', linewidth=2)
    ax1.set_xlim(0, 105)
    ax1.set_xlabel('Weighted Score (%)', fontsize=12)
    ax1.set_title('Overall Weighted Score (Hybrid Scoring)', fontsize=14, fontweight='bold', pad=10)
    
    for bar, score in zip(bars, scores):
        ax1.text(score + 1, bar.get_y() + bar.get_height()/2, f'{score:.1f}%', 
                va='center', fontsize=12, fontweight='bold')
    
    # Add thresholds
    ax1.axvline(x=90, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax1.axvline(x=80, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    ax1.text(90, -0.5, '90% (Excellent)', fontsize=9, color='green', ha='center')
    ax1.text(80, -0.5, '80% (Good)', fontsize=9, color='orange', ha='center')
    
    # Winner badge
    ax1.text(scores[0] - 20, 0, 'WINNER', fontsize=10, fontweight='bold', 
            color='white', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#27ae60', edgecolor='none'))
    
    # ===== 2. Field Accuracy Comparison (Top Right) =====
    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(len(models))
    width = 0.2
    
    ax2.bar(x - 1.5*width, use_case, width, label='Use Case (50%)', color='#3498db')
    ax2.bar(x - 0.5*width, user_count, width, label='User Count (25%)', color='#2ecc71')
    ax2.bar(x + 0.5*width, priority, width, label='Priority (15%)', color='#f39c12')
    ax2.bar(x + 1.5*width, json_valid, width, label='JSON Valid', color='#9b59b6')
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Field-Level Accuracy by Model', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    
    # ===== 3. Speed vs Quality Scatter (Bottom Left) =====
    ax3 = fig.add_subplot(2, 2, 3)
    
    scatter = ax3.scatter(latencies, scores, s=[s*80 for s in model_sizes], c=scores, 
                         cmap='RdYlGn', edgecolors='black', linewidth=1.5, alpha=0.8)
    
    for i, (lat, sc, name, size) in enumerate(zip(latencies, scores, short_names, model_sizes)):
        ax3.annotate(f'{name}\n({size:.1f}GB)', (lat, sc), 
                    textcoords="offset points", xytext=(0, 15), 
                    ha='center', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Average Latency (ms) - Lower is Better', fontsize=12)
    ax3.set_ylabel('Weighted Score (%) - Higher is Better', fontsize=12)
    ax3.set_title('Speed vs Quality Tradeoff (bubble size = model size)', fontsize=14, fontweight='bold', pad=10)
    
    # Add quadrants
    ax3.axhline(y=85, color='orange', linestyle='--', alpha=0.5)
    ax3.axvline(x=400, color='orange', linestyle='--', alpha=0.5)
    ax3.text(200, 95, 'IDEAL\n(Fast + Accurate)', fontsize=10, ha='center', color='green', fontweight='bold')
    ax3.text(600, 70, 'Slow + Poor', fontsize=10, ha='center', color='red', alpha=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
    cbar.set_label('Score (%)', fontsize=10)
    
    # ===== 4. Summary & Recommendation (Bottom Right) =====
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate gaps
    winner_score = scores[0]
    summary_text = f"""
KEY FINDINGS
{'='*50}

Winner: {short_names[0]} ({scores[0]:.1f}%)

Performance Gap vs Winner:
"""
    for i, (name, score) in enumerate(zip(short_names[1:], scores[1:])):
        gap = winner_score - score
        summary_text += f"  {name}: -{gap:.1f}% (Score: {score:.1f}%)\n"
    
    summary_text += f"""
{'='*50}

VERDICT: Smaller Models Trade Quality for Speed

Model Size vs Quality:
  - Qwen 7B (4.7GB):   91.6% score, 747ms
  - Qwen 3B (1.9GB):   81.9% score, 374ms  (-9.7%)
  - Qwen 1.5B (1.0GB): 78.8% score, 251ms (-12.8%)
  - Gemma 2B (1.6GB):  65.0% score, 436ms (-26.6%)

{'='*50}

RECOMMENDATION:
  - Production: Use Qwen 2.5 7B (best accuracy)
  - Edge/Mobile: Use Qwen 2.5 3B (acceptable quality)
  - Latency-critical: Consider Qwen 1.5B + fine-tuning
  - Avoid: Gemma 2B (poor priority detection)
"""
    
    ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6', pad=0.5))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    
    output_dir = Path(__file__).parent.parent / "results" / "presentation"
    output_path = output_dir / "small_model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    create_comparison()

