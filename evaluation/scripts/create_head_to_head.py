#!/usr/bin/env python3
"""
Create head-to-head comparison visualization for Qwen 2.5 7B vs Mistral 7B
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

def load_results():
    results_path = Path(__file__).parent.parent / "results" / "hybrid_evaluation_results.json"
    with open(results_path) as f:
        return json.load(f)

def create_head_to_head():
    data = load_results()
    
    # Extract Qwen and Mistral data
    qwen = next(r for r in data["results"] if r["model"] == "qwen2.5:7b")
    mistral = next(r for r in data["results"] if r["model"] == "mistral:7b")
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Head-to-Head: Qwen 2.5 7B vs Mistral 7B\nCompass Business Context Extraction', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Overall Score Comparison (Top Left)
    ax1 = fig.add_subplot(2, 2, 1)
    models = ['Mistral 7B', 'Qwen 2.5 7B']  # Reversed order for better visual
    scores = [mistral['weighted_score'] * 100, qwen['weighted_score'] * 100]
    colors = ['#3498db', '#2ecc71']
    bars = ax1.barh(models, scores, color=colors, height=0.5, edgecolor='white', linewidth=2)
    ax1.set_xlim(0, 115)
    ax1.set_xlabel('Weighted Score (%)', fontsize=12)
    ax1.set_title('Overall Weighted Score', fontsize=14, fontweight='bold', pad=10)
    
    # Add score labels (outside bars)
    for bar, score in zip(bars, scores):
        ax1.text(score + 2, bar.get_y() + bar.get_height()/2, f'{score:.1f}%', 
                va='center', fontsize=13, fontweight='bold')
    
    # Add winner badge (inside the Qwen bar)
    ax1.text(45, 1, 'WINNER', fontsize=11, fontweight='bold', 
            color='white', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a8f4a', edgecolor='none'))
    
    # Add difference annotation - positioned to the right of the bars, not overlapping legend
    diff = scores[1] - scores[0]
    ax1.text(scores[1] + 12, 1, f'+{diff:.1f}%', fontsize=12, color='#27ae60', 
            fontweight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#27ae60'))
    
    # Threshold lines
    ax1.axvline(x=90, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
    ax1.axvline(x=80, color='orange', linestyle='--', alpha=0.4, linewidth=1.5)
    
    # Labels at bottom, outside the chart area
    ax1.text(90, -0.55, 'Excellent\n(90%)', ha='center', fontsize=8, color='green', alpha=0.8)
    ax1.text(80, -0.55, 'Good\n(80%)', ha='center', fontsize=8, color='orange', alpha=0.8)

    # 2. Field Accuracy Comparison (Top Right)
    ax2 = fig.add_subplot(2, 2, 2)
    fields = ['Use Case', 'User Count', 'Priority', 'Hardware']
    weights = ['(50%)', '(25%)', '(15%)', '(10%)']
    qwen_acc = [qwen['field_accuracy']['use_case'] * 100, 
                qwen['field_accuracy']['user_count'] * 100,
                qwen['field_accuracy']['priority'] * 100,
                qwen['field_accuracy']['hardware'] * 100]
    mistral_acc = [mistral['field_accuracy']['use_case'] * 100,
                   mistral['field_accuracy']['user_count'] * 100,
                   mistral['field_accuracy']['priority'] * 100,
                   mistral['field_accuracy']['hardware'] * 100]
    
    x = np.arange(len(fields))
    width = 0.32
    
    bars1 = ax2.bar(x - width/2, qwen_acc, width, label='Qwen 2.5 7B', color='#2ecc71', edgecolor='white', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, mistral_acc, width, label='Mistral 7B', color='#3498db', edgecolor='white', linewidth=1.5)
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Field-by-Field Accuracy Comparison', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{f}\n{w}' for f, w in zip(fields, weights)], fontsize=10)
    ax2.set_ylim(0, 120)
    
    # No legend needed - colors established in first chart
    
    # Add value labels on bars
    for bar, val in zip(bars1, qwen_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.0f}%', 
                ha='center', fontsize=9, fontweight='bold', color='#1a8f4a')
    for bar, val in zip(bars2, mistral_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.0f}%', 
                ha='center', fontsize=9, fontweight='bold', color='#2980b9')
    
    # Add difference indicators at top (with more space)
    for i, (q, m) in enumerate(zip(qwen_acc, mistral_acc)):
        diff = q - m
        color = '#27ae60' if diff > 0 else '#e74c3c'
        sign = '+' if diff > 0 else ''
        marker = 'Q' if diff > 0 else 'M'
        ax2.text(i, 113, f'{marker} {sign}{diff:.1f}%', ha='center', fontsize=9, 
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.9))

    # 3. Speed & Quality Metrics (Bottom Left)
    ax3 = fig.add_subplot(2, 2, 3)
    
    metrics = ['Avg Latency\n(ms)', 'p90 Latency\n(ms)', 'JSON Validity\n(%)', 'Schema\nCompliance (%)']
    qwen_metrics = [qwen['latency_ms']['avg'], qwen['latency_ms']['p90'], 
                    qwen['json_validity'] * 100, qwen['schema_compliance'] * 100]
    mistral_metrics = [mistral['latency_ms']['avg'], mistral['latency_ms']['p90'],
                       mistral['json_validity'] * 100, mistral['schema_compliance'] * 100]
    
    x = np.arange(len(metrics))
    
    # Normalize for visualization (latency: lower is better, others: higher is better)
    ax3_twin = ax3.twinx()
    
    # Latency bars (left axis)
    bars1 = ax3.bar(x[:2] - width/2, qwen_metrics[:2], width, label='Qwen 2.5 7B', color='#2ecc71')
    bars2 = ax3.bar(x[:2] + width/2, mistral_metrics[:2], width, label='Mistral 7B', color='#3498db')
    ax3.set_ylabel('Latency (ms) - Lower is better', fontsize=10)
    ax3.set_ylim(0, 1500)
    
    # Quality bars (right axis)
    bars3 = ax3_twin.bar(x[2:] - width/2, qwen_metrics[2:], width, color='#2ecc71', alpha=0.7)
    bars4 = ax3_twin.bar(x[2:] + width/2, mistral_metrics[2:], width, color='#3498db', alpha=0.7)
    ax3_twin.set_ylabel('Quality (%) - Higher is better', fontsize=10)
    ax3_twin.set_ylim(80, 105)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.set_title('Speed & Quality Metrics', fontsize=14, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', fontsize=9)
    
    # Add value labels
    for i, (q, m) in enumerate(zip(qwen_metrics[:2], mistral_metrics[:2])):
        ax3.text(i - width/2, q + 30, f'{q:.0f}', ha='center', fontsize=9, color='#2ecc71', fontweight='bold')
        ax3.text(i + width/2, m + 30, f'{m:.0f}', ha='center', fontsize=9, color='#3498db', fontweight='bold')
    
    for i, (q, m) in enumerate(zip(qwen_metrics[2:], mistral_metrics[2:])):
        ax3_twin.text(i + 2 - width/2, q + 1, f'{q:.1f}%', ha='center', fontsize=9, color='#2ecc71', fontweight='bold')
        ax3_twin.text(i + 2 + width/2, m + 1, f'{m:.1f}%', ha='center', fontsize=9, color='#3498db', fontweight='bold')

    # 4. Summary & Recommendation (Bottom Right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = """EVALUATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Cases:    400 (unified dataset)
Categories:    9 use cases, 5 priorities, 8 GPUs
Scoring:       Weighted hybrid scoring

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HEAD-TO-HEAD RESULTS
─────────────────────────────────────────────

Qwen 2.5 7B WINS:
  Overall Score:      91.6% vs 86.9%  (+4.7%)
  Use Case:           88.8% vs 82.0%  (+6.8%)
  Priority:           90.1% vs 72.5%  (+17.6%)
  Schema Compliance:  99.5% vs 87.8%  (+11.7%)
  Speed:              747ms vs 872ms  (faster)

Mistral 7B WINS:
  Hardware:           95.9% vs 91.8%  (+4.1%)
  User Count:         94.5% vs 94.0%  (+0.5%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDATION: Use Qwen 2.5 7B
  - Best overall accuracy (91.6%)
  - Best priority detection (+17.6%)
  - Faster inference, same size
  - Near-perfect schema compliance
"""
    
    ax4.text(0.03, 0.97, summary_text, transform=ax4.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    output_dir = Path(__file__).parent.parent / "results" / "presentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qwen_vs_mistral_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    create_head_to_head()
    print("Head-to-head comparison created!")

