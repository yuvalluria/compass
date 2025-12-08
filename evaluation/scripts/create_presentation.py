#!/usr/bin/env python3
"""
Create Presentation-Ready Visualizations for LLM Evaluation Results

This script generates professional plots and tables with legends and explanations
for presenting the LLM evaluation findings to stakeholders.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_results(results_path: Path) -> dict:
    """Load evaluation results from JSON."""
    with open(results_path, "r") as f:
        return json.load(f)


def create_executive_summary(results: dict, output_dir: Path):
    """Create an executive summary dashboard."""
    models = list(results.keys())
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('LLM Model Evaluation for Compass Business Context Extraction', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Add subtitle with context
    fig.text(0.5, 0.94, 
             'Comparing 6 open-source LLMs on 540 test cases across 5 datasets',
             ha='center', fontsize=12, style='italic', color='gray')
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, 
                          left=0.08, right=0.92, top=0.88, bottom=0.08)
    
    # ===== 1. Overall Accuracy Bar Chart =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    use_case_acc = []
    for model in models:
        field_acc = results[model].get("field_accuracy", {})
        uc = field_acc.get("use_case", {})
        acc = uc.get("correct", 0) / max(uc.get("total", 1), 1) * 100
        use_case_acc.append(acc)
    
    colors = ['#2ecc71' if acc == max(use_case_acc) else '#3498db' for acc in use_case_acc]
    short_names = [m.split(':')[0] for m in models]
    
    bars = ax1.bar(short_names, use_case_acc, color=colors, edgecolor='black', linewidth=1)
    ax1.axhline(y=85, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.7, linewidth=2)
    
    for bar, acc in zip(bars, use_case_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Use Case Detection Accuracy', fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Legend for thresholds
    ax1.legend([mpatches.Patch(color='green', alpha=0.7), 
                mpatches.Patch(color='orange', alpha=0.7)],
               ['90% Target', '85% Minimum'], loc='lower right', fontsize=9)
    
    # ===== 2. JSON Validity & Latency =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    json_valid = []
    latencies = []
    for model in models:
        overall = results[model].get("overall", {})
        json_valid.append(overall.get("json_validity", 0) * 100)
        latencies.append(results[model].get("latency_p90", 0))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, json_valid, width, label='JSON Validity (%)', color='#9b59b6')
    
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, latencies, width, label='P90 Latency (ms)', color='#e74c3c', alpha=0.7)
    
    ax2.set_ylabel('JSON Validity (%)', fontsize=11, color='#9b59b6')
    ax2_twin.set_ylabel('Latency (ms)', fontsize=11, color='#e74c3c')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names)
    ax2.set_title('Quality & Performance', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(0, 110)
    ax2_twin.set_ylim(0, max(latencies) * 1.3)
    
    # Combined legend
    ax2.legend([bars1, bars2], ['JSON Validity (%)', 'P90 Latency (ms)'], 
               loc='upper right', fontsize=9)
    
    # ===== 3. Field-Level Accuracy Comparison =====
    ax3 = fig.add_subplot(gs[1, :])
    
    fields = ['use_case', 'user_count', 'priority', 'hardware']
    field_labels = ['Use Case', 'User Count', 'Priority', 'Hardware']
    field_colors = ['#3498db', '#9b59b6', '#f39c12', '#e74c3c']
    
    data = {field: [] for field in fields}
    for model in models:
        field_acc = results[model].get("field_accuracy", {})
        for field in fields:
            acc_data = field_acc.get(field, {})
            total = acc_data.get("total", 0) or acc_data.get("expected", 0)
            correct = acc_data.get("correct", 0)
            data[field].append(correct / max(total, 1) * 100 if total > 0 else 0)
    
    x = np.arange(len(models))
    width = 0.18
    
    for i, (field, label, color) in enumerate(zip(fields, field_labels, field_colors)):
        offset = (i - 1.5) * width
        bars = ax3.bar(x + offset, data[field], width, label=label, color=color)
    
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names)
    ax3.set_title('Field-Level Extraction Accuracy by Model', fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylim(0, 115)
    ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax3.legend(loc='lower right', ncol=4, fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax3.text(0.02, 0.95, 
             '• Use Case: Correct task type identification\n'
             '• User Count: Accurate number extraction\n'
             '• Priority: Latency/cost preference detection\n'
             '• Hardware: GPU type extraction (H100, A100, etc.)',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== 4. Winner Summary Box =====
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    # Find winner
    best_idx = use_case_acc.index(max(use_case_acc))
    winner = models[best_idx]
    winner_short = winner.split(':')[0]
    
    summary_text = f"""
    RECOMMENDED MODEL: {winner}
    
    +-------------------------------------------+
    |  Use Case Accuracy:     {use_case_acc[best_idx]:.1f}%            |
    |  JSON Validity:         {json_valid[best_idx]:.0f}%              |
    |  P90 Latency:           {latencies[best_idx]:.0f}ms             |
    |  User Count Accuracy:   {data['user_count'][best_idx]:.0f}%              |
    |  Priority Detection:    {data['priority'][best_idx]:.0f}%              |
    |  Hardware Detection:    {data['hardware'][best_idx]:.0f}%              |
    +-------------------------------------------+
    
    Why {winner_short}?
    * Highest overall use case detection accuracy
    * 100% JSON validity (no parsing failures)
    * Good balance of accuracy and latency
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.8))
    ax4.set_title('Recommendation', fontsize=13, fontweight='bold')
    
    # ===== 5. Evaluation Context =====
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    context_text = """
    EVALUATION METHODOLOGY
    
    Datasets Used (540 total test cases):
    - Compass Intent Extraction: 200 cases
    - JSON Generation Benchmark: 100 cases  
    - NLU Intent Classification: 80 cases
    - IFEval Instruction Following: 60 cases
    - Robustness Edge Cases: 100 cases
      (sentence order variations, typos, informal)
    
    Models Evaluated:
    - phi3:medium (14B) - Microsoft
    - phi3:mini (3.8B) - Microsoft  
    - gemma2:9b (9B) - Google
    - qwen2.5:7b (7B) - Alibaba
    - mistral:7b (7B) - Mistral AI
    - llama3.1:8b (8B) - Meta
    
    All models run locally via Ollama on M4 Mac.
    """
    
    ax5.text(0.05, 0.95, context_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#ebf5fb', alpha=0.8))
    ax5.set_title('Methodology', fontsize=13, fontweight='bold')
    
    # Save
    plt.savefig(output_dir / 'executive_summary.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_dir / 'executive_summary.png'}")
    plt.close()


def create_detailed_heatmap(results: dict, output_dir: Path):
    """Create a detailed heatmap with legend."""
    models = list(results.keys())
    short_names = [m.split(':')[0] for m in models]
    
    # Get all use cases
    all_use_cases = set()
    for model in models:
        all_use_cases.update(results[model].get("use_case_accuracies", {}).keys())
    use_cases = sorted(all_use_cases)
    
    display_names = {
        "chatbot_conversational": "Chatbot",
        "code_completion": "Code Completion",
        "code_generation_detailed": "Code Generation",
        "translation": "Translation",
        "content_generation": "Content Gen",
        "summarization_short": "Summarization",
        "document_analysis_rag": "RAG/Doc Analysis",
        "long_document_summarization": "Long Doc Summary",
        "research_legal_analysis": "Legal/Research",
    }
    
    # Build matrix
    matrix = []
    for model in models:
        row = []
        for uc in use_cases:
            acc = results[model].get("use_case_accuracies", {}).get(uc, 0) * 100
            row.append(acc)
        matrix.append(row)
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(use_cases)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([display_names.get(uc, uc) for uc in use_cases], rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(short_names, fontsize=11)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(use_cases)):
            val = matrix[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', 
                   color=color, fontsize=11, fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel('Accuracy (%)', fontsize=12, rotation=-90, va='bottom')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title('Use Case Detection Accuracy by Model\n'
                 '(Green = High Accuracy, Red = Low Accuracy)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add legend/explanation box
    legend_text = (
        "How to read this heatmap:\n"
        "• Each cell shows the % of test cases where the model\n"
        "  correctly identified the use case type\n"
        "• Green (100%) = Perfect accuracy\n"
        "• Red (0%) = Complete failure on this category\n"
        "• Summarization is challenging for all models"
    )
    
    fig.text(0.02, 0.02, legend_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'usecase_heatmap_detailed.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    print(f"✓ Saved: {output_dir / 'usecase_heatmap_detailed.png'}")
    plt.close()


def create_comparison_table(results: dict, output_dir: Path):
    """Create a styled comparison table."""
    models = list(results.keys())
    short_names = [m.split(':')[0] for m in models]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.98, 'LLM Model Comparison Summary', 
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)
    
    ax.text(0.5, 0.92, 
            'Evaluation on 540 test cases | Metrics: Accuracy, Quality, Performance',
            ha='center', va='top', fontsize=11, style='italic', color='gray',
            transform=ax.transAxes)
    
    # Prepare data
    columns = ['Model', 'Use Case\nAccuracy', 'User Count\nAccuracy', 'Priority\nAccuracy', 
               'Hardware\nAccuracy', 'JSON\nValidity', 'P90\nLatency', 'Verdict']
    
    table_data = []
    for i, model in enumerate(models):
        field_acc = results[model].get("field_accuracy", {})
        overall = results[model].get("overall", {})
        
        uc = field_acc.get("use_case", {})
        uc_acc = uc.get("correct", 0) / max(uc.get("total", 1), 1) * 100
        
        user = field_acc.get("user_count", {})
        user_acc = user.get("correct", 0) / max(user.get("total", 1), 1) * 100
        
        pri = field_acc.get("priority", {})
        pri_acc = pri.get("correct", 0) / max(pri.get("expected", 1), 1) * 100 if pri.get("expected", 0) > 0 else 0
        
        hw = field_acc.get("hardware", {})
        hw_acc = hw.get("correct", 0) / max(hw.get("expected", 1), 1) * 100 if hw.get("expected", 0) > 0 else 0
        
        json_val = overall.get("json_validity", 0) * 100
        latency = results[model].get("latency_p90", 0)
        
        # Determine verdict
        if uc_acc >= 89:
            verdict = '*** BEST ***'
        elif uc_acc >= 87:
            verdict = 'Good'
        elif uc_acc >= 82:
            verdict = 'OK'
        else:
            verdict = 'Poor'
        
        table_data.append([
            short_names[i],
            f'{uc_acc:.1f}%',
            f'{user_acc:.0f}%',
            f'{pri_acc:.0f}%',
            f'{hw_acc:.0f}%',
            f'{json_val:.0f}%',
            f'{latency:.0f}ms',
            verdict
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.15, 0.9, 0.7]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color code cells based on values
    for i in range(len(models)):
        for j in range(1, 6):  # Accuracy columns
            cell = table[(i+1, j)]
            val = float(table_data[i][j].replace('%', '').replace('ms', ''))
            if j <= 5:  # Accuracy
                if val >= 100:
                    cell.set_facecolor('#d5f5e3')
                elif val >= 90:
                    cell.set_facecolor('#abebc6')
                elif val >= 80:
                    cell.set_facecolor('#f9e79f')
                else:
                    cell.set_facecolor('#f5b7b1')
    
    # Legend
    legend_text = (
        "Color Legend:\n"
        "Green: >=90% accuracy (Excellent)\n"
        "Yellow: 80-89% accuracy (Good)\n"  
        "Red: <80% accuracy (Needs improvement)"
    )
    ax.text(0.02, 0.08, legend_text, fontsize=9, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    plt.savefig(output_dir / 'comparison_table.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    print(f"✓ Saved: {output_dir / 'comparison_table.png'}")
    plt.close()


def main():
    script_dir = Path(__file__).parent
    results_path = script_dir.parent / "results" / "usecase_evaluation_results.json"
    output_dir = script_dir.parent / "results" / "presentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return
    
    print("=" * 60)
    print("  Creating Presentation-Ready Visualizations")
    print("=" * 60)
    
    results = load_results(results_path)
    
    print("\nGenerating visualizations...")
    create_executive_summary(results, output_dir)
    create_detailed_heatmap(results, output_dir)
    create_comparison_table(results, output_dir)
    
    print("\n" + "=" * 60)
    print("✓ All presentation materials generated!")
    print("=" * 60)
    print(f"\nFiles saved to: {output_dir}")
    print("  📊 executive_summary.png - Main dashboard")
    print("  🗺️  usecase_heatmap_detailed.png - Detailed breakdown")
    print("  📋 comparison_table.png - Side-by-side comparison")


if __name__ == "__main__":
    main()

