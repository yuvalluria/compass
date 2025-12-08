#!/usr/bin/env python3
"""
Generate comprehensive plots for LLM evaluation results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: Path) -> dict:
    """Load evaluation results from JSON."""
    with open(results_path, "r") as f:
        return json.load(f)


def plot_field_accuracy(results: dict, output_dir: Path):
    """Plot field-level accuracy comparison."""
    models = list(results.keys())
    
    # Extract field accuracies
    fields = ["use_case", "user_count", "priority", "hardware"]
    field_labels = ["Use Case", "User Count", "Priority", "Hardware"]
    
    data = {field: [] for field in fields}
    
    for model in models:
        field_acc = results[model].get("field_accuracy", {})
        for field in fields:
            acc_data = field_acc.get(field, {})
            total = acc_data.get("total", 0) or acc_data.get("expected", 0)
            correct = acc_data.get("correct", 0)
            if total > 0:
                data[field].append(correct / total * 100)
            else:
                data[field].append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(models))
    width = 0.2
    multiplier = 0
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (field, label) in enumerate(zip(fields, field_labels)):
        offset = width * multiplier
        bars = ax.bar(x + offset, data[field], width, label=label, color=colors[i])
        
        # Add value labels on bars
        for bar, val in zip(bars, data[field]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
        
        multiplier += 1
    
    # Customize
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Field-Level Extraction Accuracy by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace(':7b', '').replace(':8b', '').replace(':9b', '') for m in models])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'field_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'field_accuracy_comparison.png'}")
    plt.close()


def plot_overall_comparison(results: dict, output_dir: Path):
    """Plot overall model comparison with latency."""
    models = list(results.keys())
    short_names = [m.replace(':7b', '').replace(':8b', '').replace(':9b', '') for m in models]
    
    # Extract metrics
    use_case_acc = []
    latencies = []
    
    for model in models:
        field_acc = results[model].get("field_accuracy", {})
        uc = field_acc.get("use_case", {})
        total = uc.get("total", 1)
        correct = uc.get("correct", 0)
        use_case_acc.append(correct / total * 100 if total > 0 else 0)
        latencies.append(results[model].get("latency_p90", 0))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Accuracy bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax1.bar(short_names, use_case_acc, color=colors, edgecolor='black', linewidth=1.2)
    
    # Highlight best
    best_idx = np.argmax(use_case_acc)
    bars[best_idx].set_color('#2ECC71')
    bars[best_idx].set_edgecolor('#27AE60')
    bars[best_idx].set_linewidth(2)
    
    for bar, val in zip(bars, use_case_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Use Case Accuracy (%)', fontsize=12)
    ax1.set_title('Use Case Detection Accuracy', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% target')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Latency comparison
    colors2 = plt.cm.plasma(np.linspace(0.2, 0.8, len(models)))
    bars2 = ax2.bar(short_names, latencies, color=colors2, edgecolor='black', linewidth=1.2)
    
    # Highlight fastest
    fastest_idx = np.argmin(latencies)
    bars2[fastest_idx].set_color('#3498DB')
    bars2[fastest_idx].set_edgecolor('#2980B9')
    bars2[fastest_idx].set_linewidth(2)
    
    for bar, val in zip(bars2, latencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('P90 Latency (ms)', fontsize=12)
    ax2.set_title('Response Latency (P90)', fontsize=13, fontweight='bold')
    ax2.axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='1s target')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('LLM Model Comparison for Compass Intent Extraction', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'model_comparison.png'}")
    plt.close()


def plot_use_case_heatmap(results: dict, output_dir: Path):
    """Plot heatmap of use case accuracy by model."""
    models = list(results.keys())
    short_names = [m.replace(':7b', '').replace(':8b', '').replace(':9b', '') for m in models]
    
    # Get all use cases
    all_use_cases = set()
    for model in models:
        all_use_cases.update(results[model].get("use_case_accuracies", {}).keys())
    
    use_cases = sorted(all_use_cases)
    
    # Use case display names
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
            acc = results[model].get("use_case_accuracies", {}).get(uc, 0)
            row.append(acc * 100)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Labels
    ax.set_xticks(np.arange(len(use_cases)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([display_names.get(uc, uc) for uc in use_cases], rotation=45, ha='right')
    ax.set_yticklabels(short_names)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(use_cases)):
            val = matrix[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom')
    
    ax.set_title('Use Case Accuracy Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'usecase_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'usecase_heatmap.png'}")
    plt.close()


def create_summary_table(results: dict, output_dir: Path):
    """Create a summary table as an image."""
    models = list(results.keys())
    short_names = [m.replace(':7b', '').replace(':8b', '').replace(':9b', '') for m in models]
    
    # Prepare data
    table_data = []
    for model in models:
        field_acc = results[model].get("field_accuracy", {})
        
        uc = field_acc.get("use_case", {})
        uc_acc = uc.get("correct", 0) / max(uc.get("total", 1), 1) * 100
        
        user = field_acc.get("user_count", {})
        user_acc = user.get("correct", 0) / max(user.get("total", 1), 1) * 100
        
        pri = field_acc.get("priority", {})
        pri_acc = pri.get("correct", 0) / max(pri.get("expected", 1), 1) * 100 if pri.get("expected", 0) > 0 else 0
        
        hw = field_acc.get("hardware", {})
        hw_acc = hw.get("correct", 0) / max(hw.get("expected", 1), 1) * 100 if hw.get("expected", 0) > 0 else 0
        
        latency = results[model].get("latency_p90", 0)
        
        table_data.append([
            f'{uc_acc:.1f}%',
            f'{user_acc:.1f}%', 
            f'{pri_acc:.1f}%',
            f'{hw_acc:.1f}%',
            f'{latency:.0f}ms'
        ])
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    columns = ['Use Case', 'User Count', 'Priority', 'Hardware', 'P90 Latency']
    
    table = ax.table(
        cellText=table_data,
        rowLabels=short_names,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] * 5
    )
    
    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color header
    for j, col in enumerate(columns):
        table[(0, j)].set_facecolor('#3498DB')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color row labels
    for i, name in enumerate(short_names):
        table[(i+1, -1)].set_facecolor('#ECF0F1')
        table[(i+1, -1)].set_text_props(fontweight='bold')
    
    # Highlight best values
    for j in range(5):
        vals = [float(table_data[i][j].replace('%', '').replace('ms', '')) for i in range(len(models))]
        if j == 4:  # Latency - lower is better
            best_idx = vals.index(min(vals))
        else:  # Accuracy - higher is better
            best_idx = vals.index(max(vals))
        table[(best_idx + 1, j)].set_facecolor('#2ECC71')
        table[(best_idx + 1, j)].set_text_props(fontweight='bold')
    
    ax.set_title('LLM Evaluation Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'summary_table.png'}")
    plt.close()


def main():
    script_dir = Path(__file__).parent
    results_path = script_dir.parent / "results" / "usecase_evaluation_results.json"
    output_dir = script_dir.parent / "results"
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return
    
    print("Loading results...")
    results = load_results(results_path)
    
    print("\nGenerating plots...")
    plot_field_accuracy(results, output_dir)
    plot_overall_comparison(results, output_dir)
    plot_use_case_heatmap(results, output_dir)
    create_summary_table(results, output_dir)
    
    print("\n✓ All plots generated!")
    print(f"\nFiles saved to: {output_dir}")
    print("  - field_accuracy_comparison.png")
    print("  - model_comparison.png")
    print("  - usecase_heatmap.png")
    print("  - summary_table.png")


if __name__ == "__main__":
    main()

