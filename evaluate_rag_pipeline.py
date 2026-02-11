"""
RAG Pipeline Evaluation Script
===============================
Standalone script to evaluate the RAG pipeline against the golden dataset.

This script:
1. Loads the golden dataset
2. Runs both RAG and baseline evaluations
3. Compares performance metrics
4. Generates detailed JSON reports

Usage:
    python evaluate_rag_pipeline.py
"""

import json
import time
from pathlib import Path
from rag_financial_pipeline import (
    FinancialNewsVectorDB,
    evaluate_on_golden_dataset,
    populate_vector_db_from_golden_dataset
)


def compare_rag_vs_baseline(
    db: FinancialNewsVectorDB,
    dataset_path: str = "golden_financial_reasoning.json"
):
    """
    Run both RAG and baseline evaluations and compare results.
    
    Args:
        db: FinancialNewsVectorDB instance
        dataset_path: Path to golden dataset
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*70)
    print("RAG VS BASELINE COMPARISON")
    print("="*70)
    
    # Run baseline evaluation
    print("\n[1/2] Running BASELINE evaluation (no RAG)...")
    baseline_results = evaluate_on_golden_dataset(db, dataset_path, use_rag=False)
    
    # Run RAG evaluation
    print("\n[2/2] Running RAG evaluation (with context)...")
    rag_results = evaluate_on_golden_dataset(db, dataset_path, use_rag=True)
    
    # Compare metrics
    baseline_metrics = baseline_results['metrics']
    rag_metrics = rag_results['metrics']
    
    accuracy_improvement = rag_metrics['accuracy_percent'] - baseline_metrics['accuracy_percent']
    latency_change = rag_metrics['average_latency_ms'] - baseline_metrics['average_latency_ms']
    
    comparison = {
        'baseline': baseline_metrics,
        'rag': rag_metrics,
        'improvements': {
            'accuracy_delta': round(accuracy_improvement, 2),
            'latency_delta_ms': round(latency_change, 2),
            'accuracy_improved': accuracy_improvement > 0,
            'latency_acceptable': latency_change < 500  # Less than 500ms overhead
        }
    }
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\nAccuracy:")
    print(f"  Baseline: {baseline_metrics['accuracy_percent']}%")
    print(f"  RAG:      {rag_metrics['accuracy_percent']}%")
    print(f"  Delta:    {accuracy_improvement:+.2f}% {'✓' if accuracy_improvement > 0 else '✗'}")
    
    print(f"\nLatency:")
    print(f"  Baseline: {baseline_metrics['average_latency_ms']:.0f}ms")
    print(f"  RAG:      {rag_metrics['average_latency_ms']:.0f}ms")
    print(f"  Delta:    {latency_change:+.0f}ms {'✓' if latency_change < 500 else '⚠'}")
    
    print(f"\nHigh Confidence Accuracy:")
    print(f"  Baseline: {baseline_metrics['high_confidence_accuracy']}%")
    print(f"  RAG:      {rag_metrics['high_confidence_accuracy']}%")
    
    return {
        'comparison': comparison,
        'baseline_detailed': baseline_results['detailed_results'],
        'rag_detailed': rag_results['detailed_results']
    }


def main():
    """Main evaluation workflow."""
    print("="*70)
    print("RAG FINANCIAL PIPELINE - EVALUATION")
    print("="*70)
    
    # Initialize vector database
    print("\n[Step 1/4] Initializing vector database...")
    db = FinancialNewsVectorDB()
    db.create_index()
    
    # Check if database is populated
    stats = db.get_collection_stats()
    doc_count = stats.get('count', 0)
    
    if doc_count == 0:
        print(f"\n[Step 2/4] Vector database is empty. Populating from golden dataset...")
        populate_vector_db_from_golden_dataset(db)
    else:
        print(f"\n[Step 2/4] Vector database already populated ({doc_count} documents)")
    
    # Run comparison evaluation
    print(f"\n[Step 3/4] Running evaluation...")
    results = compare_rag_vs_baseline(db)
    
    # Save results
    print(f"\n[Step 4/4] Saving results...")
    
    output_file = "evaluation_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_file}")
    
    # Summary
    comparison = results['comparison']
    improvements = comparison['improvements']
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if improvements['accuracy_improved']:
        print("✓ RAG improves accuracy over baseline")
    else:
        print("⚠ RAG does not improve accuracy (may need more context)")
    
    if improvements['latency_acceptable']:
        print("✓ RAG latency overhead is acceptable (<500ms)")
    else:
        print("⚠ RAG has significant latency overhead")
    
    print(f"\nOverall: RAG is {'RECOMMENDED' if improvements['accuracy_improved'] and improvements['latency_acceptable'] else 'NEEDS TUNING'}")
    print("="*70)


if __name__ == "__main__":
    main()
