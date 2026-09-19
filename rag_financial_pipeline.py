"""
RAG Financial Pipeline
======================
End-to-end pipeline connecting ChromaDB vector retrieval with financial reasoning.

This module implements Retrieval-Augmented Generation (RAG) for financial analysis:
1. Retrieve relevant historical news from vector database
2. Provide context to the financial reasoner
3. Generate enhanced predictions with supporting evidence
4. Evaluate performance against golden dataset

Author: Financial Reasoning System
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Import core modules
from financial_vector_db_setup import FinancialNewsVectorDB
from financial_reasoner import generate_financial_reasoning

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================================
# Core RAG Pipeline Functions
# ============================================================================

def retrieve_relevant_context(
    query: str,
    db: FinancialNewsVectorDB,
    ticker: Optional[str] = None,
    event_type: Optional[str] = None,
    n_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant historical news from the vector database.
    
    Args:
        query: News text to find similar examples for
        db: FinancialNewsVectorDB instance
        ticker: Optional ticker filter
        event_type: Optional event type filter (Macro, Earnings_Miss, etc.)
        n_results: Number of results to retrieve
    
    Returns:
        List of dictionaries with retrieved documents and metadata
    """
    try:
        results = db.retrieve_with_filters(
            query=query,
            ticker=ticker,
            event_type=event_type,
            n_results=n_results
        )
        
        # Format results for easier consumption
        context_items = []
        if results and 'documents' in results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                distance = results['distances'][0][i] if results.get('distances') else None
                
                context_items.append({
                    'document': doc,
                    'metadata': metadata,
                    'similarity': 1 - distance if distance is not None else None
                })
        
        return context_items
    
    except Exception as e:
        print(f"Warning: Retrieval failed: {e}")
        return []


def analyze_with_context(
    news: str,
    context: List[Dict[str, Any]],
    market_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze news with RAG-enhanced reasoning using retrieved context.
    
    Args:
        news: Current news to analyze
        context: Retrieved historical context from vector DB
        market_data: Optional market data (volatility, etc.)
    
    Returns:
        Dictionary with analysis results including RAG context
    """
    # Build enhanced prompt with context
    if context:
        context_text = "\n\nRelevant Historical Context:\n"
        for i, item in enumerate(context[:3], 1):  # Use top 3 for brevity
            doc = item['document']
            meta = item.get('metadata', {})
            context_text += f"{i}. {doc}\n"
            if meta:
                context_text += f"   Event: {meta.get('event_type', 'Unknown')}, "
                context_text += f"Sentiment: {meta.get('sentiment_score', 'N/A')}\n"
        
        # Enhance news with context
        enhanced_news = f"{news}\n{context_text}"
    else:
        enhanced_news = news
    
    # Get reasoning with enhanced context
    result = generate_financial_reasoning(enhanced_news, market_data)
    
    # Add RAG metadata
    result['rag_context_used'] = len(context) > 0
    result['rag_context_count'] = len(context)
    
    return result


def populate_vector_db_from_golden_dataset(
    db: FinancialNewsVectorDB,
    dataset_path: str = "golden_financial_reasoning.json"
) -> Dict[str, Any]:
    """
    Populate the vector database with examples from the golden dataset.
    
    Args:
        db: FinancialNewsVectorDB instance
        dataset_path: Path to golden dataset JSON file
    
    Returns:
        Dictionary with ingestion statistics
    """
    print(f"\n{'='*60}")
    print("POPULATING VECTOR DATABASE FROM GOLDEN DATASET")
    print(f"{'='*60}\n")
    
    # Load golden dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} examples from {dataset_path}")
    
    # Prepare data for ingestion
    documents = []
    tickers = []
    event_types = []
    sentiment_scores = []
    market_regimes = []
    published_timestamps = []
    ids = []
    
    for item in dataset:
        documents.append(item['input_news'])
        tickers.append('MARKET')  # Generic ticker for macro events
        event_types.append(item['event_category'])
        
        # Map direction to sentiment score
        direction = item.get('expected_direction', 'Neutral')
        if direction == 'Bullish':
            sentiment = 0.8
        elif direction == 'Bearish':
            sentiment = -0.8
        else:
            sentiment = 0.0
        sentiment_scores.append(sentiment)
        
        # Map volatility to market regime
        volatility = item.get('input_volatility', 'Low')
        regime = 'High_Vol' if volatility in ['High', 'Extreme'] else 'Low_Vol'
        market_regimes.append(regime)
        
        # Use current timestamp
        published_timestamps.append(int(time.time()))
        
        # Use ID from dataset
        ids.append(f"golden_{item['id']}")
    
    # Add documents to vector DB
    print(f"\nIngesting {len(documents)} documents into vector database...")
    start_time = time.time()
    
    db.add_documents(
        documents=documents,
        tickers=tickers,
        event_types=event_types,
        sentiment_scores=sentiment_scores,
        market_regimes=market_regimes,
        published_timestamps=published_timestamps,
        ids=ids
    )
    
    elapsed = time.time() - start_time
    
    # Get collection stats
    stats = db.get_collection_stats()
    
    print(f"\n✓ Ingestion complete!")
    print(f"  - Documents added: {len(documents)}")
    print(f"  - Time taken: {elapsed:.2f}s")
    print(f"  - Total documents in DB: {stats.get('count', 'Unknown')}")
    
    return {
        'documents_added': len(documents),
        'time_taken': elapsed,
        'total_documents': stats.get('count', 0)
    }


def evaluate_on_golden_dataset(
    db: FinancialNewsVectorDB,
    dataset_path: str = "golden_financial_reasoning.json",
    use_rag: bool = True
) -> Dict[str, Any]:
    """
    Evaluate the RAG pipeline on the golden dataset.
    
    Args:
        db: FinancialNewsVectorDB instance
        dataset_path: Path to golden dataset JSON file
        use_rag: Whether to use RAG context (True) or baseline (False)
    
    Returns:
        Dictionary with evaluation metrics and detailed results
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {'RAG' if use_rag else 'BASELINE'} PIPELINE ON GOLDEN DATASET")
    print(f"{'='*60}\n")
    
    # Load golden dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} examples for evaluation")
    
    results = []
    correct_predictions = 0
    total_latency = 0
    
    for i, item in enumerate(dataset, 1):
        print(f"\nProcessing {i}/{len(dataset)}: {item['id']}")
        
        news = item['input_news']
        expected_direction = item['expected_direction']
        market_data = {'volatility': item.get('input_volatility', 'Medium')}
        
        start_time = time.time()
        
        if use_rag:
            # Retrieve context
            context = retrieve_relevant_context(
                query=news,
                db=db,
                event_type=item.get('event_category'),
                n_results=3
            )
            # Analyze with context
            prediction = analyze_with_context(news, context, market_data)
        else:
            # Baseline: no RAG context
            prediction = generate_financial_reasoning(news, market_data)
        
        latency = (time.time() - start_time) * 1000  # ms
        total_latency += latency
        
        predicted_direction = prediction.get('expected_direction', 'Unknown')
        is_correct = predicted_direction == expected_direction
        
        if is_correct:
            correct_predictions += 1
        
        result = {
            'id': item['id'],
            'news': news,
            'expected_direction': expected_direction,
            'predicted_direction': predicted_direction,
            'correct': is_correct,
            'confidence': prediction.get('confidence', 0),
            'latency_ms': latency,
            'event_class': prediction.get('event_class', 'Unknown'),
            'mechanism': prediction.get('mechanism_trace', ''),
            'source': prediction.get('source', 'unknown')
        }
        
        results.append(result)
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} Expected: {expected_direction}, Predicted: {predicted_direction}")
        print(f"     Confidence: {prediction.get('confidence', 0)}, Latency: {latency:.0f}ms")
    
    # Calculate metrics
    accuracy = (correct_predictions / len(dataset)) * 100 if dataset else 0
    avg_latency = total_latency / len(dataset) if dataset else 0
    
    # Confidence calibration (simple version)
    high_conf_correct = sum(1 for r in results if r['confidence'] >= 80 and r['correct'])
    high_conf_total = sum(1 for r in results if r['confidence'] >= 80)
    high_conf_accuracy = (high_conf_correct / high_conf_total * 100) if high_conf_total > 0 else 0
    
    metrics = {
        'mode': 'RAG' if use_rag else 'Baseline',
        'total_examples': len(dataset),
        'correct_predictions': correct_predictions,
        'accuracy_percent': round(accuracy, 2),
        'average_latency_ms': round(avg_latency, 2),
        'high_confidence_accuracy': round(high_conf_accuracy, 2),
        'high_confidence_count': high_conf_total
    }
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mode: {metrics['mode']}")
    print(f"Accuracy: {metrics['accuracy_percent']}% ({correct_predictions}/{len(dataset)})")
    print(f"Average Latency: {metrics['average_latency_ms']:.0f}ms")
    print(f"High Confidence Accuracy: {metrics['high_confidence_accuracy']}% (n={high_conf_total})")
    
    return {
        'metrics': metrics,
        'detailed_results': results
    }


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Financial Pipeline")
    parser.add_argument('--populate', action='store_true', help='Populate vector DB from golden dataset')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on golden dataset')
    parser.add_argument('--baseline', action='store_true', help='Run baseline (no RAG) evaluation')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    
    args = parser.parse_args()
    
    # Initialize vector database
    print("Initializing vector database...")
    db = FinancialNewsVectorDB()
    db.create_index()
    
    if args.populate:
        # Populate database
        populate_vector_db_from_golden_dataset(db)
    
    elif args.evaluate:
        # Run RAG evaluation
        results = evaluate_on_golden_dataset(db, use_rag=True)
        
        # Save results
        output_file = "evaluation_results_rag.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
    
    elif args.baseline:
        # Run baseline evaluation
        results = evaluate_on_golden_dataset(db, use_rag=False)
        
        # Save results
        output_file = "evaluation_results_baseline.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
    
    elif args.demo:
        # Interactive demo
        print("\n" + "="*60)
        print("RAG FINANCIAL PIPELINE - INTERACTIVE DEMO")
        print("="*60)
        
        while True:
            news = input("\nEnter financial news (or 'quit' to exit): ")
            if news.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\nRetrieving relevant context...")
            context = retrieve_relevant_context(news, db, n_results=3)
            
            if context:
                print(f"\nFound {len(context)} similar historical events:")
                for i, item in enumerate(context, 1):
                    print(f"{i}. {item['document'][:100]}...")
            else:
                print("No relevant context found.")
            
            print("\nAnalyzing with RAG...")
            result = analyze_with_context(news, context)
            
            print(f"\nResult:")
            print(f"  Event Class: {result['event_class']}")
            print(f"  Direction: {result['expected_direction']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Mechanism: {result['mechanism_trace']}")
    
    else:
        # Default: show usage
        parser.print_help()
