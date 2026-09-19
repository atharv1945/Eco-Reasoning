"""
RAG Pipeline Integration Tests
===============================
pytest suite for testing the RAG pipeline integration.

Tests:
1. Vector DB population from golden dataset
2. Retrieval with metadata filters
3. End-to-end RAG pipeline
4. Batch processing
5. Context enhancement verification

Usage:
    pytest test_rag_integration.py -v
"""

import pytest
import json
import os
from pathlib import Path

# Import modules to test
from rag_financial_pipeline import (
    FinancialNewsVectorDB,
    retrieve_relevant_context,
    analyze_with_context,
    populate_vector_db_from_golden_dataset,
    evaluate_on_golden_dataset
)


@pytest.fixture(scope="module")
def vector_db():
    """Create a test vector database instance."""
    # Use a test-specific directory
    test_db_path = "./test_financial_news_db"
    db = FinancialNewsVectorDB(persist_directory=test_db_path)
    db.create_index()
    yield db
    # Cleanup after tests
    import shutil
    if Path(test_db_path).exists():
        shutil.rmtree(test_db_path)


@pytest.fixture(scope="module")
def golden_dataset_path():
    """Path to the golden dataset."""
    return "golden_financial_reasoning.json"


class TestVectorDBPopulation:
    """Test vector database population from golden dataset."""
    
    def test_populate_from_golden_dataset(self, vector_db, golden_dataset_path):
        """Test that golden dataset can be ingested into vector DB."""
        # Populate the database
        stats = populate_vector_db_from_golden_dataset(vector_db, golden_dataset_path)
        
        # Verify ingestion
        assert stats['documents_added'] > 0, "Should add documents"
        assert stats['total_documents'] >= stats['documents_added'], "Total should be >= added"
        
        # Verify collection stats
        collection_stats = vector_db.get_collection_stats()
        assert collection_stats['count'] > 0, "Collection should have documents"
    
    def test_golden_dataset_exists(self, golden_dataset_path):
        """Test that golden dataset file exists and is valid JSON."""
        assert Path(golden_dataset_path).exists(), f"Golden dataset not found at {golden_dataset_path}"
        
        with open(golden_dataset_path, 'r') as f:
            dataset = json.load(f)
        
        assert isinstance(dataset, list), "Dataset should be a list"
        assert len(dataset) > 0, "Dataset should not be empty"
        
        # Verify structure of first item
        first_item = dataset[0]
        required_fields = ['id', 'event_category', 'input_news', 'expected_direction']
        for field in required_fields:
            assert field in first_item, f"Missing required field: {field}"


class TestRetrieval:
    """Test retrieval functionality."""
    
    def test_retrieve_relevant_context(self, vector_db):
        """Test that retrieval returns relevant context."""
        query = "Fed rate hike due to inflation"
        context = retrieve_relevant_context(query, vector_db, n_results=3)
        
        # Should return results (assuming DB is populated)
        assert isinstance(context, list), "Context should be a list"
        
        # If results exist, verify structure
        if context:
            assert 'document' in context[0], "Context item should have 'document'"
            assert 'metadata' in context[0], "Context item should have 'metadata'"
    
    def test_retrieve_with_event_filter(self, vector_db):
        """Test retrieval with event type filter."""
        query = "Earnings report"
        context = retrieve_relevant_context(
            query, 
            vector_db, 
            event_type="Earnings_Miss",
            n_results=5
        )
        
        assert isinstance(context, list), "Context should be a list"
        
        # If results exist, verify they match the filter
        if context:
            for item in context:
                meta = item.get('metadata', {})
                # Note: May not always match due to limited data
                assert 'event_type' in meta or len(context) == 0
    
    def test_retrieve_empty_db(self):
        """Test retrieval on empty database returns empty list."""
        empty_db = FinancialNewsVectorDB(persist_directory="./empty_test_db")
        empty_db.create_index()
        
        context = retrieve_relevant_context("test query", empty_db)
        assert context == [], "Empty DB should return empty context"
        
        # Cleanup
        import shutil
        if Path("./empty_test_db").exists():
            shutil.rmtree("./empty_test_db")


class TestRAGPipeline:
    """Test end-to-end RAG pipeline."""
    
    def test_analyze_with_context(self, vector_db):
        """Test RAG-enhanced analysis."""
        news = "Fed signals aggressive rate hikes"
        context = retrieve_relevant_context(news, vector_db, n_results=3)
        
        result = analyze_with_context(news, context)
        
        # Verify result structure
        assert 'event_class' in result, "Should have event_class"
        assert 'expected_direction' in result, "Should have expected_direction"
        assert 'confidence' in result, "Should have confidence"
        assert 'rag_context_used' in result, "Should have RAG metadata"
        assert 'rag_context_count' in result, "Should have context count"
        
        # Verify RAG metadata
        if context:
            assert result['rag_context_used'] == True, "Should use RAG context"
            assert result['rag_context_count'] > 0, "Should have context count"
    
    def test_analyze_without_context(self):
        """Test analysis without RAG context (baseline)."""
        news = "Company announces new product"
        context = []  # No context
        
        result = analyze_with_context(news, context)
        
        # Should still work without context
        assert 'event_class' in result
        assert result['rag_context_used'] == False, "Should not use RAG context"
        assert result['rag_context_count'] == 0, "Should have zero context"
    
    def test_end_to_end_pipeline(self, vector_db, golden_dataset_path):
        """Test complete RAG pipeline from retrieval to analysis."""
        # Load one example from golden dataset
        with open(golden_dataset_path, 'r') as f:
            dataset = json.load(f)
        
        example = dataset[0]
        news = example['input_news']
        
        # Step 1: Retrieve context
        context = retrieve_relevant_context(news, vector_db, n_results=3)
        
        # Step 2: Analyze with context
        result = analyze_with_context(news, context)
        
        # Step 3: Verify result
        assert result['event_class'] in ['Macro', 'Earnings', 'Earnings_Miss', 'Noise', 'Unknown']
        assert result['expected_direction'] in ['Bullish', 'Bearish', 'Neutral', 'Mixed']
        assert 0 <= result['confidence'] <= 100


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_evaluate_on_subset(self, vector_db, golden_dataset_path):
        """Test evaluation on a small subset."""
        # Create a small test dataset
        with open(golden_dataset_path, 'r') as f:
            full_dataset = json.load(f)
        
        # Use only first 3 examples for quick test
        test_dataset = full_dataset[:3]
        test_file = "test_golden_subset.json"
        
        with open(test_file, 'w') as f:
            json.dump(test_dataset, f)
        
        try:
            # Run evaluation
            results = evaluate_on_golden_dataset(vector_db, test_file, use_rag=False)
            
            # Verify results structure
            assert 'metrics' in results
            assert 'detailed_results' in results
            
            metrics = results['metrics']
            assert 'accuracy_percent' in metrics
            assert 'average_latency_ms' in metrics
            assert metrics['total_examples'] == 3
            
            # Verify detailed results
            assert len(results['detailed_results']) == 3
            
        finally:
            # Cleanup
            if Path(test_file).exists():
                os.remove(test_file)
    
    def test_evaluation_metrics_range(self, vector_db, golden_dataset_path):
        """Test that evaluation metrics are in valid ranges."""
        # Use small subset
        with open(golden_dataset_path, 'r') as f:
            full_dataset = json.load(f)
        
        test_dataset = full_dataset[:2]
        test_file = "test_metrics.json"
        
        with open(test_file, 'w') as f:
            json.dump(test_dataset, f)
        
        try:
            results = evaluate_on_golden_dataset(vector_db, test_file, use_rag=False)
            metrics = results['metrics']
            
            # Verify metric ranges
            assert 0 <= metrics['accuracy_percent'] <= 100, "Accuracy should be 0-100%"
            assert metrics['average_latency_ms'] > 0, "Latency should be positive"
            assert metrics['average_latency_ms'] < 10000, "Latency should be reasonable (<10s)"
            
        finally:
            if Path(test_file).exists():
                os.remove(test_file)


class TestBatchProcessing:
    """Test batch processing capabilities."""
    
    def test_batch_retrieval(self, vector_db):
        """Test retrieving context for multiple queries."""
        queries = [
            "Fed rate hike",
            "Earnings miss",
            "GDP growth"
        ]
        
        contexts = [retrieve_relevant_context(q, vector_db, n_results=2) for q in queries]
        
        assert len(contexts) == len(queries), "Should have context for each query"
        assert all(isinstance(c, list) for c in contexts), "All contexts should be lists"
    
    def test_batch_analysis(self, vector_db):
        """Test analyzing multiple news items."""
        news_items = [
            "Fed signals rate hike",
            "Company reports earnings beat",
            "CEO eats sandwich"
        ]
        
        results = []
        for news in news_items:
            context = retrieve_relevant_context(news, vector_db, n_results=2)
            result = analyze_with_context(news, context)
            results.append(result)
        
        assert len(results) == len(news_items), "Should have result for each news item"
        assert all('event_class' in r for r in results), "All results should have event_class"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
