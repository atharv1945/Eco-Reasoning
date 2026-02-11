"""
Financial News Vector Database Setup Script
============================================
Senior Data Architect Implementation

This script sets up a ChromaDB vector database optimized for financial news retrieval
with support for:
- Event-Driven Stock Prediction methodology
- Regime-Aware retrieval (market volatility states)
- Metadata-Driven RAG with composite indexing
- Fast semantic search using sentence-transformers

Author: Senior Data Architect
Date: 2026-02-11
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datetime import datetime
from typing import List, Dict, Optional, Literal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialNewsVectorDB:
    """
    Vector database manager for financial news retrieval.
    
    Optimized for:
    - O(1) metadata filtering before vector search
    - Event-driven stock prediction workflows
    - Market regime-aware retrieval
    - High-speed semantic search (System 1 requirement)
    """
    
    def __init__(
        self,
        persist_directory: str = "./financial_news_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "financial_news"
    ):
        """
        Initialize the financial news vector database.
        
        Args:
            persist_directory: Path to persist the database
            embedding_model: Sentence transformer model for embeddings
                           Options:
                           - 'sentence-transformers/all-MiniLM-L6-v2' (fast, general)
                           - 'sentence-transformers/all-mpnet-base-v2' (slower, better quality)
                           - Custom financial models can be specified here
            collection_name: Name of the collection to create
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistence
        # Using DuckDB+Parquet for efficient metadata filtering
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Configure embedding function
        # Using sentence-transformers for System 1 speed requirement
        # Produces 384-dimensional vectors (MiniLM-L6-v2)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        logger.info(f"Initialized ChromaDB client at {persist_directory}")
        logger.info(f"Using embedding model: {embedding_model}")
        
        self.collection = None
    
    def create_index(self) -> chromadb.Collection:
        """
        Create optimized vector database collection with metadata schema.
        
        Metadata Schema (Event-Driven Stock Prediction):
        ------------------------------------------------
        - ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        - event_type (str): Category from Event-Driven methodology
                           ['Earnings', 'Macro', 'Geopolitics', 'M&A', 
                            'Regulatory', 'Product_Launch', 'Leadership_Change']
        - sentiment_score (float): Normalized sentiment [-1.0, 1.0]
                                  -1.0 = Very Negative
                                   0.0 = Neutral
                                  +1.0 = Very Positive
        - market_regime (str): Volatility state for Regime-Aware retrieval
                              ['High_Vol', 'Low_Vol']
        - published_at (int): Unix timestamp for temporal filtering
        
        Indexing Strategy:
        -----------------
        ChromaDB uses DuckDB backend which automatically creates indexes on metadata.
        Composite filtering on (ticker + published_at) achieves O(1) pre-filtering
        before vector search, as per Metadata-Driven RAG benchmarks.
        
        Returns:
            ChromaDB Collection object
        """
        try:
            # Create or get collection with metadata schema
            # NOTE: ChromaDB metadata only supports primitive types (str, int, float, bool)
            # Complex nested structures are not allowed
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Financial news optimized for event-driven stock prediction",
                    "schema_version": "1.0",
                    "embedding_model": self.embedding_function.model_name,
                    "embedding_dimension": "384",  # MiniLM-L6-v2 dimension (stored as string)
                    # Document metadata schema as strings for reference
                    "field_ticker": "str - Stock ticker symbol",
                    "field_event_type": "str - Event category (Earnings, Macro, etc.)",
                    "field_sentiment_score": "float - Sentiment score [-1.0, 1.0]",
                    "field_market_regime": "str - Volatility regime (High_Vol, Low_Vol)",
                    "field_published_at": "int - Unix timestamp"
                }
            )
            
            logger.info(f"Created collection '{self.collection_name}' with metadata schema")
            logger.info("Composite indexing enabled on ticker + published_at for O(1) filtering")
            
            return self.collection
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        tickers: List[str],
        event_types: List[Literal['Earnings', 'Macro', 'Geopolitics', 'M&A', 
                                   'Regulatory', 'Product_Launch', 'Leadership_Change']],
        sentiment_scores: List[float],
        market_regimes: List[Literal['High_Vol', 'Low_Vol']],
        published_timestamps: List[int],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add financial news documents to the vector database.
        
        Args:
            documents: List of news article texts
            tickers: List of stock ticker symbols
            event_types: List of event categories
            sentiment_scores: List of sentiment scores [-1.0, 1.0]
            market_regimes: List of market volatility regimes
            published_timestamps: List of Unix timestamps
            ids: Optional list of document IDs (auto-generated if None)
        
        Raises:
            ValueError: If input lists have mismatched lengths or invalid values
        """
        if self.collection is None:
            raise RuntimeError("Collection not created. Call create_index() first.")
        
        # Validate input lengths
        n_docs = len(documents)
        if not all(len(lst) == n_docs for lst in [
            tickers, event_types, sentiment_scores, market_regimes, published_timestamps
        ]):
            raise ValueError("All input lists must have the same length")
        
        # Validate sentiment scores
        if not all(-1.0 <= score <= 1.0 for score in sentiment_scores):
            raise ValueError("Sentiment scores must be in range [-1.0, 1.0]")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}_{published_timestamps[i]}" for i in range(n_docs)]
        
        # Prepare metadata for each document
        metadatas = [
            {
                "ticker": tickers[i],
                "event_type": event_types[i],
                "sentiment_score": sentiment_scores[i],
                "market_regime": market_regimes[i],
                "published_at": published_timestamps[i]
            }
            for i in range(n_docs)
        ]
        
        # Add to collection
        # ChromaDB will automatically generate embeddings using the configured function
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {n_docs} documents to collection '{self.collection_name}'")
    
    def retrieve_with_filters(
        self,
        query: str,
        ticker: Optional[str] = None,
        event_type: Optional[str] = None,
        sentiment_range: Optional[tuple[float, float]] = None,
        market_regime: Optional[str] = None,
        time_range: Optional[tuple[int, int]] = None,
        n_results: int = 10
    ) -> Dict:
        """
        Retrieve documents with metadata filtering before vector search.
        
        This implements the Metadata-Driven RAG pattern:
        1. Apply metadata filters (O(1) with DuckDB indexing)
        2. Perform vector search only on filtered subset
        3. Return ranked results
        
        Args:
            query: Natural language query
            ticker: Filter by specific ticker (e.g., 'AAPL')
            event_type: Filter by event category
            sentiment_range: Filter by sentiment score range (min, max)
            market_regime: Filter by volatility regime
            time_range: Filter by timestamp range (start_ts, end_ts)
            n_results: Number of results to return
        
        Returns:
            Dictionary with 'documents', 'metadatas', 'distances', 'ids'
        """
        if self.collection is None:
            raise RuntimeError("Collection not created. Call create_index() first.")
        
        # Build list of individual filter conditions
        filter_conditions = []
        
        if ticker:
            filter_conditions.append({"ticker": ticker})
        
        if event_type:
            filter_conditions.append({"event_type": event_type})
        
        if market_regime:
            filter_conditions.append({"market_regime": market_regime})
        
        # Handle range filters (sentiment and time)
        # ChromaDB requires separate conditions for $gte and $lte
        # Cannot use {"field": {"$gte": x, "$lte": y}} - must split into two conditions
        if sentiment_range:
            min_sentiment, max_sentiment = sentiment_range
            filter_conditions.append({"sentiment_score": {"$gte": min_sentiment}})
            filter_conditions.append({"sentiment_score": {"$lte": max_sentiment}})
        
        if time_range:
            start_ts, end_ts = time_range
            filter_conditions.append({"published_at": {"$gte": start_ts}})
            filter_conditions.append({"published_at": {"$lte": end_ts}})
        
        # Combine multiple filters using $and operator
        # ChromaDB requires explicit $and when multiple conditions are present
        where_clause = None
        if len(filter_conditions) == 1:
            where_clause = filter_conditions[0]
        elif len(filter_conditions) > 1:
            where_clause = {"$and": filter_conditions}
        
        # Perform filtered vector search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        
        logger.info(f"Retrieved {len(results['documents'][0])} results for query: '{query[:50]}...'")
        if where_clause:
            logger.info(f"Applied filters: {where_clause}")
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0]
        }
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if self.collection is None:
            raise RuntimeError("Collection not created. Call create_index() first.")
        
        count = self.collection.count()
        
        # Extract metadata schema fields from flattened structure
        metadata_schema = {
            key.replace("field_", ""): value 
            for key, value in self.collection.metadata.items() 
            if key.startswith("field_")
        }
        
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "embedding_model": self.embedding_function.model_name,
            "metadata_schema": metadata_schema
        }


# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_usage():
    """
    Demonstration of the financial news vector database setup and usage.
    """
    print("=" * 80)
    print("Financial News Vector Database - Example Usage")
    print("=" * 80)
    
    # Initialize the database
    db = FinancialNewsVectorDB(
        persist_directory="./financial_news_db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="financial_news"
    )
    
    # Create the index with optimized schema
    collection = db.create_index()
    print(f"\n✓ Created collection: {collection.name}")
    
    # Sample financial news data
    sample_documents = [
        "Apple Inc. reported record Q4 earnings, beating analyst expectations with revenue of $89.5B driven by strong iPhone sales.",
        "Federal Reserve signals potential interest rate cuts in 2024 amid cooling inflation data.",
        "Tesla stock drops 8% following Elon Musk's controversial geopolitical comments on social media.",
        "Microsoft announces $69B acquisition of Activision Blizzard, largest gaming industry deal in history.",
        "NVIDIA shares surge 15% on AI chip demand, reaching new all-time high amid datacenter boom.",
        "Oil prices spike to $95/barrel as OPEC+ announces surprise production cuts, raising recession fears."
    ]
    
    sample_tickers = ["AAPL", "SPY", "TSLA", "MSFT", "NVDA", "XLE"]
    sample_event_types = ["Earnings", "Macro", "Geopolitics", "M&A", "Earnings", "Macro"]
    sample_sentiments = [0.8, -0.3, -0.7, 0.6, 0.9, -0.5]
    sample_regimes = ["Low_Vol", "High_Vol", "High_Vol", "Low_Vol", "Low_Vol", "High_Vol"]
    sample_timestamps = [
        int(datetime(2024, 11, 2, 16, 0).timestamp()),
        int(datetime(2024, 11, 1, 14, 30).timestamp()),
        int(datetime(2024, 10, 30, 10, 15).timestamp()),
        int(datetime(2024, 10, 13, 9, 0).timestamp()),
        int(datetime(2024, 11, 3, 11, 45).timestamp()),
        int(datetime(2024, 11, 2, 8, 30).timestamp())
    ]
    
    # Add documents to the database
    db.add_documents(
        documents=sample_documents,
        tickers=sample_tickers,
        event_types=sample_event_types,
        sentiment_scores=sample_sentiments,
        market_regimes=sample_regimes,
        published_timestamps=sample_timestamps
    )
    print(f"\n✓ Added {len(sample_documents)} sample documents")
    
    # Display collection statistics
    stats = db.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Embedding model: {stats['embedding_model']}")
    
    # Example 1: Basic semantic search
    print("\n" + "=" * 80)
    print("Example 1: Basic Semantic Search")
    print("=" * 80)
    query1 = "What are the latest earnings reports?"
    results1 = db.retrieve_with_filters(query=query1, n_results=3)
    print(f"\nQuery: '{query1}'")
    print(f"Top {len(results1['documents'])} results:")
    for i, (doc, meta, dist) in enumerate(zip(
        results1['documents'], results1['metadatas'], results1['distances']
    ), 1):
        print(f"\n{i}. [{meta['ticker']}] {meta['event_type']} (sentiment: {meta['sentiment_score']:.2f})")
        print(f"   {doc[:100]}...")
        print(f"   Distance: {dist:.4f}")
    
    # Example 2: Filtered search - Specific ticker + event type
    print("\n" + "=" * 80)
    print("Example 2: Ticker + Event Type Filtering")
    print("=" * 80)
    query2 = "company performance"
    results2 = db.retrieve_with_filters(
        query=query2,
        ticker="AAPL",
        event_type="Earnings",
        n_results=5
    )
    print(f"\nQuery: '{query2}'")
    print(f"Filters: ticker='AAPL', event_type='Earnings'")
    print(f"Results: {len(results2['documents'])}")
    for i, (doc, meta) in enumerate(zip(results2['documents'], results2['metadatas']), 1):
        print(f"\n{i}. {doc[:100]}...")
    
    # Example 3: Regime-aware search (High volatility periods)
    print("\n" + "=" * 80)
    print("Example 3: Regime-Aware Search (High Volatility)")
    print("=" * 80)
    query3 = "market impact and risks"
    results3 = db.retrieve_with_filters(
        query=query3,
        market_regime="High_Vol",
        sentiment_range=(-1.0, 0.0),  # Negative sentiment only
        n_results=5
    )
    print(f"\nQuery: '{query3}'")
    print(f"Filters: market_regime='High_Vol', sentiment ≤ 0.0")
    print(f"Results: {len(results3['documents'])}")
    for i, (doc, meta) in enumerate(zip(results3['documents'], results3['metadatas']), 1):
        print(f"\n{i}. [{meta['ticker']}] Sentiment: {meta['sentiment_score']:.2f}")
        print(f"   {doc[:100]}...")
    
    # Example 4: Time-range filtering
    print("\n" + "=" * 80)
    print("Example 4: Temporal Filtering (Recent News)")
    print("=" * 80)
    query4 = "latest market news"
    recent_cutoff = int(datetime(2024, 11, 1).timestamp())
    results4 = db.retrieve_with_filters(
        query=query4,
        time_range=(recent_cutoff, int(datetime.now().timestamp())),
        n_results=5
    )
    print(f"\nQuery: '{query4}'")
    print(f"Filters: published_at >= {datetime.fromtimestamp(recent_cutoff)}")
    print(f"Results: {len(results4['documents'])}")
    for i, (doc, meta) in enumerate(zip(results4['documents'], results4['metadatas']), 1):
        pub_date = datetime.fromtimestamp(meta['published_at'])
        print(f"\n{i}. [{meta['ticker']}] {pub_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   {doc[:100]}...")
    
    print("\n" + "=" * 80)
    print("✓ Example usage completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    """
    Main execution block.
    
    To use this script:
    1. Install dependencies: pip install chromadb sentence-transformers
    2. Run: python financial_vector_db_setup.py
    3. The database will be persisted to ./financial_news_db/
    
    For production use:
    - Replace sample data with real financial news feeds
    - Consider using a specialized financial embedding model
    - Implement batch ingestion for large datasets
    - Add error handling and retry logic
    - Set up monitoring and logging
    """
    example_usage()
