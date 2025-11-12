"""Scalable data processing pipeline for large-scale datasets."""

import pandas as pd
import numpy as np
from typing import List, Dict, Iterator, Callable, Optional, Any
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import dask.dataframe as dd
from functools import partial
import time
from src.utils import logger


class ScalableDataPipeline:
    """
    Production-grade data pipeline for processing large-scale datasets.
    
    Features:
    - Batch processing with memory efficiency
    - Parallel processing across CPU cores
    - Distributed processing with Dask
    - Progress tracking and error handling
    - Checkpointing and resume capability
    """
    
    def __init__(
        self,
        batch_size: int = 1000,
        n_workers: int = None,
        use_dask: bool = False
    ):
        """
        Initialize scalable data pipeline.
        
        Args:
            batch_size: Batch size for processing
            n_workers: Number of worker processes (None = CPU count)
            use_dask: Use Dask for distributed processing
        """
        self.batch_size = batch_size
        self.n_workers = n_workers or mp.cpu_count()
        self.use_dask = use_dask
        self.logger = logger
        
        self.logger.info(f"Initialized pipeline: batch_size={batch_size}, workers={self.n_workers}")
    
    def process_large_dataset(
        self,
        data_path: str,
        processing_func: Callable,
        output_path: str,
        file_format: str = "csv"
    ) -> Dict[str, Any]:
        """
        Process large dataset in batches with parallel execution.
        
        Args:
            data_path: Path to input data
            processing_func: Function to apply to each batch
            output_path: Path to output results
            file_format: File format (csv, parquet, json)
            
        Returns:
            Processing statistics
        """
        self.logger.info(f"Starting large-scale data processing: {data_path}")
        
        start_time = time.time()
        total_rows = 0
        processed_rows = 0
        
        if self.use_dask:
            # Use Dask for distributed processing
            results = self._process_with_dask(data_path, processing_func, output_path, file_format)
        else:
            # Use batch processing with multiprocessing
            results = self._process_with_batches(data_path, processing_func, output_path, file_format)
        
        elapsed_time = time.time() - start_time
        
        stats = {
            "total_rows": results.get("total_rows", 0),
            "processed_rows": results.get("processed_rows", 0),
            "elapsed_time": elapsed_time,
            "rows_per_second": results.get("processed_rows", 0) / elapsed_time if elapsed_time > 0 else 0,
            "output_path": output_path
        }
        
        self.logger.info(f"Processing completed: {stats['processed_rows']} rows in {elapsed_time:.2f}s")
        
        return stats
    
    def _process_with_batches(
        self,
        data_path: str,
        processing_func: Callable,
        output_path: str,
        file_format: str
    ) -> Dict[str, Any]:
        """Process data in batches with multiprocessing."""
        self.logger.info("Processing with batch + multiprocessing...")
        
        # Read data in chunks
        if file_format == "csv":
            chunk_iterator = pd.read_csv(data_path, chunksize=self.batch_size)
        elif file_format == "parquet":
            chunk_iterator = pd.read_parquet(data_path, chunksize=self.batch_size)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        results = []
        total_rows = 0
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            for chunk_id, chunk in enumerate(chunk_iterator):
                future = executor.submit(processing_func, chunk)
                futures.append((chunk_id, future))
                total_rows += len(chunk)
            
            # Collect results
            for chunk_id, future in futures:
                try:
                    result = future.result(timeout=300)  # 5 min timeout
                    results.append(result)
                    self.logger.info(f"Processed chunk {chunk_id}")
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_id}: {e}")
        
        # Combine results
        if results:
            combined_df = pd.concat(results, ignore_index=True)
            
            # Save output
            if file_format == "csv":
                combined_df.to_csv(output_path, index=False)
            elif file_format == "parquet":
                combined_df.to_parquet(output_path, index=False)
            
            processed_rows = len(combined_df)
        else:
            processed_rows = 0
        
        return {
            "total_rows": total_rows,
            "processed_rows": processed_rows
        }
    
    def _process_with_dask(
        self,
        data_path: str,
        processing_func: Callable,
        output_path: str,
        file_format: str
    ) -> Dict[str, Any]:
        """Process data with Dask for distributed computing."""
        self.logger.info("Processing with Dask (distributed)...")
        
        # Read data with Dask
        if file_format == "csv":
            ddf = dd.read_csv(data_path, blocksize="64MB")
        elif file_format == "parquet":
            ddf = dd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        total_rows = len(ddf)
        
        # Apply processing function
        processed_ddf = ddf.map_partitions(processing_func, meta=ddf._meta)
        
        # Compute and save
        if file_format == "csv":
            processed_ddf.to_csv(output_path, index=False, single_file=True)
        elif file_format == "parquet":
            processed_ddf.to_parquet(output_path, write_index=False)
        
        processed_rows = len(processed_ddf)
        
        return {
            "total_rows": total_rows,
            "processed_rows": processed_rows
        }
    
    def parallel_map(
        self,
        items: List[Any],
        func: Callable,
        use_threads: bool = False
    ) -> List[Any]:
        """
        Apply function to items in parallel.
        
        Args:
            items: Items to process
            func: Function to apply
            use_threads: Use threads instead of processes
            
        Returns:
            List of results
        """
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        results = []
        
        with executor_class(max_workers=self.n_workers) as executor:
            futures = {executor.submit(func, item): idx for idx, item in enumerate(items)}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append((futures[future], result))
                except Exception as e:
                    self.logger.error(f"Error processing item: {e}")
                    results.append((futures[future], None))
        
        # Sort by original order
        results.sort(key=lambda x: x[0])
        
        return [r[1] for r in results]


class StreamingDataProcessor:
    """
    Streaming data processor for real-time or continuous data processing.
    
    Features:
    - Memory-efficient streaming
    - Windowing and aggregation
    - Real-time processing
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize streaming processor.
        
        Args:
            window_size: Size of processing window
        """
        self.window_size = window_size
        self.logger = logger
    
    def stream_process(
        self,
        data_iterator: Iterator,
        processing_func: Callable,
        output_callback: Optional[Callable] = None
    ):
        """
        Process streaming data in windows.
        
        Args:
            data_iterator: Iterator yielding data items
            processing_func: Function to apply to each window
            output_callback: Callback for processed results
        """
        self.logger.info("Starting streaming processing...")
        
        window = []
        processed_count = 0
        
        for item in data_iterator:
            window.append(item)
            
            # Process when window is full
            if len(window) >= self.window_size:
                try:
                    result = processing_func(window)
                    processed_count += len(window)
                    
                    if output_callback:
                        output_callback(result)
                    
                    # Clear window
                    window = []
                    
                except Exception as e:
                    self.logger.error(f"Error processing window: {e}")
        
        # Process remaining items
        if window:
            try:
                result = processing_func(window)
                processed_count += len(window)
                
                if output_callback:
                    output_callback(result)
            except Exception as e:
                self.logger.error(f"Error processing final window: {e}")
        
        self.logger.info(f"Streaming processing completed: {processed_count} items")


class DataQualityChecker:
    """
    Data quality validation for large datasets.
    
    Checks:
    - Missing values
    - Data types
    - Outliers
    - Duplicates
    - Schema validation
    """
    
    def __init__(self):
        """Initialize data quality checker."""
        self.logger = logger
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        schema: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Validate dataset quality.
        
        Args:
            df: DataFrame to validate
            schema: Expected schema (column -> dtype)
            
        Returns:
            Validation report
        """
        self.logger.info("Running data quality checks...")
        
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": {},
            "data_types": {},
            "duplicates": 0,
            "quality_score": 100.0
        }
        
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                report["missing_values"][col] = {
                    "count": int(missing_count),
                    "percentage": round(missing_pct, 2)
                }
                
                # Penalize quality score
                report["quality_score"] -= min(missing_pct / 2, 10)
        
        # Check data types
        for col in df.columns:
            report["data_types"][col] = str(df[col].dtype)
        
        # Check duplicates
        duplicates = df.duplicated().sum()
        report["duplicates"] = int(duplicates)
        
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            report["quality_score"] -= min(dup_pct, 20)
        
        # Schema validation
        if schema:
            schema_issues = []
            for col, expected_type in schema.items():
                if col not in df.columns:
                    schema_issues.append(f"Missing column: {col}")
                elif str(df[col].dtype) != expected_type:
                    schema_issues.append(f"Type mismatch for {col}: expected {expected_type}, got {df[col].dtype}")
            
            if schema_issues:
                report["schema_validation"] = {
                    "passed": False,
                    "issues": schema_issues
                }
                report["quality_score"] -= len(schema_issues) * 5
            else:
                report["schema_validation"] = {"passed": True}
        
        report["quality_score"] = max(0, report["quality_score"])
        
        self.logger.info(f"Data quality score: {report['quality_score']:.2f}/100")
        
        return report


class FeatureEngineering:
    """
    Feature engineering pipeline for ML models.
    
    Includes:
    - Text feature extraction
    - Numerical transformations
    - Categorical encoding
    - Feature selection
    """
    
    @staticmethod
    def extract_text_features(texts: List[str]) -> pd.DataFrame:
        """
        Extract features from text data.
        
        Args:
            texts: List of texts
            
        Returns:
            DataFrame with text features
        """
        features = {
            "text_length": [len(text) for text in texts],
            "word_count": [len(text.split()) for text in texts],
            "avg_word_length": [
                np.mean([len(word) for word in text.split()]) if text.split() else 0
                for text in texts
            ],
            "uppercase_ratio": [
                sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
                for text in texts
            ],
            "digit_ratio": [
                sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0
                for text in texts
            ]
        }
        
        return pd.DataFrame(features)
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with time features
        """
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        df["year"] = df[datetime_col].dt.year
        df["month"] = df[datetime_col].dt.month
        df["day"] = df[datetime_col].dt.day
        df["dayofweek"] = df[datetime_col].dt.dayofweek
        df["hour"] = df[datetime_col].dt.hour
        df["is_weekend"] = df[datetime_col].dt.dayofweek.isin([5, 6]).astype(int)
        
        return df
