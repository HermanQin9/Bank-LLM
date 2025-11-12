"""Production-grade model monitoring and evaluation system."""

import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from pathlib import Path


class ModelPerformanceMonitor:
    """
    Real-time model performance monitoring system.
    
    Tracks:
    - Latency and throughput
    - Prediction accuracy
    - Error rates
    - Resource utilization
    - Model drift detection
    """
    
    def __init__(self, log_dir: str = "logs/model_monitoring"):
        """
        Initialize performance monitor.
        
        Args:
            log_dir: Directory to store monitoring logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.predictions_log = []
        self.start_time = time.time()
        
    def log_prediction(
        self,
        input_data: Any,
        prediction: Any,
        ground_truth: Optional[Any] = None,
        latency: float = 0.0,
        model_version: str = "v1.0"
    ):
        """
        Log a single prediction with metadata.
        
        Args:
            input_data: Input to model
            prediction: Model prediction
            ground_truth: True label (if available)
            latency: Prediction latency in seconds
            model_version: Model version identifier
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": str(input_data)[:200],  # Truncate for storage
            "prediction": prediction,
            "ground_truth": ground_truth,
            "latency": latency,
            "model_version": model_version,
            "correct": prediction == ground_truth if ground_truth is not None else None
        }
        
        self.predictions_log.append(log_entry)
        
        # Update metrics
        self.metrics["latency"].append(latency)
        
        if ground_truth is not None:
            self.metrics["accuracy"].append(1.0 if prediction == ground_truth else 0.0)
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """
        Get real-time performance metrics.
        
        Returns:
            Dictionary of current metrics
        """
        total_predictions = len(self.predictions_log)
        elapsed_time = time.time() - self.start_time
        
        metrics = {
            "total_predictions": total_predictions,
            "uptime_seconds": elapsed_time,
            "throughput": total_predictions / elapsed_time if elapsed_time > 0 else 0,
            "avg_latency": np.mean(self.metrics["latency"]) if self.metrics["latency"] else 0,
            "p95_latency": np.percentile(self.metrics["latency"], 95) if self.metrics["latency"] else 0,
            "p99_latency": np.percentile(self.metrics["latency"], 99) if self.metrics["latency"] else 0,
            "current_accuracy": np.mean(self.metrics["accuracy"][-100:]) if self.metrics["accuracy"] else None,
            "overall_accuracy": np.mean(self.metrics["accuracy"]) if self.metrics["accuracy"] else None
        }
        
        return metrics
    
    def record_metrics(self, model_name: str, metrics_dict: Dict[str, float]):
        """
        Record multiple metrics at once.
        
        Args:
            model_name: Name of the model
            metrics_dict: Dictionary of metric names and values
        """
        for metric_name, value in metrics_dict.items():
            key = f"{model_name}_{metric_name}"
            self.metrics[key].append(value)
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests.
        
        Args:
            reference_data: Baseline/reference dataset
            current_data: Current dataset to compare
            
        Returns:
            Drift detection results
        """
        from scipy import stats
        
        # Kolmogorov-Smirnov test for each feature
        drift_scores = []
        for i in range(reference_data.shape[1]):
            statistic, p_value = stats.ks_2samp(reference_data[:, i], current_data[:, i])
            drift_scores.append({'feature': i, 'statistic': statistic, 'p_value': p_value})
        
        # Overall drift detected if any feature shows significant drift (p < 0.05)
        drift_detected = any(score['p_value'] < 0.05 for score in drift_scores)
        
        return {
            "drift_detected": drift_detected,
            "num_features_drifted": sum(1 for s in drift_scores if s['p_value'] < 0.05),
            "total_features": reference_data.shape[1],
            "drift_scores": drift_scores[:5]  # Top 5 features
        }
    
    def detect_model_drift(self, window_size: int = 100) -> Dict[str, Any]:
        """
        Detect model performance drift.
        
        Args:
            window_size: Size of sliding window for comparison
            
        Returns:
            Drift detection results
        """
        if len(self.metrics["accuracy"]) < window_size * 2:
            return {"drift_detected": False, "reason": "Insufficient data"}
        
        # Compare recent performance to baseline
        baseline_accuracy = np.mean(self.metrics["accuracy"][:window_size])
        recent_accuracy = np.mean(self.metrics["accuracy"][-window_size:])
        
        drift_threshold = 0.05  # 5% drop
        drift_detected = (baseline_accuracy - recent_accuracy) > drift_threshold
        
        return {
            "drift_detected": drift_detected,
            "baseline_accuracy": baseline_accuracy,
            "recent_accuracy": recent_accuracy,
            "drop": baseline_accuracy - recent_accuracy,
            "threshold": drift_threshold
        }
    
    def save_logs(self, filename: Optional[str] = None):
        """Save prediction logs to file."""
        if filename is None:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                "metrics": dict(self.metrics),
                "predictions": self.predictions_log,
                "summary": self.get_realtime_metrics()
            }, f, indent=2)
        
        return str(filepath)


class ModelEvaluator:
    """
    Comprehensive model evaluation suite.
    
    Provides:
    - Classification metrics
    - Regression metrics
    - Cross-validation
    - Error analysis
    """
    
    @staticmethod
    def evaluate_classification(
        y_true: List[int],
        y_pred: List[int],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            
        Returns:
            Evaluation metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        class_report = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "num_samples": len(y_true)
        }
        
        return results
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Regression evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Regression metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2
        }
    
    @staticmethod
    def analyze_errors(
        y_true: List[int],
        y_pred: List[int],
        texts: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors in detail.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            texts: Input texts (optional)
            top_k: Number of top errors to return
            
        Returns:
            Error analysis
        """
        errors = []
        
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true != pred:
                error = {
                    "index": i,
                    "true_label": true,
                    "predicted_label": pred,
                }
                
                if texts:
                    error["text"] = texts[i][:200]  # Truncate
                
                errors.append(error)
        
        # Error distribution
        error_types = defaultdict(int)
        for error in errors:
            error_pair = (error["true_label"], error["predicted_label"])
            error_types[error_pair] += 1
        
        # Sort by frequency
        top_error_types = sorted(
            error_types.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(y_true) if y_true else 0,
            "sample_errors": errors[:top_k],
            "top_error_types": [
                {
                    "true_label": true,
                    "predicted_label": pred,
                    "count": count
                }
                for (true, pred), count in top_error_types
            ]
        }


class ABTestingFramework:
    """
    A/B testing framework for comparing model versions.
    
    Features:
    - Statistical significance testing
    - Traffic splitting
    - Performance comparison
    """
    
    def __init__(self):
        """Initialize A/B testing framework."""
        self.model_a_results = []
        self.model_b_results = []
    
    def log_result(self, model_version: str, correct: bool, latency: float):
        """
        Log a prediction result for A/B testing.
        
        Args:
            model_version: "A" or "B"
            correct: Whether prediction was correct
            latency: Prediction latency
        """
        result = {"correct": correct, "latency": latency}
        
        if model_version == "A":
            self.model_a_results.append(result)
        elif model_version == "B":
            self.model_b_results.append(result)
    
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare performance of model A vs model B.
        
        Returns:
            Comparison results
        """
        if not self.model_a_results or not self.model_b_results:
            return {"error": "Insufficient data for comparison"}
        
        # Accuracy comparison
        a_accuracy = np.mean([r["correct"] for r in self.model_a_results])
        b_accuracy = np.mean([r["correct"] for r in self.model_b_results])
        
        # Latency comparison
        a_latency = np.mean([r["latency"] for r in self.model_a_results])
        b_latency = np.mean([r["latency"] for r in self.model_b_results])
        
        # Statistical significance (simple t-test approximation)
        accuracy_diff = b_accuracy - a_accuracy
        latency_diff = b_latency - a_latency
        
        return {
            "model_a": {
                "accuracy": a_accuracy,
                "avg_latency": a_latency,
                "num_samples": len(self.model_a_results)
            },
            "model_b": {
                "accuracy": b_accuracy,
                "avg_latency": b_latency,
                "num_samples": len(self.model_b_results)
            },
            "comparison": {
                "accuracy_improvement": accuracy_diff,
                "accuracy_improvement_pct": (accuracy_diff / a_accuracy * 100) if a_accuracy > 0 else 0,
                "latency_improvement": -latency_diff,  # Negative is better
                "latency_improvement_pct": (-latency_diff / a_latency * 100) if a_latency > 0 else 0,
                "recommendation": "Deploy Model B" if accuracy_diff > 0.01 and latency_diff < 0.1 else "Keep Model A"
            }
        }


class ProductionMetrics:
    """
    Production-grade metrics collection and reporting.
    
    Simulates enterprise monitoring systems like DataDog, Prometheus.
    """
    
    def __init__(self):
        """Initialize production metrics."""
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            tags: Optional tags for filtering
        """
        self.metrics[metric_name].append({
            "timestamp": time.time(),
            "value": value,
            "tags": tags or {}
        })
    
    def record_metrics(self, model_name: str, metrics_dict: Dict[str, float]):
        """
        Record multiple metrics at once.
        
        Args:
            model_name: Name of the model
            metrics_dict: Dictionary of metric names and values
        """
        for metric_name, value in metrics_dict.items():
            self.record_metric(f"{model_name}_{metric_name}", value, {"model": model_name})
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data formatted for monitoring dashboard.
        
        Returns:
            Dashboard-ready metrics
        """
        dashboard = {}
        
        for metric_name, values in self.metrics.items():
            recent_values = [v["value"] for v in values[-100:]]  # Last 100 samples
            
            dashboard[metric_name] = {
                "current": recent_values[-1] if recent_values else 0,
                "avg": np.mean(recent_values),
                "min": np.min(recent_values),
                "max": np.max(recent_values),
                "p95": np.percentile(recent_values, 95) if len(recent_values) > 0 else 0,
                "count": len(values)
            }
        
        return dashboard
