"""GPU-accelerated deep learning training pipeline for document understanding."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
from src.utils import logger

# Optional transformers import with graceful fallback
try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transformers not available: {e}. GPU training features will be limited.")
    TRANSFORMERS_AVAILABLE = False
    # Create dummy classes for type hints
    class AutoModelForSequenceClassification: pass
    class AutoTokenizer: pass
    class Trainer: pass
    class TrainingArguments: pass


class DocumentDataset(Dataset):
    """PyTorch Dataset for document classification/extraction tasks."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of document texts
            labels: List of labels
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class GPUAcceleratedTrainer:
    """
    GPU-accelerated model training for document classification.
    
    Features:
    - Multi-GPU training support
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Model checkpointing
    - Distributed training ready
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 5,
        use_gpu: bool = True
    ):
        """
        Initialize GPU-accelerated trainer.
        
        Args:
            model_name: Pretrained model name
            num_labels: Number of classification labels
            use_gpu: Whether to use GPU acceleration
        """
        self.logger = logger
        
        # Check GPU availability
        self.device = self._setup_device(use_gpu)
        self.logger.info(f"Using device: {self.device}")
        
        # Log GPU information
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Move model to device
        self.model.to(self.device)
        
        self.logger.info(f"Loaded model: {model_name}")
    
    def _setup_device(self, use_gpu: bool) -> torch.device:
        """Setup computing device (GPU/CPU)."""
        if use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        elif use_gpu and not torch.cuda.is_available():
            self.logger.warning("GPU requested but not available. Using CPU.")
            return torch.device("cpu")
        else:
            return torch.device("cpu")
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        output_dir: str = "models/document_classifier",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        use_fp16: bool = True
    ) -> Dict[str, float]:
        """
        Train the model with GPU acceleration.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            output_dir: Output directory for model checkpoints
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_fp16: Use mixed precision training (FP16)
            
        Returns:
            Training metrics
        """
        self.logger.info("Starting GPU-accelerated training...")
        
        # Create datasets
        train_dataset = DocumentDataset(
            train_texts,
            train_labels,
            self.tokenizer
        )
        
        eval_dataset = None
        if val_texts and val_labels:
            eval_dataset = DocumentDataset(
                val_texts,
                val_labels,
                self.tokenizer
            )
        
        # Training arguments with GPU optimizations
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            # GPU optimizations
            fp16=use_fp16 and torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            dataloader_num_workers=4,
            # Multi-GPU support
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        self.logger.info("Training started...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info("Training completed!")
        
        # Return metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"]
        }
        
        return metrics
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with GPU acceleration.
        
        Args:
            texts: Texts to classify
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        
        # Create dataloader
        dataset = DocumentDataset(
            texts,
            [0] * len(texts),  # Dummy labels
            self.tokenizer
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def fine_tune_on_domain(
        self,
        domain_texts: List[str],
        domain_labels: List[int],
        output_dir: str = "models/domain_finetuned"
    ):
        """
        Fine-tune model on domain-specific data (e.g., financial documents).
        
        Args:
            domain_texts: Domain-specific texts
            domain_labels: Domain-specific labels
            output_dir: Output directory
        """
        self.logger.info("Starting domain-specific fine-tuning...")
        
        return self.train(
            train_texts=domain_texts,
            train_labels=domain_labels,
            output_dir=output_dir,
            epochs=5,
            batch_size=8,
            learning_rate=1e-5,
            use_fp16=True
        )


class DistributedTrainingManager:
    """
    Manager for distributed training across multiple GPUs.
    
    Demonstrates:
    - Data parallelism
    - Model parallelism awareness
    - Gradient synchronization
    """
    
    def __init__(self):
        """Initialize distributed training manager."""
        self.logger = logger
        self.num_gpus = torch.cuda.device_count()
        
        self.logger.info(f"Available GPUs: {self.num_gpus}")
        
        for i in range(self.num_gpus):
            self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    def setup_distributed(self):
        """Setup distributed training environment."""
        if self.num_gpus > 1:
            self.logger.info("Setting up distributed training...")
            # In production, this would include torch.distributed.init_process_group()
            # For now, we'll use DataParallel
            return True
        return False
    
    def wrap_model_for_distributed(self, model: nn.Module) -> nn.Module:
        """
        Wrap model for distributed training.
        
        Args:
            model: PyTorch model
            
        Returns:
            Wrapped model
        """
        if self.num_gpus > 1:
            self.logger.info("Wrapping model for multi-GPU training...")
            model = nn.DataParallel(model)
        
        return model


class ModelOptimizer:
    """
    Model optimization utilities for production deployment.
    
    Includes:
    - Model quantization
    - ONNX export
    - TorchScript compilation
    """
    
    @staticmethod
    def quantize_model(model: nn.Module) -> nn.Module:
        """
        Quantize model for faster inference with lower memory.
        
        Args:
            model: PyTorch model
            
        Returns:
            Quantized model
        """
        logger.info("Quantizing model...")
        
        # Dynamic quantization (good for LSTM, Transformer models)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        logger.info("Model quantization completed")
        
        return quantized_model
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str
    ):
        """
        Export model to ONNX format for cross-platform deployment.
        
        Args:
            model: PyTorch model
            dummy_input: Example input tensor
            output_path: Output file path
        """
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        model.eval()
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info("ONNX export completed")
    
    @staticmethod
    def compile_torchscript(model: nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """
        Compile model to TorchScript for optimized inference.
        
        Args:
            model: PyTorch model
            example_input: Example input
            
        Returns:
            TorchScript compiled model
        """
        logger.info("Compiling model to TorchScript...")
        
        model.eval()
        traced_model = torch.jit.trace(model, example_input)
        
        logger.info("TorchScript compilation completed")
        
        return traced_model
