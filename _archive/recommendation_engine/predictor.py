"""
SLO Predictor - ML-Based SLO Estimation

Uses transformer/attention mechanisms to predict SLOs based on:
- Model characteristics (size, architecture, params)
- Hardware specifications
- Workload patterns

This is optional - can be used when actual SLO measurements are not available.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os


@dataclass
class ModelFeatures:
    """Features extracted from model metadata"""
    total_params_b: float      # Total parameters in billions
    active_params_b: float     # Active parameters (for MoE models)
    is_moe: bool               # Is Mixture of Experts
    context_length: int        # Max context length
    architecture: str          # Architecture type (transformer, mamba, hybrid)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to input tensor"""
        return torch.tensor([
            self.total_params_b,
            self.active_params_b,
            1.0 if self.is_moe else 0.0,
            self.context_length / 1000.0,  # Normalize to k
            self._architecture_to_int(),
        ], dtype=torch.float32)
    
    def _architecture_to_int(self) -> float:
        arch_map = {
            'transformer': 0.0,
            'mamba': 1.0,
            'hybrid': 0.5,
            'unknown': 0.25,
        }
        return arch_map.get(self.architecture.lower(), 0.25)


@dataclass
class HardwareFeatures:
    """Features extracted from hardware specs"""
    memory_gb: float
    fp16_tflops: float
    cost_per_hour: float
    max_batch_size: int
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.memory_gb / 100.0,    # Normalize
            self.fp16_tflops / 1000.0, # Normalize
            self.cost_per_hour / 10.0, # Normalize
            self.max_batch_size / 100.0,
        ], dtype=torch.float32)


@dataclass
class WorkloadFeatures:
    """Features from workload specification"""
    rps_mean: float
    rps_p95: float
    avg_input_tokens: int
    avg_output_tokens: int
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.rps_mean,
            self.rps_p95,
            self.avg_input_tokens / 1000.0,
            self.avg_output_tokens / 100.0,
        ], dtype=torch.float32)


class SLOPredictorNetwork(nn.Module):
    """
    Neural network for SLO prediction using attention.
    
    Architecture:
    1. Feature encoders for model, hardware, workload
    2. Cross-attention between features
    3. MLP decoder for SLO predictions
    
    Outputs: (TTFT_p50, TTFT_p95, ITL_p50, ITL_p95, Throughput)
    """
    
    def __init__(
        self,
        model_dim: int = 5,
        hardware_dim: int = 4,
        workload_dim: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        output_dim: int = 5,  # TTFT_p50, TTFT_p95, ITL_p50, ITL_p95, Throughput
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Feature encoders
        self.model_encoder = nn.Linear(model_dim, hidden_dim)
        self.hardware_encoder = nn.Linear(hardware_dim, hidden_dim)
        self.workload_encoder = nn.Linear(workload_dim, hidden_dim)
        
        # Cross-attention: how model interacts with hardware
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Self-attention for combined features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),  # SLOs are always positive
        )
    
    def forward(
        self,
        model_features: torch.Tensor,
        hardware_features: torch.Tensor,
        workload_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            model_features: (batch, model_dim)
            hardware_features: (batch, hardware_dim)
            workload_features: (batch, workload_dim)
        
        Returns:
            slo_predictions: (batch, output_dim)
        """
        batch_size = model_features.shape[0]
        
        # Encode features
        model_enc = self.model_encoder(model_features)      # (batch, hidden)
        hardware_enc = self.hardware_encoder(hardware_features)
        workload_enc = self.workload_encoder(workload_features)
        
        # Reshape for attention: (batch, seq_len=1, hidden)
        model_enc = model_enc.unsqueeze(1)
        hardware_enc = hardware_enc.unsqueeze(1)
        workload_enc = workload_enc.unsqueeze(1)
        
        # Cross-attention: model attends to hardware
        model_hw, _ = self.cross_attention(
            query=model_enc,
            key=hardware_enc,
            value=hardware_enc,
        )
        
        # Combine all features
        combined = torch.cat([model_hw, hardware_enc, workload_enc], dim=1)  # (batch, 3, hidden)
        
        # Self-attention over combined
        attended, _ = self.self_attention(combined, combined, combined)
        
        # Pool and decode
        pooled = attended.mean(dim=1)  # (batch, hidden)
        
        # Also use skip connection
        skip = torch.cat([
            model_enc.squeeze(1),
            hardware_enc.squeeze(1),
            workload_enc.squeeze(1),
        ], dim=-1)  # (batch, hidden * 3)
        
        # Decode to SLOs
        output = self.decoder(skip)
        
        return output


class SLOPredictor:
    """
    High-level SLO predictor using neural network.
    
    Usage:
        predictor = SLOPredictor()
        predictor.load_model("slo_predictor.pt")  # Load trained weights
        
        slo = predictor.predict(
            model_info={"params": "70B", "architecture": "transformer"},
            hardware="A100_80GB",
            workload={"rps_mean": 1.0, "avg_output_tokens": 100}
        )
    """
    
    def __init__(self):
        self.network = SLOPredictorNetwork()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        self.network.eval()
        
        # Fallback heuristics when model not trained
        self._use_heuristics = True
    
    def load_model(self, path: str) -> bool:
        """Load trained model weights"""
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=self.device))
            self.network.eval()
            self._use_heuristics = False
            return True
        return False
    
    def predict(
        self,
        model_info: Dict,
        hardware: str,
        workload: Dict,
    ) -> Dict[str, float]:
        """
        Predict SLOs for a model-hardware-workload combination.
        
        Args:
            model_info: Model metadata
                {"params": "70B", "active_params": "70B", "architecture": "transformer", "context": 128000}
            hardware: Hardware config ID (e.g., "A100_80GB")
            workload: Workload specification
                {"rps_mean": 1.0, "rps_p95": 2.0, "avg_input_tokens": 500, "avg_output_tokens": 100}
        
        Returns:
            SLO predictions: {"ttft_p50": ..., "ttft_p95": ..., "itl_p50": ..., "itl_p95": ..., "throughput": ...}
        """
        
        if self._use_heuristics:
            return self._predict_with_heuristics(model_info, hardware, workload)
        
        # Extract features
        model_features = self._extract_model_features(model_info)
        hardware_features = self._extract_hardware_features(hardware)
        workload_features = self._extract_workload_features(workload)
        
        # Convert to tensors
        model_tensor = model_features.to_tensor().unsqueeze(0).to(self.device)
        hardware_tensor = hardware_features.to_tensor().unsqueeze(0).to(self.device)
        workload_tensor = workload_features.to_tensor().unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.network(model_tensor, hardware_tensor, workload_tensor)
        
        # Denormalize outputs
        output = output.squeeze(0).cpu().numpy()
        
        return {
            "ttft_p50": float(output[0] * 100),      # Scale back to ms
            "ttft_p95": float(output[1] * 100),
            "itl_p50": float(output[2] * 10),
            "itl_p95": float(output[3] * 10),
            "throughput_tokens_per_sec": float(output[4] * 100),
        }
    
    def _predict_with_heuristics(
        self,
        model_info: Dict,
        hardware: str,
        workload: Dict,
    ) -> Dict[str, float]:
        """
        Heuristic-based SLO prediction when no trained model available.
        
        Based on empirical observations:
        - TTFT scales with model size and input length
        - ITL scales with model size and hardware compute
        - Throughput scales with hardware compute and model efficiency
        """
        from .config import HARDWARE_CONFIGS
        
        # Parse model params
        params_str = model_info.get('params', '7B')
        total_params = self._parse_params(params_str)
        active_params = self._parse_params(model_info.get('active_params', params_str))
        
        # Get hardware specs
        hw_config = HARDWARE_CONFIGS.get(hardware, HARDWARE_CONFIGS.get('A100_40GB', {}))
        memory_gb = hw_config.get('memory_gb', 40)
        tflops = hw_config.get('fp16_tflops', 312)
        
        # Get workload params
        avg_input = workload.get('avg_input_tokens', 500)
        avg_output = workload.get('avg_output_tokens', 100)
        rps = workload.get('rps_mean', 1.0)
        
        # Heuristic formulas (calibrated from observations)
        
        # TTFT: dominated by prefill phase
        # Base ~50ms + scales with params and input length
        ttft_base = 50 + (active_params / 10) * 20 + (avg_input / 1000) * 30
        ttft_hw_factor = 312 / tflops  # Slower hardware = higher TTFT
        ttft_p50 = ttft_base * ttft_hw_factor
        ttft_p95 = ttft_p50 * 1.8  # P95 typically 1.5-2x P50
        
        # ITL: dominated by decode phase
        # Base ~10ms + scales with params
        itl_base = 10 + (active_params / 70) * 15
        itl_hw_factor = 312 / tflops
        itl_p50 = itl_base * itl_hw_factor
        itl_p95 = itl_p50 * 1.5
        
        # Throughput: tokens per second
        # Scales with hardware compute, inversely with model size
        throughput_base = 200 * (tflops / 312) * (10 / max(active_params, 1))
        throughput = max(10, min(500, throughput_base))  # Cap between 10-500
        
        return {
            "ttft_p50": round(ttft_p50, 1),
            "ttft_p95": round(ttft_p95, 1),
            "itl_p50": round(itl_p50, 1),
            "itl_p95": round(itl_p95, 1),
            "throughput_tokens_per_sec": round(throughput, 1),
        }
    
    def _parse_params(self, params_str: str) -> float:
        """Parse parameter string like '70B' or '70.6B' to float"""
        if isinstance(params_str, (int, float)):
            return float(params_str)
        
        params_str = str(params_str).upper().strip()
        
        # Handle MoE format: "236B (21B active)"
        if '(' in params_str:
            params_str = params_str.split('(')[0].strip()
        
        # Parse multiplier
        multiplier = 1.0
        if 'T' in params_str:
            multiplier = 1000.0
            params_str = params_str.replace('T', '').replace('B', '')
        elif 'B' in params_str:
            params_str = params_str.replace('B', '')
        elif 'M' in params_str:
            multiplier = 0.001
            params_str = params_str.replace('M', '')
        
        try:
            return float(params_str) * multiplier
        except ValueError:
            return 7.0  # Default to 7B if parsing fails
    
    def _extract_model_features(self, model_info: Dict) -> ModelFeatures:
        """Extract model features from info dict"""
        params = self._parse_params(model_info.get('params', '7B'))
        active = self._parse_params(model_info.get('active_params', model_info.get('params', '7B')))
        
        return ModelFeatures(
            total_params_b=params,
            active_params_b=active,
            is_moe=params != active,
            context_length=model_info.get('context', 8000),
            architecture=model_info.get('architecture', 'transformer'),
        )
    
    def _extract_hardware_features(self, hardware: str) -> HardwareFeatures:
        """Extract hardware features from config"""
        from .config import HARDWARE_CONFIGS
        
        hw_config = HARDWARE_CONFIGS.get(hardware, HARDWARE_CONFIGS.get('A100_40GB', {}))
        
        return HardwareFeatures(
            memory_gb=hw_config.get('memory_gb', 40),
            fp16_tflops=hw_config.get('fp16_tflops', 312),
            cost_per_hour=hw_config.get('cost_per_hour', 2.5),
            max_batch_size=hw_config.get('max_batch_size', 64),
        )
    
    def _extract_workload_features(self, workload: Dict) -> WorkloadFeatures:
        """Extract workload features"""
        return WorkloadFeatures(
            rps_mean=workload.get('rps_mean', 1.0),
            rps_p95=workload.get('rps_p95', 2.0),
            avg_input_tokens=workload.get('avg_input_tokens', 500),
            avg_output_tokens=workload.get('avg_output_tokens', 100),
        )


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES (for when you have ground truth SLO data)
# ═══════════════════════════════════════════════════════════════════════════

class SLOPredictorTrainer:
    """
    Trainer for the SLO prediction network.
    
    Usage:
        trainer = SLOPredictorTrainer()
        trainer.train(training_data, epochs=100)
        trainer.save_model("slo_predictor.pt")
    """
    
    def __init__(self, learning_rate: float = 0.001):
        self.network = SLOPredictorNetwork()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train(
        self,
        training_data: List[Dict],
        epochs: int = 100,
        batch_size: int = 32,
    ) -> List[float]:
        """
        Train the network on SLO measurement data.
        
        Args:
            training_data: List of training examples
                [{"model": {...}, "hardware": "A100", "workload": {...}, 
                  "slo": {"ttft_p50": ..., "ttft_p95": ..., ...}}]
            epochs: Number of training epochs
            batch_size: Batch size
        
        Returns:
            List of training losses per epoch
        """
        self.network.train()
        losses = []
        
        # Prepare data
        predictor = SLOPredictor()
        predictor.network = self.network  # Share network
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                # Extract features
                model_tensors = []
                hardware_tensors = []
                workload_tensors = []
                targets = []
                
                for item in batch:
                    model_feat = predictor._extract_model_features(item['model'])
                    hw_feat = predictor._extract_hardware_features(item['hardware'])
                    wl_feat = predictor._extract_workload_features(item['workload'])
                    
                    model_tensors.append(model_feat.to_tensor())
                    hardware_tensors.append(hw_feat.to_tensor())
                    workload_tensors.append(wl_feat.to_tensor())
                    
                    slo = item['slo']
                    targets.append(torch.tensor([
                        slo['ttft_p50'] / 100,
                        slo['ttft_p95'] / 100,
                        slo['itl_p50'] / 10,
                        slo['itl_p95'] / 10,
                        slo.get('throughput', 100) / 100,
                    ]))
                
                # Stack and move to device
                model_batch = torch.stack(model_tensors).to(self.device)
                hardware_batch = torch.stack(hardware_tensors).to(self.device)
                workload_batch = torch.stack(workload_tensors).to(self.device)
                target_batch = torch.stack(targets).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.network(model_batch, hardware_batch, workload_batch)
                
                # Compute loss
                loss = self.criterion(output, target_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(training_data) / batch_size)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def save_model(self, path: str):
        """Save trained model weights"""
        torch.save(self.network.state_dict(), path)
        print(f"Model saved to {path}")

