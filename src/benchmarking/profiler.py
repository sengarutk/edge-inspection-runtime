import time
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.mvtec import IMAGENET_MEAN, IMAGENET_STD


class CUDAPerformanceProfiler:
    """
    Synchronized Hardware Performance & Dual-Latency Profiler.
    Measures both:
      1. T_model (ms): Tensor-in to raw anomaly score/map via inference_mode().
      2. T_e2e (ms): End-to-end pipeline latency (Image read -> normalize -> CUDA transfer -> forward -> postprocess).
    """
    def __init__(self, warmup_runs: int = 50, active_runs: int = 300, device: Optional[str] = None):
        self.warmup_runs = warmup_runs
        self.active_runs = active_runs
        self.device = device

    def _resolve_device(self, model: Any, sample_input: Optional[torch.Tensor] = None) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        if sample_input is not None:
            return sample_input.device
        if isinstance(model, nn.Module):
            try:
                return next(model.parameters()).device
            except StopIteration:
                pass
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def profile(
        self,
        model_fn_or_module: Any,
        sample_input: torch.Tensor
    ) -> Dict[str, Any]:
        dev = self._resolve_device(model_fn_or_module, sample_input)
        is_cuda = (dev.type == "cuda")

        is_module = isinstance(model_fn_or_module, nn.Module)
        if is_module:
            model = model_fn_or_module.to(dev)
            model.eval()
            forward_fn = lambda x: model(x)
        elif hasattr(model_fn_or_module, "predict"):
            forward_fn = lambda x: model_fn_or_module.predict(x)
        else:
            forward_fn = model_fn_or_module

        sample_input = sample_input.to(dev)

        # Warmup
        with torch.inference_mode():
            for _ in range(self.warmup_runs):
                _ = forward_fn(sample_input)
        if is_cuda:
            torch.cuda.synchronize(dev)

        latencies: List[float] = []
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.active_runs)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.active_runs)]

            with torch.inference_mode():
                for i in range(self.active_runs):
                    start_events[i].record()
                    _ = forward_fn(sample_input)
                    end_events[i].record()

            torch.cuda.synchronize(dev)
            latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            peak_vram_bytes = torch.cuda.max_memory_allocated(dev)
            peak_vram_mb = peak_vram_bytes / (1024.0 * 1024.0)
        else:
            with torch.inference_mode():
                for _ in range(self.active_runs):
                    t0 = time.perf_counter()
                    _ = forward_fn(sample_input)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)
            peak_vram_mb = 0.0

        mean_ms = float(np.mean(latencies))
        std_ms = float(np.std(latencies))
        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        fps = float(1000.0 / p50) if p50 > 0 else 0.0

        return {
            "p50_ms": p50,
            "p90_ms": p90,
            "p95_ms": p95,
            "p99_ms": p99,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "fps": fps,
            "p50_model_ms": p50,
            "p95_model_ms": p95,
            "fps_model": fps,
            "peak_vram_mb": float(peak_vram_mb),
            "device": str(dev)
        }

    def profile_dual(
        self,
        model_fn_or_module: Any,
        sample_input: Optional[torch.Tensor] = None,
        input_shape: tuple = (1, 3, 256, 256),
        e2e_fn: Optional[Callable] = None,
        sample_raw_input: Optional[str] = None
    ) -> Dict[str, Any]:
        dev = self._resolve_device(model_fn_or_module, sample_input)
        is_cuda = (dev.type == "cuda")

        if sample_input is None:
            sample_input = torch.randn(input_shape, device=dev)
        else:
            sample_input = sample_input.to(dev)

        base_res = self.profile(model_fn_or_module, sample_input)

        # Profile E2E Pipeline
        e2e_latencies: List[float] = []
        if e2e_fn is not None and sample_raw_input is not None:
            with torch.inference_mode():
                for _ in range(min(self.active_runs, 100)):
                    t0 = time.perf_counter()
                    _ = e2e_fn(sample_raw_input)
                    if is_cuda:
                        torch.cuda.synchronize(dev)
                    t1 = time.perf_counter()
                    e2e_latencies.append((t1 - t0) * 1000.0)
        else:
            dummy_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            is_module = isinstance(model_fn_or_module, nn.Module)
            if is_module:
                forward_fn = lambda x: model_fn_or_module(x)
            elif hasattr(model_fn_or_module, "predict"):
                forward_fn = lambda x: model_fn_or_module.predict(x)
            else:
                forward_fn = model_fn_or_module

            with torch.inference_mode():
                for _ in range(min(self.active_runs, 100)):
                    t0 = time.perf_counter()
                    x_tensor = transform(dummy_img).unsqueeze(0).to(dev)
                    _ = forward_fn(x_tensor)
                    if is_cuda:
                        torch.cuda.synchronize(dev)
                    t1 = time.perf_counter()
                    e2e_latencies.append((t1 - t0) * 1000.0)

        p50_e2e = float(np.percentile(e2e_latencies, 50))
        p95_e2e = float(np.percentile(e2e_latencies, 95))
        fps_e2e = float(1000.0 / p50_e2e) if p50_e2e > 0 else 0.0

        base_res.update({
            "p50_e2e_ms": p50_e2e,
            "p95_e2e_ms": p95_e2e,
            "fps_e2e": fps_e2e
        })
        return base_res
