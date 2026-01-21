"""Timing utilities for performance logging."""

import time
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class Timer:
    """
    Timer class for tracking step durations in the pipeline.
    
    Usage:
        timer = Timer()
        with timer.measure("pose_estimation"):
            # do pose estimation
        
        print(timer.get_summary())
    """
    
    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id
        self.steps: Dict[str, float] = {}
        self._start_time: Optional[float] = None
        self._current_step: Optional[str] = None
    
    @contextmanager
    def measure(self, step_name: str):
        """Context manager to measure duration of a step."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.steps[step_name] = duration
            logger.debug(f"[{self.request_id}] {step_name}: {duration:.3f}s")
    
    def start(self, step_name: str) -> None:
        """Start timing a step manually."""
        self._current_step = step_name
        self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing the current step and return duration."""
        if self._start_time is None or self._current_step is None:
            raise RuntimeError("Timer not started")
        
        duration = time.perf_counter() - self._start_time
        self.steps[self._current_step] = duration
        
        logger.debug(f"[{self.request_id}] {self._current_step}: {duration:.3f}s")
        
        self._start_time = None
        self._current_step = None
        
        return duration
    
    def get_total(self) -> float:
        """Get total duration of all measured steps."""
        return sum(self.steps.values())
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all step durations."""
        summary = dict(self.steps)
        summary["total"] = self.get_total()
        return summary
    
    def log_summary(self, level: int = logging.INFO) -> None:
        """Log the timing summary."""
        summary = self.get_summary()
        parts = [f"{k}: {v:.3f}s" for k, v in summary.items()]
        msg = f"[{self.request_id}] Timing: {', '.join(parts)}"
        logger.log(level, msg)


def log_timing(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_timing
        def my_function():
            pass
        
        @log_timing(name="custom_name")
        def another_function():
            pass
    """
    def decorator(f: Callable) -> Callable:
        step_name = name or f.__name__
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                logger.info(f"{step_name}: {duration:.3f}s")
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator

