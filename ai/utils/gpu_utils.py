"""
GPU Utilities
=============

Helper functions for GPU memory management and device detection.
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def clear_gpu_memory():
    """
    Clear GPU memory cache to free up unused memory.
    
    Useful when encountering OutOfMemoryError during training/inference.
    Call this between epochs or before large operations.
    """
    if GPU_AVAILABLE:
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            print("🧹 GPU memory cache cleared")
            return True
        except Exception as e:
            print(f"⚠️  Failed to clear GPU memory: {e}")
            return False
    else:
        print("ℹ️  No GPU available (using CPU)")
        return False

def get_gpu_memory_info():
    """
    Get current GPU memory usage information.
    
    Returns:
        dict: Memory info with 'used', 'total', 'free' in bytes
              Returns None if GPU not available
    """
    if GPU_AVAILABLE:
        try:
            mempool = cp.get_default_memory_pool()
            used = mempool.used_bytes()
            total = mempool.total_bytes()
            
            return {
                'used_mb': used / (1024**2),
                'total_mb': total / (1024**2),
                'used_gb': used / (1024**3),
                'total_gb': total / (1024**3)
            }
        except Exception as e:
            print(f"⚠️  Failed to get GPU memory info: {e}")
            return None
    else:
        return None

def print_gpu_memory():
    """Print current GPU memory usage in human-readable format."""
    info = get_gpu_memory_info()
    if info:
        print(f"💾 GPU Memory: {info['used_mb']:.1f} MB / {info['total_mb']:.1f} MB used ({info['used_gb']:.2f} GB / {info['total_gb']:.2f} GB)")
    else:
        print("ℹ️  GPU memory info not available")

def check_gpu_available():
    """
    Check if GPU (CUDA) is available.
    
    Returns:
        bool: True if GPU available, False otherwise
    """
    return GPU_AVAILABLE
