"""
Provides tools helpinmg with memory management and other topics related to rendering and loading the dataset.
"""

from typing import Dict, List
import sys
import psutil
import time
import gc
import logging


log = logging.getLogger(__name__)


def _generate_uci_map() -> Dict[str, int]:
    """Generate a mapping from UCI strings to unique integers."""
    uci_map = {}
    counter = 0
    
    # Generate all possible UCI moves
    files = 'abcdefgh'
    ranks = '12345678'
    pieces = 'qrbn'  # promotion pieces
    
    # Normal moves (4 characters)
    for from_file in files:
        for from_rank in ranks:
            for to_file in files:
                for to_rank in ranks:
                    uci = f"{from_file}{from_rank}{to_file}{to_rank}"
                    uci_map[uci] = counter
                    counter += 1
    
    # Promotion moves (5 characters)
    for from_file in files:
        for to_file in files:
            for piece in pieces:
                # White promotions (7th to 8th rank)
                uci = f"{from_file}7{to_file}8{piece}"
                uci_map[uci] = counter
                counter += 1
                # Black promotions (2nd to 1st rank)
                uci = f"{from_file}2{to_file}1{piece}"
                uci_map[uci] = counter
                counter += 1
    
    return uci_map


def _generate_reverse_uci_map() -> Dict[int, str]:
    """Generate reverse mapping from integers to UCI strings."""
    uci_map = _generate_uci_map()
    return {v: k for k, v in uci_map.items()}


# Global UCI mappings
UCI_TO_INT = _generate_uci_map()
INT_TO_UCI = _generate_reverse_uci_map()


def convert_uci_strings_to_ints(moves_uci: List[str]) -> List[int]:
    """Convert list of UCI strings to integers."""
    return [UCI_TO_INT.get(uci, -1) for uci in moves_uci]


def convert_uci_ints_to_strings(moves_ints: List[int]) -> List[str]:
    """Convert list of UCI integers to strings."""
    return [INT_TO_UCI.get(uci_int, '') for uci_int in moves_ints] 


def safe_delete_and_wait(*objects, safety_margin_mb=1_000_000, timeout=30):
    """
    Delete objects and wait until memory is actually freed.
    
    Args:
        *objects: Objects to delete
        safety_margin_mb: Indicates how much of the memory yet unfreed is acceptable (also may help to mitigate incorrect readings).
        timeout: Max time to wait in seconds
    
    Returns:
        bool: True if memory was freed, False if timeout
    """
    # Calculate total size of objects
    total_size_bytes = sum(sys.getsizeof(obj) for obj in objects)
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # Delete objects
    for i, obj in enumerate(objects):
        # Delete from caller's namespace if possible
        del obj
    
    # Calculate target memory
    target_memory_mb = initial_memory_mb - total_size_mb + safety_margin_mb
    
    # Wait for memory to be freed
    start_time = time.time()
    while time.time() - start_time < timeout:
        gc.collect()
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        if current_memory_mb <= target_memory_mb:
            freed_mb = initial_memory_mb - current_memory_mb
            log.info(f"Memory freed: {freed_mb:.1f} MB (expected: {total_size_mb:.1f} MB)")
            return True
        
        time.sleep(1.0)
    
    log.warn(f"Timeout: Only freed {initial_memory_mb - current_memory_mb:.1f} MB of expected {total_size_mb:.1f} MB")
    return False