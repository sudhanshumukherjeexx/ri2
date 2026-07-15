import time
import gc
import psutil
import streamlit as st
from memory_profiler import memory_usage


def monitor(func):
    def wrapper(*args, **kwargs):
        #gc.collect()  # Ensure garbage collection before measurement
        mem_before = memory_usage(timeout=1, interval=0.1)[0]
        start_time = time.time()

        # Execute the function
        result = func(*args, **kwargs)

        mem_after = memory_usage(timeout=1, interval=0.1)[0]
        end_time = time.time()

        # Calculate differences
        mem_diff = mem_after - mem_before
        if abs(mem_diff) < 0.1:  # Threshold to ignore insignificant changes
            mem_diff = 0

        # Print results
        st.info(f"""
                1. Execution Time: {end_time - start_time:.4f} seconds \n
                2. Memory Usage: {max(0, mem_diff):.4f} MiB \n
                3. CPU Usage: {psutil.cpu_percent(interval=1):.2f}%  
                """)
        return result
    return wrapper


