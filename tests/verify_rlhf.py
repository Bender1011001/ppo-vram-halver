#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E2E Verification script for benchmark_rlhf.py.
Runs the PPO memory benchmark and asserts that Single-Model Dual-Pass PPO
achieves at least a 35% VRAM savings compared to standard PPO.
"""

import sys
import subprocess
import re
from pathlib import Path


def parse_benchmark_result(result, label):
    """Return peak MiB, or None only for a confirmed CUDA OOM."""
    output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        if re.search(r"cuda.{0,30}out of memory", output, re.IGNORECASE):
            return None
        raise RuntimeError(f"{label} benchmark exited with code {result.returncode}:\n{output}")
    if "CUDA OUT OF MEMORY" in result.stdout:
        return None
    match = re.search(rf"{re.escape(label)} Peak:\s+([\d.]+)\s+MiB", result.stdout)
    if match is None:
        raise ValueError(f"Could not parse {label} peak VRAM from benchmark output")
    return float(match.group(1))


def main():
    project_root = Path(__file__).resolve().parent.parent
    benchmark_script = project_root / "benchmark_rlhf.py"
    
    print("======================================================================")
    print("RUNNING VRAM PPO RLHF BENCHMARK VALIDATION")
    print("======================================================================")
    
    # 1. Run Standard PPO Step in a separate process
    cmd_standard = [
        sys.executable,
        str(benchmark_script),
        "--model_name", "Qwen/Qwen2.5-3B-Instruct",
        "--step", "standard"
    ]
    print(f"Command: {' '.join(cmd_standard)}")
    res_standard = subprocess.run(cmd_standard, capture_output=True, text=True, encoding="utf-8")
    
    # Print output of standard step
    print("\n=== Standard PPO Output ===")
    print(res_standard.stdout)
    if res_standard.stderr:
        print("=== Standard PPO Error ===")
        print(res_standard.stderr)
        
    try:
        standard_vram = parse_benchmark_result(res_standard, "Standard PPO")
    except (RuntimeError, ValueError) as exc:
        print(f"[-] ERROR: {exc}")
        sys.exit(1)
    standard_oom = standard_vram is None
            
    # 2. Run Single-Model Dual-Pass PPO Step in a separate process
    cmd_dual = [
        sys.executable,
        str(benchmark_script),
        "--model_name", "Qwen/Qwen2.5-3B-Instruct",
        "--step", "dual_pass"
    ]
    print(f"\nCommand: {' '.join(cmd_dual)}")
    res_dual = subprocess.run(cmd_dual, capture_output=True, text=True, encoding="utf-8")
    
    print("\n=== Single-Model PPO Output ===")
    print(res_dual.stdout)
    if res_dual.stderr:
        print("=== Single-Model PPO Error ===")
        print(res_dual.stderr)
        
    try:
        dual_pass_vram = parse_benchmark_result(res_dual, "Single-Model PPO")
    except (RuntimeError, ValueError) as exc:
        print(f"[-] ERROR: {exc}")
        sys.exit(1)
    if dual_pass_vram is None:
        print("[-] ERROR: Single-Model PPO ran out of CUDA memory")
        sys.exit(2)
    
    # Compare results
    print(f"\n============================================================")
    print(f" COMPARISON SUMMARY")
    print(f"============================================================")
    if standard_oom:
        print("  Standard PPO Peak:        CUDA OUT OF MEMORY")
    else:
        print(f"  Standard PPO Peak:        {standard_vram:.2f} MiB")
    print(f"  Single-Model PPO Peak:    {dual_pass_vram:.2f} MiB")
    
    if not standard_oom:
        savings = standard_vram - dual_pass_vram
        pct_savings = (savings / standard_vram) * 100.0
        print(f"  VRAM Savings:             {savings:.2f} MiB ({pct_savings:.1f}% reduction)")
        
        # Assert savings
        if pct_savings >= 35.0:
            print(f"[PASS] VRAM savings ({pct_savings:.1f}%) meets target threshold (>= 35.0%)")
            print("======================================================================")
            print("[+] SUCCESS: VRAM validation passed!")
            sys.exit(0)
        else:
            print(f"[-] FAILURE: VRAM savings ({pct_savings:.1f}%) is below the target threshold (35.0%)")
            sys.exit(3)
    else:
        print("[PASS] Single-Model PPO successfully avoided OOM and fit in memory!")
        print("======================================================================")
        print("[+] SUCCESS: VRAM validation passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
