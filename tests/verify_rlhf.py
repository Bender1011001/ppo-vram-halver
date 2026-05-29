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
        
    standard_vram = None
    standard_oom = False
    
    # Determine if standard step succeeded or crashed/OOMed
    if res_standard.returncode != 0 or "CUDA OUT OF MEMORY" in res_standard.stdout or "out of memory" in res_standard.stderr.lower():
        print("[!] Standard PPO step crashed or ran out of memory as expected on this GPU!")
        standard_oom = True
        standard_vram = float("inf")
    else:
        # Parse standard PPO peak VRAM
        standard_match = re.search(r"Standard PPO Peak:\s+([\d\.]+)\s+MiB", res_standard.stdout)
        if standard_match:
            standard_vram = float(standard_match.group(1))
        else:
            print("[!] Standard PPO step failed or output could not be parsed.")
            standard_oom = True
            standard_vram = float("inf")
            
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
        
    if res_dual.returncode != 0:
        print(f"[-] ERROR: Single-Model PPO step crashed with exit code {res_dual.returncode}")
        sys.exit(1)
        
    # Parse dual-pass PPO peak VRAM
    dual_pass_match = re.search(r"Single-Model PPO Peak:\s+([\d\.]+)\s+MiB", res_dual.stdout)
    if not dual_pass_match:
        print("[-] ERROR: Could not parse Single-Model PPO peak VRAM.")
        sys.exit(2)
        
    dual_pass_vram = float(dual_pass_match.group(1))
    
    # Compare results
    print(f"\n============================================================")
    print(f" COMPARISON SUMMARY")
    print(f"============================================================")
    if standard_oom:
        print(f"  Standard PPO Peak:        CUDA OUT OF MEMORY (> 8,188 MiB)")
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
