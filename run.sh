#!/bin/bash
# Simple entrypoint to run the tokenomics tools
set -e

python3 - <<'PY'
from tokenomics_simulator import run_simulation
from risk_analysis import get_llm_analysis
from report_generator import generate_report

simulation_df = run_simulation()
try:
    analysis = get_llm_analysis(simulation_df)
except Exception as e:
    analysis = f"LLM analysis failed: {e}"

print("=== Simulation Output ===")
print(simulation_df)
print("\n=== Analysis ===")
print(analysis)

generate_report(simulation_df, analysis)
print("\nReport written to report.txt")
PY
EOF
