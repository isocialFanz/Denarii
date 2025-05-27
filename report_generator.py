from pathlib import Path
import pandas as pd


def generate_report(
    simulation_df: pd.DataFrame, analysis: str, output_path: str = "report.txt"
) -> None:
    """Write the simulation results and analysis to a simple report file."""
    report_lines = [
        "Tokenomics Simulation Report",
        "===========================",
        "",
        "Simulation Data:",
        simulation_df.to_string(index=False),
        "",
        "Analysis:",
        analysis,
        "",
    ]
    Path(output_path).write_text("\n".join(report_lines))


if __name__ == "__main__":
    sample_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    generate_report(sample_df, "Sample analysis")
    print("Report written to report.txt")
