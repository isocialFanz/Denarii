import pandas as pd
import numpy as np

def run_simulation(num_periods: int = 12) -> pd.DataFrame:
    """Return a placeholder tokenomics simulation as a DataFrame."""
    periods = np.arange(1, num_periods + 1)
    supply = 1_000 + np.cumsum(np.random.randint(0, 10, size=num_periods))
    demand = 1_000 + np.cumsum(np.random.randint(0, 10, size=num_periods))

    df = pd.DataFrame({
        "period": periods,
        "supply": supply,
        "demand": demand,
    })
    return df


if __name__ == "__main__":
    df = run_simulation()
    print(df)
