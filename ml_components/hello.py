#!/usr/bin/env python3
"""Simple test script to verify ML environment."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


def main():
    print("TraderJoe ML Components - Environment Check")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"XGBoost version: {xgb.__version__}")
    print(f"LightGBM version: {lgb.__version__}")
    print("\nEnvironment ready for Phase 2!")


if __name__ == "__main__":
    main()
