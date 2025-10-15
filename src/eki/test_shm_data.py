#!/usr/bin/env python
"""
Test script to verify all EKI shared memory data.

This script displays all configuration and observation data
received from LDM-EKI via shared memory.

Usage:
    python test_shm_data.py
"""

from Model_Connection_np_Ensemble import print_all_eki_data

if __name__ == "__main__":
    print_all_eki_data()
