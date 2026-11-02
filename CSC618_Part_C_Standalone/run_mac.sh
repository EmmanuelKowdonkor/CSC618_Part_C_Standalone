#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 part_c.py
echo "SUCCESS: Check results/ for outputs."
