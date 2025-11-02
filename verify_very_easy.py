#!/usr/bin/env python3
"""
Verification script for very_easy difficulty mode.
This checks that the code changes are syntactically correct and the enums/mappings are properly defined.
"""

import ast
import sys

def check_chess_mcp():
    """Verify chess_mcp.py has correct very_easy configuration."""
    with open('chess_mcp.py', 'r') as f:
        content = f.read()

    # Parse the file to ensure it's syntactically correct
    try:
        tree = ast.parse(content)
        print("✓ chess_mcp.py is syntactically valid")
    except SyntaxError as e:
        print(f"✗ Syntax error in chess_mcp.py: {e}")
        return False

    # Check for VERY_EASY in Difficulty enum
    if 'VERY_EASY = "very_easy"' in content:
        print("✓ VERY_EASY added to Difficulty enum")
    else:
        print("✗ VERY_EASY not found in Difficulty enum")
        return False

    # Check for very_easy in DIFFICULTY_DEPTH_MAP
    if '"very_easy": 1' in content:
        print("✓ very_easy depth mapping set to 1")
    else:
        print("✗ very_easy depth mapping not found or incorrect")
        return False

    # Check for very_easy in AI_RATING_MAP
    if '"very_easy": 300' in content:
        print("✓ very_easy ELO rating set to 300")
    else:
        print("✗ very_easy ELO rating not found or incorrect")
        return False

    # Check for random import
    if 'import random' in content:
        print("✓ random module imported")
    else:
        print("✗ random module not imported")
        return False

    # Check for randomness logic in move selection
    if 'if difficulty_value == "very_easy" and random.random() < 0.4:' in content:
        print("✓ Randomness logic added for very_easy (40% random moves)")
    else:
        print("✗ Randomness logic not found for very_easy")
        return False

    # Check docstring mentions very_easy
    if 'very_easy' in content.lower() and '300' in content and 'complete beginner' in content.lower():
        print("✓ Documentation updated to mention very_easy mode")
    else:
        print("⚠ Documentation may not fully describe very_easy mode")

    print("\n" + "="*60)
    print("All checks passed! Very easy mode (300 ELO) has been added.")
    print("="*60)
    return True

if __name__ == "__main__":
    success = check_chess_mcp()
    sys.exit(0 if success else 1)
