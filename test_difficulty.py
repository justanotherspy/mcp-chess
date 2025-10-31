#!/usr/bin/env python3
"""Test script for difficulty selector feature"""

import asyncio
import json
import time
from chess_mcp import (
    chess_new_game, chess_submit_move,
    NewGameInput, SubmitMoveInput,
    Difficulty, ResponseFormat
)


async def test_difficulty_levels():
    """Test all three difficulty levels"""
    print("=" * 60)
    print("DIFFICULTY SELECTOR FEATURE TESTS")
    print("=" * 60)

    # Test 1: Easy difficulty
    print("\n--- Test 1: Easy Difficulty ---")
    easy_input = NewGameInput(
        game_id="easy_test",
        difficulty=Difficulty.EASY,
        response_format=ResponseFormat.JSON
    )
    result = await chess_new_game(easy_input)
    data = json.loads(result)
    assert data["ai_difficulty"] == "easy", "Easy difficulty not set"
    assert data["ai_search_depth"] == 2, "Easy depth should be 2"
    print(f"✓ Easy game created with depth {data['ai_search_depth']}")

    # Make a move and time it
    move_input = SubmitMoveInput(
        game_id="easy_test",
        move="e4",
        response_format=ResponseFormat.JSON
    )
    start = time.time()
    result = await chess_submit_move(move_input)
    easy_time = time.time() - start
    data = json.loads(result)
    assert data["ai_difficulty"] == "easy", "Difficulty not persisted"
    print(f"✓ AI responded in {easy_time:.3f}s")

    # Test 2: Medium difficulty (default)
    print("\n--- Test 2: Medium Difficulty (Default) ---")
    medium_input = NewGameInput(
        game_id="medium_test",
        response_format=ResponseFormat.JSON
    )
    result = await chess_new_game(medium_input)
    data = json.loads(result)
    assert data["ai_difficulty"] == "medium", "Default should be medium"
    assert data["ai_search_depth"] == 3, "Medium depth should be 3"
    print(f"✓ Medium game created (default) with depth {data['ai_search_depth']}")

    # Test 3: Hard difficulty
    print("\n--- Test 3: Hard Difficulty ---")
    hard_input = NewGameInput(
        game_id="hard_test",
        difficulty=Difficulty.HARD,
        response_format=ResponseFormat.JSON
    )
    result = await chess_new_game(hard_input)
    data = json.loads(result)
    assert data["ai_difficulty"] == "hard", "Hard difficulty not set"
    assert data["ai_search_depth"] == 4, "Hard depth should be 4"
    print(f"✓ Hard game created with depth {data['ai_search_depth']}")

    # Make a move and time it
    move_input = SubmitMoveInput(
        game_id="hard_test",
        move="e4",
        response_format=ResponseFormat.JSON
    )
    start = time.time()
    result = await chess_submit_move(move_input)
    hard_time = time.time() - start
    data = json.loads(result)
    assert data["ai_difficulty"] == "hard", "Difficulty not persisted"
    print(f"✓ AI responded in {hard_time:.3f}s")

    # Test 4: Markdown format shows difficulty
    print("\n--- Test 4: Markdown Format Shows Difficulty ---")
    md_input = NewGameInput(
        game_id="markdown_test",
        difficulty=Difficulty.HARD,
        response_format=ResponseFormat.MARKDOWN
    )
    result = await chess_new_game(md_input)
    # Check for bold markdown format
    assert "**AI Difficulty**: Hard (search depth: 4)" in result, "Difficulty not shown in markdown"
    print("✓ Difficulty displayed in markdown format")

    # Test 5: Concurrent games with different difficulties
    print("\n--- Test 5: Concurrent Games with Different Difficulties ---")
    easy_game = NewGameInput(game_id="concurrent_easy", difficulty=Difficulty.EASY, response_format=ResponseFormat.JSON)
    hard_game = NewGameInput(game_id="concurrent_hard", difficulty=Difficulty.HARD, response_format=ResponseFormat.JSON)

    await chess_new_game(easy_game)
    await chess_new_game(hard_game)

    # Make same move in both games
    easy_move = SubmitMoveInput(game_id="concurrent_easy", move="e4", response_format=ResponseFormat.JSON)
    hard_move = SubmitMoveInput(game_id="concurrent_hard", move="e4", response_format=ResponseFormat.JSON)

    easy_result = await chess_submit_move(easy_move)
    hard_result = await chess_submit_move(hard_move)

    easy_data = json.loads(easy_result)
    hard_data = json.loads(hard_result)

    assert easy_data["ai_difficulty"] == "easy", "Easy game changed difficulty"
    assert hard_data["ai_difficulty"] == "hard", "Hard game changed difficulty"
    print("✓ Each game maintains its own difficulty independently")

    # Performance comparison
    print("\n--- Performance Comparison ---")
    print(f"Easy (depth 2):  {easy_time:.3f}s")
    print(f"Hard (depth 4):  {hard_time:.3f}s")
    print(f"Speed ratio:     {hard_time/easy_time:.1f}x slower")

    print("\n" + "=" * 60)
    print("ALL DIFFICULTY TESTS PASSED! ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_difficulty_levels())
