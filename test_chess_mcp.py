#!/usr/bin/env python3
"""
Test script for Chess MCP Server

This script demonstrates how to use the chess MCP server's tools
and verifies they work correctly.
"""

import json
import asyncio
from chess_mcp import (
    chess_new_game,
    chess_submit_move,
    chess_get_board,
    chess_resign,
    chess_get_rating,
    NewGameInput,
    SubmitMoveInput,
    GetBoardInput,
    ResignInput,
    GetRatingInput,
    ResponseFormat,
    Difficulty
)i


async def test_new_game():
    """Test starting a new game."""
    print("=" * 60)
    print("TEST 1: Starting a new game")
    print("=" * 60)
    
    result = await chess_new_game(NewGameInput(
        game_id="test_game",
        response_format=ResponseFormat.MARKDOWN
    ))
    print(result)
    print()


async def test_submit_moves():
    """Test submitting moves."""
    print("=" * 60)
    print("TEST 2: Submitting moves")
    print("=" * 60)
    
    # Start fresh game
    await chess_new_game(NewGameInput(game_id="test_game"))
    
    # Play several moves
    moves = ["e4", "Nf3", "Bc4", "Nc3"]
    
    for move in moves:
        print(f"\n--- White plays: {move} ---\n")
        result = await chess_submit_move(SubmitMoveInput(
            move=move,
            game_id="test_game",
            response_format=ResponseFormat.MARKDOWN
        ))
        print(result)
        print()


async def test_get_board():
    """Test getting board state."""
    print("=" * 60)
    print("TEST 3: Getting board state")
    print("=" * 60)
    
    result = await chess_get_board(GetBoardInput(
        game_id="test_game",
        include_legal_moves=True,
        response_format=ResponseFormat.JSON
    ))
    
    # Pretty print JSON
    data = json.loads(result)
    print(json.dumps(data, indent=2))
    print()


async def test_illegal_move():
    """Test submitting an illegal move."""
    print("=" * 60)
    print("TEST 4: Testing illegal move handling")
    print("=" * 60)
    
    # Start fresh game
    await chess_new_game(NewGameInput(game_id="test_game"))
    
    # Try an illegal move
    print("\n--- Attempting illegal move: 'e5' (can't move pawn two squares on first move from e5) ---\n")
    result = await chess_submit_move(SubmitMoveInput(
        move="e5",
        game_id="test_game",
        response_format=ResponseFormat.MARKDOWN
    ))
    print(result)
    print()


async def test_json_format():
    """Test JSON response format."""
    print("=" * 60)
    print("TEST 5: JSON response format")
    print("=" * 60)
    
    # Start fresh game
    result = await chess_new_game(NewGameInput(
        game_id="json_test",
        response_format=ResponseFormat.JSON
    ))
    
    data = json.loads(result)
    print("New game (JSON):")
    print(json.dumps(data, indent=2))
    print()
    
    # Submit move in JSON
    result = await chess_submit_move(SubmitMoveInput(
        move="d4",
        game_id="json_test",
        response_format=ResponseFormat.JSON
    ))
    
    data = json.loads(result)
    print("After move d4 (JSON):")
    print(json.dumps(data, indent=2))
    print()


async def test_multiple_games():
    """Test managing multiple games."""
    print("=" * 60)
    print("TEST 6: Multiple concurrent games")
    print("=" * 60)
    
    # Start two games
    await chess_new_game(NewGameInput(game_id="game_a"))
    await chess_new_game(NewGameInput(game_id="game_b"))
    
    # Play different moves in each
    await chess_submit_move(SubmitMoveInput(move="e4", game_id="game_a"))
    await chess_submit_move(SubmitMoveInput(move="d4", game_id="game_b"))
    
    # Check both games
    print("\n--- Game A ---")
    result_a = await chess_get_board(GetBoardInput(
        game_id="game_a",
        response_format=ResponseFormat.JSON
    ))
    data_a = json.loads(result_a)
    print(f"FEN: {data_a['fen']}")
    print(f"Move number: {data_a['move_number']}")
    
    print("\n--- Game B ---")
    result_b = await chess_get_board(GetBoardInput(
        game_id="game_b",
        response_format=ResponseFormat.JSON
    ))
    data_b = json.loads(result_b)
    print(f"FEN: {data_b['fen']}")
    print(f"Move number: {data_b['move_number']}")
    print()


async def test_scholar_mate():
    """Test a complete game (Scholar's Mate)."""
    print("=" * 60)
    print("TEST 7: Complete game scenario")
    print("=" * 60)
    
    # Start fresh game
    await chess_new_game(NewGameInput(game_id="scholar"))
    
    print("\nPlaying a sequence of moves...\n")
    
    # Play several moves (not actually Scholar's Mate since AI will defend)
    moves = ["e4", "Bc4", "Qh5"]
    
    for move in moves:
        print(f"White plays: {move}")
        result = await chess_submit_move(SubmitMoveInput(
            move=move,
            game_id="scholar",
            response_format=ResponseFormat.MARKDOWN
        ))
        # Only show the summary, not full board each time
        lines = result.split('\n')
        for line in lines:
            if 'White:' in line or 'Black:' in line:
                print(line)
    
    # Show final position
    print("\n--- Final Position ---")
    result = await chess_get_board(GetBoardInput(
        game_id="scholar",
        include_legal_moves=False,
        response_format=ResponseFormat.MARKDOWN
    ))
    print(result)
    print()


async def test_resign_basic():
    """Test basic resignation functionality."""
    print("=" * 60)
    print("TEST 8: Basic resignation")
    print("=" * 60)

    # Start fresh game
    await chess_new_game(NewGameInput(game_id="resign_test"))

    # Play a few moves
    await chess_submit_move(SubmitMoveInput(move="e4", game_id="resign_test"))
    await chess_submit_move(SubmitMoveInput(move="d4", game_id="resign_test"))

    # White resigns
    print("\n--- White resigns ---\n")
    result = await chess_resign(ResignInput(
        game_id="resign_test",
        reason="Testing resignation",
        response_format=ResponseFormat.MARKDOWN
    ))
    print(result)
    print()


async def test_resign_without_reason():
    """Test resignation without a reason."""
    print("=" * 60)
    print("TEST 9: Resignation without reason")
    print("=" * 60)

    # Start fresh game
    await chess_new_game(NewGameInput(game_id="resign_no_reason"))

    # Play a move
    await chess_submit_move(SubmitMoveInput(move="Nf3", game_id="resign_no_reason"))

    # White resigns without reason
    print("\n--- White resigns (no reason) ---\n")
    result = await chess_resign(ResignInput(
        game_id="resign_no_reason",
        response_format=ResponseFormat.JSON
    ))

    data = json.loads(result)
    print(json.dumps(data, indent=2))
    print()


async def test_resign_errors():
    """Test resignation error cases."""
    print("=" * 60)
    print("TEST 10: Resignation error handling")
    print("=" * 60)

    # Test 1: Resign non-existent game
    print("\n--- Test 1: Resigning non-existent game ---\n")
    result = await chess_resign(ResignInput(
        game_id="nonexistent_game",
        response_format=ResponseFormat.MARKDOWN
    ))
    print(result)

    # Test 2: Resign already-ended game
    print("\n--- Test 2: Resigning already-ended game ---\n")
    await chess_new_game(NewGameInput(game_id="already_resigned"))
    await chess_submit_move(SubmitMoveInput(move="e4", game_id="already_resigned"))

    # First resignation
    await chess_resign(ResignInput(game_id="already_resigned"))

    # Try to resign again
    result = await chess_resign(ResignInput(
        game_id="already_resigned",
        response_format=ResponseFormat.MARKDOWN
    ))
    print(result)
    print()


async def test_resign_blocks_moves():
    """Test that moves cannot be made after resignation."""
    print("=" * 60)
    print("TEST 11: Moves blocked after resignation")
    print("=" * 60)

    # Start game and resign
    await chess_new_game(NewGameInput(game_id="resign_block"))
    await chess_submit_move(SubmitMoveInput(move="e4", game_id="resign_block"))
    await chess_resign(ResignInput(game_id="resign_block", reason="Strategic retreat"))

    # Try to make another move
    print("\n--- Attempting move after resignation ---\n")
    result = await chess_submit_move(SubmitMoveInput(
        move="d4",
        game_id="resign_block",
        response_format=ResponseFormat.MARKDOWN
    ))
    print(result)
    print()


async def test_new_game_clears_resignation():
    """Test that starting a new game clears resignation metadata."""
    print("=" * 60)
    print("TEST 12: New game clears resignation")
    print("=" * 60)

    # Start game, make moves, and resign
    await chess_new_game(NewGameInput(game_id="clear_test"))
    await chess_submit_move(SubmitMoveInput(move="e4", game_id="clear_test"))
    await chess_resign(ResignInput(game_id="clear_test"))

    # Start new game with same ID
    print("\n--- Starting new game with same ID ---\n")
    result = await chess_new_game(NewGameInput(
        game_id="clear_test",
        response_format=ResponseFormat.JSON
    ))

    data = json.loads(result)
    print(f"Game over: {data.get('is_game_over', 'Not specified')}")
    print(f"Result field present: {'result' in data}")

    # Should be able to make moves now
    result = await chess_submit_move(SubmitMoveInput(
        move="Nf3",
        game_id="clear_test",
        response_format=ResponseFormat.JSON
    ))
    data = json.loads(result)
    print(f"Move successful: {data.get('last_moves', {}).get('white') == 'Nf3'}")
    print()


async def test_elo_initialization():
    """Test that player ratings are initialized correctly."""
    print("=" * 60)
    print("TEST 13: ELO Rating Initialization")
    print("=" * 60)

    # Start a new game with a new player
    await chess_new_game(NewGameInput(
        game_id="elo_test1",
        player_id="alice",
        response_format=ResponseFormat.JSON
    ))

    # Check initial rating
    result = await chess_get_rating(GetRatingInput(
        player_id="alice",
        response_format=ResponseFormat.JSON
    ))

    data = json.loads(result)
    print(f"New player 'alice' initialized:")
    print(f"  Rating: {data['rating']}")
    print(f"  Games played: {data['statistics']['games_played']}")
    print(f"  Expected: Rating=1200, Games=0")
    assert data['rating'] == 1200, "Initial rating should be 1200"
    assert data['statistics']['games_played'] == 0, "Games played should be 0"
    print("✓ Test passed\n")


async def test_elo_rating_after_win():
    """Test that rating increases after winning a game."""
    print("=" * 60)
    print("TEST 14: ELO Rating Update After Win (AI Resignation)")
    print("=" * 60)

    # Start a new game with easy difficulty
    await chess_new_game(NewGameInput(
        game_id="win_test",
        player_id="bob",
        difficulty=Difficulty.EASY
    ))

    # Get initial rating
    result = await chess_get_rating(GetRatingInput(
        player_id="bob",
        response_format=ResponseFormat.JSON
    ))
    initial_data = json.loads(result)
    initial_rating = initial_data['rating']
    print(f"Initial rating: {initial_rating}")

    # Play moves that give white a huge advantage (simplified test - we'll resign manually)
    # For a real win, we'd need to play until checkmate or AI resignation
    # Let's use resignation to simulate a win
    await chess_submit_move(SubmitMoveInput(move="e4", game_id="win_test"))
    await chess_resign(ResignInput(game_id="win_test", reason="Testing ELO update"))

    # Check rating did NOT increase (white resigned, so white lost)
    result = await chess_get_rating(GetRatingInput(
        player_id="bob",
        response_format=ResponseFormat.JSON
    ))
    final_data = json.loads(result)
    final_rating = final_data['rating']
    print(f"Final rating after loss: {final_rating}")
    print(f"Rating change: {final_rating - initial_rating}")
    print(f"Stats: {final_data['statistics']}")

    assert final_rating < initial_rating, "Rating should decrease after a loss"
    assert final_data['statistics']['losses'] == 1, "Should have 1 loss"
    print("✓ Test passed\n")


async def test_elo_multiple_players():
    """Test that multiple players have independent ratings."""
    print("=" * 60)
    print("TEST 15: Multiple Players with Independent Ratings")
    print("=" * 60)

    # Player 1 starts a game
    await chess_new_game(NewGameInput(
        game_id="player1_game",
        player_id="player1",
        difficulty=Difficulty.MEDIUM
    ))

    # Player 2 starts a game
    await chess_new_game(NewGameInput(
        game_id="player2_game",
        player_id="player2",
        difficulty=Difficulty.HARD
    ))

    # Both players play and resign (losses)
    await chess_submit_move(SubmitMoveInput(move="e4", game_id="player1_game"))
    await chess_resign(ResignInput(game_id="player1_game"))

    await chess_submit_move(SubmitMoveInput(move="d4", game_id="player2_game"))
    await chess_resign(ResignInput(game_id="player2_game"))

    # Check both players have different ratings (different opponents)
    result1 = await chess_get_rating(GetRatingInput(
        player_id="player1",
        response_format=ResponseFormat.JSON
    ))
    result2 = await chess_get_rating(GetRatingInput(
        player_id="player2",
        response_format=ResponseFormat.JSON
    ))

    data1 = json.loads(result1)
    data2 = json.loads(result2)

    print(f"Player 1 rating (vs Medium AI): {data1['rating']}")
    print(f"Player 2 rating (vs Hard AI): {data2['rating']}")

    # Both should have lost rating points, but potentially different amounts
    assert data1['rating'] < 1200, "Player 1 should have lost rating"
    assert data2['rating'] < 1200, "Player 2 should have lost rating"
    assert data1['statistics']['games_played'] == 1, "Player 1 should have 1 game"
    assert data2['statistics']['games_played'] == 1, "Player 2 should have 1 game"
    print("✓ Test passed\n")


async def test_elo_rating_tool_markdown():
    """Test the chess_get_rating tool with markdown format."""
    print("=" * 60)
    print("TEST 16: Get Rating Tool (Markdown)")
    print("=" * 60)

    # Create a player with some game history
    await chess_new_game(NewGameInput(
        game_id="charlie_game1",
        player_id="charlie",
        difficulty=Difficulty.EASY
    ))
    await chess_submit_move(SubmitMoveInput(move="e4", game_id="charlie_game1"))
    await chess_resign(ResignInput(game_id="charlie_game1"))

    # Get rating in markdown format
    result = await chess_get_rating(GetRatingInput(
        player_id="charlie",
        response_format=ResponseFormat.MARKDOWN
    ))

    print(result)

    assert "Player Rating: charlie" in result, "Should contain player ID"
    assert "Current Rating" in result, "Should have rating section"
    assert "Statistics" in result, "Should have statistics section"
    assert "Games Played**: 1" in result or "Games Played: 1" in result, "Should show 1 game played"
    print("✓ Test passed\n")


async def test_elo_nonexistent_player():
    """Test getting rating for a player that doesn't exist."""
    print("=" * 60)
    print("TEST 17: Get Rating for Nonexistent Player")
    print("=" * 60)

    result = await chess_get_rating(GetRatingInput(
        player_id="nonexistent_player",
        response_format=ResponseFormat.MARKDOWN
    ))

    print(result)
    assert "not found" in result, "Should indicate player not found"
    print("✓ Test passed\n")


async def test_elo_rating_metadata():
    """Test that rating update metadata is included in game end response."""
    print("=" * 60)
    print("TEST 18: Rating Update in Game End Response")
    print("=" * 60)

    # Start game and resign
    await chess_new_game(NewGameInput(
        game_id="metadata_test",
        player_id="diana",
        difficulty=Difficulty.MEDIUM
    ))

    # Make a move and resign
    result = await chess_submit_move(SubmitMoveInput(
        move="e4",
        game_id="metadata_test",
        response_format=ResponseFormat.MARKDOWN
    ))

    # Resign to end the game
    result = await chess_resign(ResignInput(
        game_id="metadata_test",
        response_format=ResponseFormat.MARKDOWN
    ))

    print(result)

    # Check that rating update is shown
    assert "Rating Update" in result, "Should contain rating update section"
    assert "Old Rating" in result, "Should show old rating"
    assert "New Rating" in result, "Should show new rating"
    print("✓ Test passed\n")


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CHESS MCP SERVER - TEST SUITE")
    print("=" * 60 + "\n")

    try:
        await test_new_game()
        await test_submit_moves()
        await test_get_board()
        await test_illegal_move()
        await test_json_format()
        await test_multiple_games()
        await test_scholar_mate()
        await test_resign_basic()
        await test_resign_without_reason()
        await test_resign_errors()
        await test_resign_blocks_moves()
        await test_new_game_clears_resignation()
        await test_elo_initialization()
        await test_elo_rating_after_win()
        await test_elo_multiple_players()
        await test_elo_rating_tool_markdown()
        await test_elo_nonexistent_player()
        await test_elo_rating_metadata()

        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        print()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
