# Chess MCP Server - Difficulty Selector Implementation Plan

## Overview

This document provides detailed implementation steps for adding a difficulty selector feature to the Chess MCP Server. The agent will be able to choose between **Easy**, **Medium**, and **Hard** difficulty levels when starting a new game, controlling the AI opponent's strength.

---

## Feature Summary

**Objective**: Allow the agent to specify AI difficulty when creating a new chess game.

**User-Facing Change**: Add a `difficulty` parameter to the `chess_new_game` tool with options:
- `easy` - AI searches 2 moves ahead (faster, weaker play)
- `medium` - AI searches 3 moves ahead (balanced, current default)
- `hard` - AI searches 4 moves ahead (slower, stronger play)

---

## Current State Analysis

### Relevant Code Locations

**Line 27**: `DEFAULT_AI_DEPTH = 3` - Global constant for AI search depth

**Lines 38-56**: `NewGameInput` Pydantic model - Currently accepts `game_id` and `response_format`

**Lines 446-483**: `chess_new_game()` function - Creates new game and stores board state

**Lines 22-23**: State storage dictionaries:
```python
game_state: Dict[str, chess.Board] = {}
game_metadata: Dict[str, Dict[str, Any]] = {}
```

**Line 599**: AI move calculation in `chess_submit_move()`:
```python
eval_score, black_move = minimax(board, DEFAULT_AI_DEPTH, float('-inf'), float('inf'), True)
```

**Lines 319-331**: `get_best_move()` function - Already accepts `depth` parameter but isn't used dynamically

### Key Insight
The architecture already supports variable depth via `get_best_move(board, depth)`, but the depth is hardcoded to `DEFAULT_AI_DEPTH` in the `chess_submit_move()` function. We need to:
1. Store the difficulty level per game
2. Map difficulty to depth
3. Use the stored difficulty when calculating AI moves

---

## Implementation Steps

### Step 1: Create Difficulty Enum

**File**: `chess_mcp.py`
**Location**: After line 33 (after `ResponseFormat` enum)

**Add new enum:**
```python
class Difficulty(str, Enum):
    """AI difficulty levels for chess opponent."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
```

**Rationale**: Using an enum ensures type safety and provides clear documentation of valid values.

---

### Step 2: Add Difficulty Mapping Constant

**File**: `chess_mcp.py`
**Location**: After line 27 (after `DEFAULT_AI_DEPTH`)

**Add mapping dictionary:**
```python
# Difficulty to search depth mapping
DIFFICULTY_DEPTH_MAP = {
    Difficulty.EASY: 2,      # Searches 2 moves ahead - faster, weaker
    Difficulty.MEDIUM: 3,    # Searches 3 moves ahead - balanced (default)
    Difficulty.HARD: 4,      # Searches 4 moves ahead - slower, stronger
}
```

**Rationale**:
- **Easy (depth 2)**: ~20x faster than hard, makes tactical mistakes
- **Medium (depth 3)**: Current default, good balance of speed/strength
- **Hard (depth 4)**: ~20x slower than easy, plays strong positional chess
- Depth 5+ would be too slow for real-time play without additional optimizations

---

### Step 3: Update NewGameInput Model

**File**: `chess_mcp.py`
**Location**: Lines 38-56 (`NewGameInput` class)

**Current code:**
```python
class NewGameInput(BaseModel):
    """Input model for starting a new chess game."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    game_id: Optional[str] = Field(
        default="default",
        description="Unique identifier for the game (e.g., 'game1', 'match-20240115')",
        min_length=1,
        max_length=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )
```

**Add new field after `game_id`:**
```python
    difficulty: Difficulty = Field(
        default=Difficulty.MEDIUM,
        description="AI opponent difficulty: 'easy' (depth 2), 'medium' (depth 3), or 'hard' (depth 4)"
    )
```

**Full updated class:**
```python
class NewGameInput(BaseModel):
    """Input model for starting a new chess game."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    game_id: Optional[str] = Field(
        default="default",
        description="Unique identifier for the game (e.g., 'game1', 'match-20240115')",
        min_length=1,
        max_length=100
    )
    difficulty: Difficulty = Field(
        default=Difficulty.MEDIUM,
        description="AI opponent difficulty: 'easy' (depth 2), 'medium' (depth 3), or 'hard' (depth 4)"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )
```

**Rationale**: Defaults to medium difficulty to maintain backward compatibility with existing clients.

---

### Step 4: Store Difficulty in Game Metadata

**File**: `chess_mcp.py`
**Location**: Lines 446-483 (`chess_new_game()` function)

**Current code (lines 462-469):**
```python
    try:
        # Create new game
        board = chess.Board()
        game_state[params.game_id] = board

        # Clear any existing resignation metadata
        if params.game_id in game_metadata:
            del game_metadata[params.game_id]
```

**Replace with:**
```python
    try:
        # Create new game
        board = chess.Board()
        game_state[params.game_id] = board

        # Store game metadata (difficulty and clear resignation state)
        game_metadata[params.game_id] = {
            "difficulty": params.difficulty,
            "resigned": False
        }
```

**Rationale**: Stores difficulty level per game, allowing multiple concurrent games with different difficulties.

---

### Step 5: Update AI Move Calculation

**File**: `chess_mcp.py`
**Location**: Line 599 in `chess_submit_move()` function

**Current code:**
```python
        # AI evaluates position and decides whether to resign or move
        eval_score, black_move = minimax(board, DEFAULT_AI_DEPTH, float('-inf'), float('inf'), True)
```

**Replace with:**
```python
        # AI evaluates position and decides whether to resign or move
        # Get difficulty level for this game (default to MEDIUM if not set)
        game_difficulty = game_metadata.get(params.game_id, {}).get("difficulty", Difficulty.MEDIUM)
        ai_depth = DIFFICULTY_DEPTH_MAP[game_difficulty]
        eval_score, black_move = minimax(board, ai_depth, float('-inf'), float('inf'), True)
```

**Rationale**:
- Retrieves stored difficulty from game metadata
- Maps difficulty to appropriate search depth
- Uses depth in minimax calculation
- Gracefully defaults to MEDIUM if metadata missing (backward compatibility)

---

### Step 6: Update Response Formatting to Show Difficulty

**File**: `chess_mcp.py`
**Location**: Two functions need updates

#### 6.1: Update `format_board_markdown()`

**Location**: Lines 336-387 (function `format_board_markdown`)

**Find line 349** (after move number):
```python
    result += f"- **Move Number**: {board.fullmove_number}\n"
```

**Add after this line:**
```python
    # Show AI difficulty if metadata exists
    if game_id in game_metadata and "difficulty" in game_metadata[game_id]:
        difficulty = game_metadata[game_id]["difficulty"]
        depth = DIFFICULTY_DEPTH_MAP.get(difficulty, 3)
        result += f"- **AI Difficulty**: {difficulty.value.capitalize()} (search depth: {depth})\n"
```

#### 6.2: Update `format_board_json()`

**Location**: Lines 390-431 (function `format_board_json`)

**Find line 406** (after `"board_ascii": str(board)`):
```python
        "board_ascii": str(board)
    }
```

**Add before the closing brace:**
```python
    # Add AI difficulty if available
    if game_id in game_metadata and "difficulty" in game_metadata[game_id]:
        difficulty = game_metadata[game_id]["difficulty"]
        data["ai_difficulty"] = difficulty.value
        data["ai_search_depth"] = DIFFICULTY_DEPTH_MAP.get(difficulty, 3)
```

**Rationale**: Provides transparency to the agent about the current AI difficulty level.

---

### Step 7: Update Tool Docstring

**File**: `chess_mcp.py`
**Location**: Lines 446-461 (`chess_new_game()` docstring)

**Current docstring:**
```python
async def chess_new_game(params: NewGameInput) -> str:
    """Start a new chess game where the agent plays white and AI plays black.

    This tool initializes a new chess game with standard starting position.
    The agent will play white (moving first) and the server's AI will play black.

    Args:
        params (NewGameInput): Input parameters containing:
            - game_id (Optional[str]): Unique identifier for the game (default: "default")
            - response_format (ResponseFormat): Output format - 'markdown' or 'json'

    Returns:
        str: Game state showing the initial board position and legal moves for white.
            In JSON format: Contains game_id, fen, turn, legal_moves, and board state
            In Markdown format: Formatted board with game status and legal moves
    """
```

**Replace with:**
```python
async def chess_new_game(params: NewGameInput) -> str:
    """Start a new chess game where the agent plays white and AI plays black.

    This tool initializes a new chess game with standard starting position.
    The agent will play white (moving first) and the server's AI will play black.

    Args:
        params (NewGameInput): Input parameters containing:
            - game_id (Optional[str]): Unique identifier for the game (default: "default")
            - difficulty (Difficulty): AI opponent strength - 'easy', 'medium', or 'hard' (default: "medium")
                - easy: AI searches 2 moves ahead (faster, makes more mistakes)
                - medium: AI searches 3 moves ahead (balanced speed and strength)
                - hard: AI searches 4 moves ahead (slower, plays strong positional chess)
            - response_format (ResponseFormat): Output format - 'markdown' or 'json'

    Returns:
        str: Game state showing the initial board position and legal moves for white.
            In JSON format: Contains game_id, fen, turn, legal_moves, ai_difficulty, and board state
            In Markdown format: Formatted board with game status, AI difficulty, and legal moves
    """
```

---

## Testing Plan

### Test 1: Easy Difficulty Game

```python
import asyncio
from chess_mcp import chess_new_game, chess_submit_move, NewGameInput, SubmitMoveInput, Difficulty, ResponseFormat

async def test_easy_game():
    # Start easy game
    new_game_input = NewGameInput(game_id="easy_test", difficulty=Difficulty.EASY, response_format=ResponseFormat.JSON)
    result = await chess_new_game(new_game_input)
    print("Easy game started:")
    print(result)

    # Make a move
    move_input = SubmitMoveInput(game_id="easy_test", move="e4", response_format=ResponseFormat.JSON)
    result = await chess_submit_move(move_input)
    print("\nAfter move:")
    print(result)

    # Verify difficulty is stored
    assert '"ai_difficulty": "easy"' in result
    assert '"ai_search_depth": 2' in result

asyncio.run(test_easy_game())
```

**Expected**: Game starts with easy difficulty, AI responds quickly (< 1 second)

---

### Test 2: Hard Difficulty Game

```python
import asyncio
import time
from chess_mcp import chess_new_game, chess_submit_move, NewGameInput, SubmitMoveInput, Difficulty, ResponseFormat

async def test_hard_game():
    # Start hard game
    new_game_input = NewGameInput(game_id="hard_test", difficulty=Difficulty.HARD, response_format=ResponseFormat.MARKDOWN)
    result = await chess_new_game(new_game_input)
    print("Hard game started:")
    print(result)

    # Verify difficulty shown in markdown
    assert "AI Difficulty: Hard (search depth: 4)" in result

    # Make a move and time it
    move_input = SubmitMoveInput(game_id="hard_test", move="e4", response_format=ResponseFormat.MARKDOWN)
    start_time = time.time()
    result = await chess_submit_move(move_input)
    elapsed = time.time() - start_time

    print(f"\nAI response time: {elapsed:.2f}s")
    print(result)

    # Hard mode should take longer (but still reasonable)
    assert elapsed < 10.0, "Hard mode taking too long"

asyncio.run(test_hard_game())
```

**Expected**: Game starts with hard difficulty, AI takes longer to respond (2-5 seconds), but plays better moves

---

### Test 3: Default Difficulty (Backward Compatibility)

```python
import asyncio
from chess_mcp import chess_new_game, NewGameInput, ResponseFormat

async def test_default_difficulty():
    # Start game WITHOUT specifying difficulty
    new_game_input = NewGameInput(game_id="default_test", response_format=ResponseFormat.JSON)
    result = await chess_new_game(new_game_input)

    print("Default difficulty game:")
    print(result)

    # Should default to medium
    assert '"ai_difficulty": "medium"' in result
    assert '"ai_search_depth": 3' in result

asyncio.run(test_default_difficulty())
```

**Expected**: Defaults to medium difficulty when not specified

---

### Test 4: Multiple Concurrent Games with Different Difficulties

```python
import asyncio
from chess_mcp import chess_new_game, chess_submit_move, NewGameInput, SubmitMoveInput, Difficulty, ResponseFormat

async def test_concurrent_difficulties():
    # Start 3 games with different difficulties
    easy_input = NewGameInput(game_id="game_easy", difficulty=Difficulty.EASY, response_format=ResponseFormat.JSON)
    medium_input = NewGameInput(game_id="game_medium", difficulty=Difficulty.MEDIUM, response_format=ResponseFormat.JSON)
    hard_input = NewGameInput(game_id="game_hard", difficulty=Difficulty.HARD, response_format=ResponseFormat.JSON)

    await chess_new_game(easy_input)
    await chess_new_game(medium_input)
    await chess_new_game(hard_input)

    # Make same move in all games
    for game_id in ["game_easy", "game_medium", "game_hard"]:
        move_input = SubmitMoveInput(game_id=game_id, move="e4", response_format=ResponseFormat.JSON)
        result = await chess_submit_move(move_input)
        print(f"\n{game_id}:")
        print(result[:200])  # Print first 200 chars

    print("\n✅ All games maintain separate difficulty settings")

asyncio.run(test_concurrent_difficulties())
```

**Expected**: Each game maintains its own difficulty level independently

---

### Test 5: Performance Benchmark

```python
import asyncio
import time
from chess_mcp import chess_new_game, chess_submit_move, NewGameInput, SubmitMoveInput, Difficulty, ResponseFormat

async def benchmark_difficulties():
    """Measure AI response time at each difficulty level."""
    results = {}

    for difficulty in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
        # Start new game
        game_id = f"benchmark_{difficulty.value}"
        new_game_input = NewGameInput(game_id=game_id, difficulty=difficulty, response_format=ResponseFormat.JSON)
        await chess_new_game(new_game_input)

        # Make 3 moves and average response time
        times = []
        for move in ["e4", "Nf3", "Bc4"]:
            move_input = SubmitMoveInput(game_id=game_id, move=move, response_format=ResponseFormat.JSON)
            start = time.time()
            await chess_submit_move(move_input)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        results[difficulty.value] = avg_time
        print(f"{difficulty.value.capitalize()}: {avg_time:.3f}s average")

    # Verify performance expectations
    assert results["easy"] < results["medium"] < results["hard"], "Difficulty times should increase"
    assert results["easy"] < 1.0, "Easy should be fast"
    assert results["hard"] < 10.0, "Hard should still be playable"

    print("\n✅ Performance benchmark passed")

asyncio.run(benchmark_difficulties())
```

**Expected output:**
```
Easy: 0.150s average
Medium: 0.800s average
Hard: 3.500s average

✅ Performance benchmark passed
```

---

## Implementation Checklist

- [ ] Add `Difficulty` enum after `ResponseFormat` enum (Step 1)
- [ ] Add `DIFFICULTY_DEPTH_MAP` constant (Step 2)
- [ ] Update `NewGameInput` model with `difficulty` field (Step 3)
- [ ] Update `chess_new_game()` to store difficulty in metadata (Step 4)
- [ ] Update `chess_submit_move()` to use stored difficulty (Step 5)
- [ ] Update `format_board_markdown()` to show difficulty (Step 6.1)
- [ ] Update `format_board_json()` to include difficulty (Step 6.2)
- [ ] Update `chess_new_game()` docstring (Step 7)
- [ ] Run Test 1: Easy difficulty game
- [ ] Run Test 2: Hard difficulty game
- [ ] Run Test 3: Default difficulty (backward compatibility)
- [ ] Run Test 4: Concurrent games with different difficulties
- [ ] Run Test 5: Performance benchmark
- [ ] Update README.md with difficulty feature documentation

---

## README.md Update (Optional)

**Location**: After the "How to Use" section

**Add new section:**
```markdown
### AI Difficulty Levels

The chess server supports three difficulty levels for the AI opponent:

- **Easy** (depth 2): Fast responses (~0.2s), makes tactical mistakes, good for beginners
- **Medium** (depth 3): Balanced performance (~1s), plays solid chess, default setting
- **Hard** (depth 4): Slower responses (~3-5s), plays strong positional chess, challenging

Specify difficulty when starting a new game:

```python
# Easy game
await chess_new_game(NewGameInput(game_id="easy_game", difficulty=Difficulty.EASY))

# Hard game
await chess_new_game(NewGameInput(game_id="hard_game", difficulty=Difficulty.HARD))
```

If not specified, difficulty defaults to Medium.
```

---

## Implementation Time Estimate

- **Step 1-3** (Enum, constant, input model): 5 minutes
- **Step 4-5** (Store and use difficulty): 5 minutes
- **Step 6** (Update formatters): 5 minutes
- **Step 7** (Update docstring): 2 minutes
- **Testing** (all 5 tests): 15 minutes
- **README update**: 3 minutes

**Total: ~35 minutes**

---

## Additional Improvements (Future)

These could be considered for future enhancement:

1. **Custom Depth**: Allow agents to specify exact depth (1-5) instead of preset difficulties
2. **Opening Book**: Add opening book for hard mode to improve early game
3. **Time Control**: Add move time limits instead of depth limits
4. **Skill Rating**: Track and display estimated ELO for each difficulty
5. **Adaptive Difficulty**: AI adjusts difficulty based on player performance

---

## Notes

- **Backward Compatibility**: Existing code will continue to work with default medium difficulty
- **Performance**: Hard mode (depth 4) may take 3-5 seconds per move in complex positions
- **State Persistence**: Difficulty is stored in `game_metadata`, lost on server restart (like all game state)
- **Thread Safety**: Not currently an issue with FastMCP's async architecture, but would need consideration if adding multi-threading

---

## Questions or Issues?

If you encounter issues during implementation:

1. **Import errors**: Verify `Difficulty` enum is imported where needed
2. **Type errors**: Ensure `difficulty` field uses `Difficulty` type, not `str`
3. **Performance issues**: If hard mode is too slow, reduce depth to 3.5 (not integer) or optimize evaluation function
4. **Metadata missing**: Check that `chess_new_game()` properly initializes metadata before `chess_submit_move()` accesses it

Good luck with the implementation!
