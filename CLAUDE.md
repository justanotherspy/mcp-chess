# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Model Context Protocol (MCP) server that enables AI agents to play chess. The agent plays white, and the server's AI opponent plays black using minimax with alpha-beta pruning. The server uses stdio transport and integrates with Claude Desktop or other MCP clients.

## Development Commands

### Testing
```bash
# Run the full test suite
python test_chess_mcp.py

# Run the server locally (stdio mode)
python chess_mcp.py
```

### Installation
```bash
# Install dependencies
pip install mcp python-chess
```

## Architecture

### Core Components

**chess_mcp.py** (550 lines) - Single-file MCP server containing:
- **MCP Tools** (lines 310-546): Three FastMCP tools (`chess_new_game`, `chess_submit_move`, `chess_get_board`)
- **AI Engine** (lines 110-230): Minimax algorithm with alpha-beta pruning and board evaluation
- **Input Models** (lines 38-106): Pydantic v2 validation models for each tool
- **Formatters** (lines 234-306): Dual output formats (Markdown for humans, JSON for machines)
- **State Management** (line 22): In-memory `game_state` dictionary (keyed by game_id)

### Key Design Patterns

**State Management**: All game state lives in a module-level dictionary `game_state: Dict[str, chess.Board]`. This means:
- Multiple concurrent games supported via `game_id` parameter
- State is lost on server restart (in-memory only)
- No persistence layer implemented

**AI Move Flow**:
1. User submits white move via `chess_submit_move`
2. Move validated against `board.legal_moves`
3. White move applied to board
4. AI calculates black's response using `get_best_move()` → `minimax()` → `evaluate_board()`
5. Both moves returned in response

**Evaluation Function** (lines 110-160):
- Material counting (standard piece values)
- Center control bonus for pieces in d4/d5/e4/e5
- Mobility evaluation (number of legal moves)
- Check bonus
- Checkmate detection

**Move Notation**: Supports both SAN (e.g., "Nf3", "O-O") and UCI (e.g., "e2e4") via python-chess's parsing

### MCP Tool Annotations

All tools properly annotated for MCP framework:
- `chess_new_game`: Non-destructive, idempotent (safe to call multiple times)
- `chess_submit_move`: Non-destructive, non-idempotent (changes game state)
- `chess_get_board`: Read-only, idempotent

### Response Formats

Each tool supports `response_format` parameter:
- **"markdown"** (default): Human-readable with ASCII board, game status, FEN, legal moves
- **"json"**: Machine-readable structured data

## Configuration

**AI Difficulty**: Set via `DEFAULT_AI_DEPTH` constant (line 27):
- Depth 2: Weaker, faster
- Depth 3: Default, balanced
- Depth 4+: Stronger, slower (exponential time increase)

## Known Limitations

1. **No Persistence**: Game state stored in memory only; restarts lose all games
2. **No Move History Export**: Can't export PGN format
3. **Single Depth Level**: AI depth not configurable per-game, only via code constant
4. **No Opening Book**: AI calculates all positions from scratch

## Testing Strategy

**test_chess_mcp.py** imports and calls tool functions directly (not via MCP protocol). Tests cover:
- New game creation
- Move submission (legal and illegal)
- Multi-game concurrency
- Both response formats
- Error handling

To test via actual MCP protocol, configure in Claude Desktop settings and interact naturally.

## Code Style

- Python 3.8+ with full type hints
- Pydantic v2 for input validation
- Async/await throughout (required by FastMCP)
- 4-space indentation
- Comprehensive docstrings on all public functions
