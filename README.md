# Chess MCP Server

A Model Context Protocol (MCP) server that enables AI agents to play chess against an intelligent AI opponent. The agent plays white, and the server plays black using a minimax algorithm with alpha-beta pruning for strategic decision-making.

## Features

- **Full Chess Rules**: Complete implementation of chess rules including castling, en passant, and pawn promotion
- **Intelligent AI Opponent**: Black is played by an AI using minimax search with alpha-beta pruning
- **Move Validation**: All moves are validated to ensure they follow chess rules
- **Multiple Response Formats**: Supports both human-readable Markdown and machine-readable JSON formats
- **Game State Management**: Maintains complete game state including move history and board position
- **Multiple Games**: Support for multiple concurrent games with unique identifiers

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -e .
```

This will install the package and its dependencies (`mcp` and `python-chess`) as defined in `pyproject.toml`.

## Usage

### Starting the Server

The chess MCP server uses stdio transport by default:

```bash
python chess_mcp.py
```

For use with Claude Desktop or other MCP clients, add to your MCP configuration:

```json
{
  "mcpServers": {
    "chess": {
      "command": "python",
      "args": ["/path/to/chess_mcp.py"]
    }
  }
}
```

## Available Tools

### 1. chess_new_game

Start a new chess game.

**Parameters:**
- `game_id` (optional): Unique identifier for the game (default: "default")
- `response_format` (optional): "markdown" or "json" (default: "markdown")

**Example:**
```json
{
  "game_id": "game1",
  "response_format": "markdown"
}
```

**Returns:** Initial board position and legal moves for white

### 2. chess_submit_move

Submit a move for white and get AI's response.

**Parameters:**
- `move`: Move in standard algebraic notation (e.g., "e4", "Nf3", "O-O") or UCI format (e.g., "e2e4")
- `game_id` (optional): Game identifier (default: "default")
- `response_format` (optional): "markdown" or "json" (default: "markdown")

**Example:**
```json
{
  "move": "e4",
  "game_id": "game1",
  "response_format": "markdown"
}
```

**Returns:** Updated board position after both white's and black's moves

### 3. chess_get_board

Get the current board state without making a move.

**Parameters:**
- `game_id` (optional): Game identifier (default: "default")
- `include_legal_moves` (optional): Include list of legal moves (default: true)
- `response_format` (optional): "markdown" or "json" (default: "markdown")

**Example:**
```json
{
  "game_id": "game1",
  "include_legal_moves": true,
  "response_format": "json"
}
```

**Returns:** Current board position and game status

## Move Notation

The server supports two move notation formats:

### Standard Algebraic Notation (SAN)
- Pawn moves: `e4`, `d5`, `e5`
- Piece moves: `Nf3`, `Bb5`, `Qh4`
- Captures: `exd5`, `Nxe5`, `Bxf7+`
- Castling: `O-O` (kingside), `O-O-O` (queenside)
- Pawn promotion: `e8=Q`, `f1=N`
- Check: `Qh5+`
- Checkmate: `Qxf7#`

### Universal Chess Interface (UCI)
- Format: `[from_square][to_square][promotion]`
- Examples: `e2e4`, `g1f3`, `e7e8q`

## AI Strength

The AI opponent uses:
- **Algorithm**: Minimax with alpha-beta pruning
- **Search Depth**: 3 moves ahead (configurable in code via `DEFAULT_AI_DEPTH`)
- **Evaluation**: Material balance, positional factors, mobility, and tactical threats

The AI is competent and will provide a good challenge for most players. To adjust difficulty:
- Increase `DEFAULT_AI_DEPTH` for stronger play (slower)
- Decrease `DEFAULT_AI_DEPTH` for faster but weaker play

## Example Game Session

```
1. Start a new game:
   Tool: chess_new_game
   Input: {"game_id": "my_game"}
   Output: Shows starting position, legal moves

2. Make opening move:
   Tool: chess_submit_move
   Input: {"move": "e4", "game_id": "my_game"}
   Output: Shows e4, black's response (e.g., e5), and updated board

3. Continue playing:
   Tool: chess_submit_move
   Input: {"move": "Nf3", "game_id": "my_game"}
   Output: Shows Nf3, black's response, and updated board

4. Check current position:
   Tool: chess_get_board
   Input: {"game_id": "my_game"}
   Output: Current board state and legal moves
```

## Response Formats

### Markdown Format (Default)

Provides human-readable output with:
- ASCII board visualization
- Game status (turn, move number, check status)
- FEN notation
- List of legal moves for white
- Move history

### JSON Format

Provides machine-readable structured data:
```json
{
  "game_id": "my_game",
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "turn": "black",
  "move_number": 1,
  "is_check": false,
  "is_checkmate": false,
  "is_stalemate": false,
  "is_game_over": false,
  "board_ascii": "...",
  "legal_moves": ["a3", "a4", "b3", ...]
}
```

## Error Handling

The server provides clear, actionable error messages:

- **Game not found**: Suggests using `chess_new_game`
- **Illegal move**: Provides list of legal moves
- **Wrong turn**: Indicates whose turn it is
- **Game over**: Shows final position and result
- **Invalid notation**: Explains correct move format

## Technical Details

### Architecture

- **Framework**: FastMCP (MCP Python SDK)
- **Chess Engine**: python-chess library
- **AI Algorithm**: Minimax with alpha-beta pruning
- **State Management**: In-memory dictionary (production would use persistent storage)
- **Transport**: Stdio (standard input/output)

### Tool Annotations

All tools include proper MCP annotations:
- `chess_new_game`: Non-destructive, idempotent, closed-world
- `chess_submit_move`: Non-destructive, non-idempotent, closed-world
- `chess_get_board`: Read-only, idempotent, closed-world

### Code Quality

The implementation follows MCP best practices:
- Pydantic v2 models for input validation
- Type hints throughout
- Comprehensive docstrings
- Clear error messages
- Support for multiple response formats
- Proper async/await patterns

## Limitations

- **State Persistence**: Game state is stored in memory and will be lost when the server restarts
- **Concurrency**: Basic in-memory state management (not suitable for high-concurrency scenarios)
- **AI Depth**: Limited search depth for performance (can be adjusted)

## Future Enhancements

Potential improvements:
- Persistent storage (SQLite, PostgreSQL)
- Opening book integration
- Endgame tablebase support
- Move history with PGN export
- Time controls
- Difficulty levels
- Analysis mode with engine evaluation

## License

This chess MCP server is provided as-is for educational and development purposes.

## Credits

Built with:
- [python-chess](https://python-chess.readthedocs.io/) - Chess library
- [FastMCP](https://github.com/modelcontextprotocol/python-sdk) - MCP Python SDK

---

**Happy playing! May your tactics be sharp and your endgames be sound! ♟️**
