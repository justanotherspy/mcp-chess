# Quick Start Guide - Chess MCP Server

## What You've Got

A fully functional chess MCP server where AI agents play white against an intelligent black AI opponent. The server includes:

1. **chess_mcp.py** - The main server implementation
2. **README.md** - Complete documentation  
3. **test_chess_mcp.py** - Test suite demonstrating all features

## Installation & Setup

```bash
# Install dependencies
pip install mcp python-chess

# Test the server
python test_chess_mcp.py
```

## Quick Configuration for Claude Desktop

Add this to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "chess": {
      "command": "python",
      "args": ["/full/path/to/chess_mcp.py"]
    }
  }
}
```

## How It Works

### Three Simple Tools

1. **chess_new_game** - Start a new game
2. **chess_submit_move** - Make a move (e.g., "e4", "Nf3", "O-O")
3. **chess_get_board** - View current position

### Example Interaction

```
Agent: Start a new chess game
Server: Uses chess_new_game → Shows starting position

Agent: I'll play e4
Server: Uses chess_submit_move with "e4"
        → White plays e4, Black responds with e5
        → Shows updated board and legal moves

Agent: Let me play Nf3
Server: Uses chess_submit_move with "Nf3"
        → White plays Nf3, Black responds (e.g., Nc6)
        → Shows updated board and legal moves

Agent: Show me the current position
Server: Uses chess_get_board → Shows full board state
```

## Features

✅ **Full Chess Rules** - Castling, en passant, pawn promotion  
✅ **Smart AI** - Minimax with alpha-beta pruning (3-move depth)  
✅ **Move Validation** - All moves checked for legality  
✅ **Multiple Formats** - Markdown (human-readable) and JSON  
✅ **Multiple Games** - Concurrent games with unique IDs  
✅ **Clear Errors** - Helpful messages with legal move suggestions

## Move Notation Supported

### Standard Algebraic Notation (SAN)
- `e4` - Pawn moves
- `Nf3` - Knight to f3
- `Bb5` - Bishop to b5
- `O-O` - Kingside castle
- `O-O-O` - Queenside castle
- `e8=Q` - Pawn promotion

### UCI Format
- `e2e4` - Move from e2 to e4
- `e7e8q` - Pawn promotion

## AI Strength

The AI uses minimax with alpha-beta pruning at depth 3, which provides:
- Good tactical awareness
- Strategic positional play
- Fast response times (~1 second per move)

To adjust difficulty, edit `DEFAULT_AI_DEPTH` in chess_mcp.py:
- Depth 2: Faster, weaker play
- Depth 3: Balanced (default)
- Depth 4+: Stronger, slower play

## Testing

Run the comprehensive test suite:
```bash
python test_chess_mcp.py
```

This tests:
- Starting games
- Submitting moves
- Illegal move handling
- Multiple concurrent games
- Both JSON and Markdown formats
- Complete game scenarios

## Next Steps

1. **Test locally**: Run `python test_chess_mcp.py`
2. **Try it with Claude**: Configure in Claude Desktop
3. **Start playing**: Ask Claude to play chess with you
4. **Customize**: Adjust AI depth, add features, or extend functionality

## Tips for Using with Claude

Good prompts:
- "Let's play a game of chess. I'll play white."
- "Start a chess game and I'll make the first move: e4"
- "Show me the current board position"
- "What are my legal moves?"

## Architecture

- **Framework**: FastMCP (MCP Python SDK)
- **Chess Engine**: python-chess
- **AI**: Minimax with alpha-beta pruning
- **State**: In-memory (extend for persistence)
- **Transport**: Stdio

## Code Quality

Follows all MCP best practices:
✅ Pydantic v2 for validation  
✅ Type hints throughout  
✅ Comprehensive docstrings  
✅ Clear error messages  
✅ Proper async patterns  
✅ Tool annotations  
✅ Multiple response formats

---

**Ready to play? Start with `python test_chess_mcp.py` to see it in action! ♟️**
