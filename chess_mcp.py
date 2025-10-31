#!/usr/bin/env python3
"""Chess MCP Server

An MCP server that allows AI agents to play chess against an AI opponent.
The agent plays white, and the server plays black using a minimax algorithm
with alpha-beta pruning.
"""

import json
from enum import Enum
from typing import Optional, List, Dict, Any
import chess
import chess.pgn
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

# Initialize the MCP server
mcp = FastMCP("chess_mcp")

# Game state storage (in a real deployment, use persistent storage)
game_state: Dict[str, chess.Board] = {}
game_metadata: Dict[str, Dict[str, Any]] = {}  # Stores resignation info: {game_id: {"resigned": bool, "resigned_side": str, "resign_reason": str}}
current_game_id = "default"

# ELO rating storage (in-memory)
player_ratings: Dict[str, int] = {}  # {player_id: rating}
game_history: Dict[str, List[Dict[str, Any]]] = {}  # {player_id: [game_records]}
player_stats: Dict[str, Dict[str, int]] = {}  # {player_id: {"wins": 0, "losses": 0, "draws": 0}}

# Constants
DEFAULT_AI_DEPTH = 3  # Search depth for AI (3-4 is reasonable)

# Difficulty to search depth mapping
DIFFICULTY_DEPTH_MAP = {
    "easy": 2,      # Searches 2 moves ahead - faster, weaker
    "medium": 3,    # Searches 3 moves ahead - balanced (default)
    "hard": 4,      # Searches 4 moves ahead - slower, stronger
}

# ELO Rating Configuration
DEFAULT_PLAYER_RATING = 1200  # Starting rating for new players
ELO_K_FACTOR = 32  # Rating change multiplier (32 is standard for developing players)

# AI difficulty to ELO rating mapping
AI_RATING_MAP = {
    "easy": 800,    # Conservative beginner level
    "medium": 1200, # Average club player
    "hard": 1600,   # Strong club player
}


class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class Difficulty(str, Enum):
    """AI difficulty levels for chess opponent."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# Pydantic Models for Input Validation

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
    player_id: Optional[str] = Field(
        default="default",
        description="Player identifier for ELO rating tracking (e.g., 'alice', 'user123')",
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


class SubmitMoveInput(BaseModel):
    """Input model for submitting a chess move."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    move: str = Field(
        ...,
        description="Move in standard algebraic notation (e.g., 'e4', 'Nf3', 'O-O') or UCI format (e.g., 'e2e4')",
        min_length=2,
        max_length=10
    )
    game_id: Optional[str] = Field(
        default="default",
        description="Game identifier (must match existing game)",
        min_length=1,
        max_length=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class GetBoardInput(BaseModel):
    """Input model for retrieving the current board state."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    game_id: Optional[str] = Field(
        default="default",
        description="Game identifier",
        min_length=1,
        max_length=100
    )
    include_legal_moves: bool = Field(
        default=True,
        description="Whether to include list of legal moves for white"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class ResignInput(BaseModel):
    """Input model for resigning a chess game."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    game_id: Optional[str] = Field(
        default="default",
        description="Game identifier",
        min_length=1,
        max_length=100
    )
    reason: Optional[str] = Field(
        default=None,
        description="Optional reason for resigning (e.g., 'Lost too much material', 'Position is hopeless')",
        max_length=200
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class GetRatingInput(BaseModel):
    """Input model for retrieving a player's ELO rating."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    player_id: str = Field(
        default="default",
        description="Player identifier to look up rating for",
        min_length=1,
        max_length=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


# ELO Rating Calculation Functions

def calculate_expected_score(player_rating: int, opponent_rating: int) -> float:
    """
    Calculate expected score for a player based on ELO rating difference.

    Uses the standard ELO formula:
    E = 1 / (1 + 10^((opponent_rating - player_rating) / 400))

    Args:
        player_rating: The player's current ELO rating
        opponent_rating: The opponent's ELO rating

    Returns:
        Expected score (probability of winning), between 0.0 and 1.0
        0.5 means equal chance, closer to 1.0 means player is favored
    """
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - player_rating) / 400.0))


def update_elo_rating(current_rating: int, expected_score: float, actual_score: float, k_factor: int = ELO_K_FACTOR) -> int:
    """
    Calculate new ELO rating after a game.

    Uses the standard ELO update formula:
    new_rating = current_rating + K * (actual_score - expected_score)

    Args:
        current_rating: Player's rating before the game
        expected_score: Expected score from calculate_expected_score()
        actual_score: Actual game result (1.0 = win, 0.5 = draw, 0.0 = loss)
        k_factor: Rating change multiplier (default: ELO_K_FACTOR)

    Returns:
        New ELO rating (rounded to nearest integer)
    """
    rating_change = k_factor * (actual_score - expected_score)
    new_rating = current_rating + rating_change
    return round(new_rating)


def get_game_result(board: chess.Board, game_id: str) -> Optional[float]:
    """
    Determine game result from white's perspective.

    Args:
        board: Current board position
        game_id: Game identifier to check for resignation

    Returns:
        1.0 if white won
        0.5 if draw
        0.0 if black won
        None if game is not over
    """
    # Check for resignation first
    if game_id in game_metadata and game_metadata[game_id].get("resigned", False):
        resigned_side = game_metadata[game_id].get("resigned_side", "")
        if resigned_side == "white":
            return 0.0  # White resigned, black wins
        elif resigned_side == "black":
            return 1.0  # Black resigned, white wins

    # Check for game over conditions
    if not board.is_game_over():
        return None

    if board.is_checkmate():
        # If it's white's turn and checkmate, white is checkmated (black wins)
        # If it's black's turn and checkmate, black is checkmated (white wins)
        return 0.0 if board.turn == chess.WHITE else 1.0

    # Stalemate, insufficient material, or other draw conditions
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5

    # Other game over conditions default to draw
    return 0.5


# Chess AI Implementation

def evaluate_board(board: chess.Board) -> float:
    """
    Evaluate the board position from black's perspective.
    Positive scores favor black, negative scores favor white.
    """
    if board.is_checkmate():
        return 10000 if board.turn == chess.WHITE else -10000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    # Piece values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    score = 0
    
    # Material evaluation
    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
    
    # Positional bonuses
    # Encourage center control
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            bonus = 30 if piece.piece_type == chess.PAWN else 10
            if piece.color == chess.BLACK:
                score += bonus
            else:
                score -= bonus
    
    # Mobility (number of legal moves for each side)
    current_turn = board.turn
    current_mobility = board.legal_moves.count()

    # Count opponent's mobility using a copy to avoid modifying original board
    board_copy = board.copy()
    board_copy.turn = not board_copy.turn
    opponent_mobility = board_copy.legal_moves.count()

    # Assign to correct sides
    if current_turn == chess.BLACK:
        black_mobility = current_mobility
        white_mobility = opponent_mobility
    else:
        white_mobility = current_mobility
        black_mobility = opponent_mobility

    # Apply mobility differential (more moves = better position)
    score += (black_mobility - white_mobility) * 2
    
    # Check bonus
    if board.is_check():
        score += 50 if board.turn == chess.WHITE else -50
    
    return score


def order_moves(board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
    """
    Order moves for better alpha-beta pruning efficiency.

    Searches good moves first to maximize cutoffs:
    1. Captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
    2. Checks
    3. Promotions
    4. Center control moves

    Args:
        board: Current board position
        moves: List of legal moves to order

    Returns:
        Ordered list of moves (best moves first for better pruning)
    """
    def move_priority(move: chess.Move) -> int:
        """Calculate priority score for a move (higher = better)."""
        priority = 0

        # 1. Captures (highest priority) - MVV-LVA heuristic
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            capturing_piece = board.piece_at(move.from_square)

            if captured_piece and capturing_piece:
                # Piece values for MVV-LVA
                piece_values = {
                    chess.PAWN: 1,
                    chess.KNIGHT: 3,
                    chess.BISHOP: 3,
                    chess.ROOK: 5,
                    chess.QUEEN: 9,
                    chess.KING: 0  # King captures are rare in legal moves
                }
                victim_value = piece_values.get(captured_piece.piece_type, 0)
                attacker_value = piece_values.get(capturing_piece.piece_type, 0)
                # Capturing valuable pieces with cheap pieces scores highest
                priority += 1000 + (victim_value * 10) - attacker_value

        # 2. Checks (second priority)
        board.push(move)
        is_check = board.is_check()
        board.pop()
        if is_check:
            priority += 100

        # 3. Promotions (third priority)
        if move.promotion:
            priority += 50

        # 4. Center control (fourth priority)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        if move.to_square in center_squares:
            priority += 10

        return priority

    # Sort moves by priority (descending - best first)
    return sorted(moves, key=move_priority, reverse=True)


def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> tuple[float, Optional[chess.Move]]:
    """
    Minimax algorithm with alpha-beta pruning.
    
    Args:
        board: Current board position
        depth: Search depth remaining
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        maximizing: True if maximizing (black), False if minimizing (white)
    
    Returns:
        Tuple of (evaluation score, best move)
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    legal_moves = list(board.legal_moves)
    legal_moves = order_moves(board, legal_moves)
    best_move = None
    
    if maximizing:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        
        return min_eval, best_move


def get_best_move(board: chess.Board, depth: int = DEFAULT_AI_DEPTH) -> chess.Move:
    """
    Get the best move for black using minimax algorithm.
    
    Args:
        board: Current board position (black to move)
        depth: Search depth for the algorithm
    
    Returns:
        Best move for black
    """
    _, best_move = minimax(board, depth, float('-inf'), float('inf'), True)
    return best_move if best_move else list(board.legal_moves)[0]


# Formatting Helpers

def format_board_markdown(board: chess.Board, game_id: str, include_legal_moves: bool = True) -> str:
    """Format board state as Markdown."""
    result = f"# Chess Game: {game_id}\n\n"

    # Board visualization
    result += "## Current Position\n\n"
    result += "```\n"
    result += str(board)
    result += "\n```\n\n"

    # Game status
    result += "## Game Status\n\n"
    result += f"- **Turn**: {'White' if board.turn == chess.WHITE else 'Black'}\n"
    result += f"- **Move Number**: {board.fullmove_number}\n"

    # Show AI difficulty if metadata exists
    if game_id in game_metadata and "difficulty" in game_metadata[game_id]:
        difficulty = game_metadata[game_id]["difficulty"]
        difficulty_value = difficulty.value if isinstance(difficulty, Difficulty) else difficulty
        depth = DIFFICULTY_DEPTH_MAP.get(difficulty_value, 3)
        result += f"- **AI Difficulty**: {difficulty_value.capitalize()} (search depth: {depth})\n"

    if board.is_check():
        result += f"- **Check**: Yes\n"

    # Check for resignation first
    if game_id in game_metadata and game_metadata[game_id].get("resigned", False):
        resigned_side = game_metadata[game_id].get("resigned_side", "white")
        resign_reason = game_metadata[game_id].get("resign_reason", "")
        winner = "Black" if resigned_side == "white" else "White"
        result += f"- **Result**: {resigned_side.capitalize()} resigned. {winner} wins!\n"
        if resign_reason:
            result += f"- **Reason**: {resign_reason}\n"
    elif board.is_checkmate():
        result += f"- **Result**: Checkmate! {'Black' if board.turn == chess.WHITE else 'White'} wins!\n"
    elif board.is_stalemate():
        result += f"- **Result**: Stalemate - Draw\n"
    elif board.is_insufficient_material():
        result += f"- **Result**: Draw by insufficient material\n"
    elif board.is_game_over():
        result += f"- **Result**: Game over\n"

    # Check if game is over (including resignations)
    game_over = board.is_game_over() or (game_id in game_metadata and game_metadata[game_id].get("resigned", False))

    # Show rating update if game ended and rating data exists
    if game_over and game_id in game_metadata and "rating_update" in game_metadata[game_id]:
        rating_data = game_metadata[game_id]["rating_update"]
        old_rating = rating_data["old_rating"]
        new_rating = rating_data["new_rating"]
        rating_change = rating_data["rating_change"]
        expected_score = rating_data["expected_score"]
        actual_score = rating_data["actual_score"]

        result += f"\n## Rating Update\n\n"

        # Show rating change with visual indicator
        change_indicator = "+" if rating_change >= 0 else ""
        result += f"- **Old Rating**: {old_rating}\n"
        result += f"- **New Rating**: {new_rating} ({change_indicator}{rating_change})\n"
        result += f"- **Expected Score**: {expected_score:.1%}\n"
        result += f"- **Actual Score**: {actual_score:.1%}\n"

    # FEN notation
    result += f"\n**FEN**: `{board.fen()}`\n"

    # Legal moves
    if include_legal_moves and not game_over and board.turn == chess.WHITE:
        result += f"\n## Legal Moves for White\n\n"
        moves = sorted([board.san(move) for move in board.legal_moves])

        # Group moves for better readability
        move_groups = [moves[i:i+10] for i in range(0, len(moves), 10)]
        for group in move_groups:
            result += f"{', '.join(group)}\n\n"

    return result


def format_board_json(board: chess.Board, game_id: str, include_legal_moves: bool = True) -> str:
    """Format board state as JSON."""
    # Check for resignation
    is_resigned = game_id in game_metadata and game_metadata[game_id].get("resigned", False)
    game_over = board.is_game_over() or is_resigned

    data = {
        "game_id": game_id,
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "move_number": board.fullmove_number,
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_game_over": game_over,
        "board_ascii": str(board)
    }

    # Add AI difficulty if available
    if game_id in game_metadata and "difficulty" in game_metadata[game_id]:
        difficulty = game_metadata[game_id]["difficulty"]
        difficulty_value = difficulty.value if isinstance(difficulty, Difficulty) else difficulty
        data["ai_difficulty"] = difficulty_value
        data["ai_search_depth"] = DIFFICULTY_DEPTH_MAP.get(difficulty_value, 3)

    # Handle resignation
    if is_resigned:
        resigned_side = game_metadata[game_id].get("resigned_side", "white")
        resign_reason = game_metadata[game_id].get("resign_reason", "")
        data["result"] = "resignation"
        data["resigned_side"] = resigned_side
        data["winner"] = "black" if resigned_side == "white" else "white"
        if resign_reason:
            data["resign_reason"] = resign_reason
    elif board.is_game_over():
        if board.is_checkmate():
            data["result"] = "checkmate"
            data["winner"] = "black" if board.turn == chess.WHITE else "white"
        elif board.is_stalemate():
            data["result"] = "stalemate"
        elif board.is_insufficient_material():
            data["result"] = "draw_insufficient_material"
        else:
            data["result"] = "game_over"

    if include_legal_moves and not game_over and board.turn == chess.WHITE:
        data["legal_moves"] = sorted([board.san(move) for move in board.legal_moves])

    # Add rating update if game ended and rating data exists
    if game_over and game_id in game_metadata and "rating_update" in game_metadata[game_id]:
        data["rating_update"] = game_metadata[game_id]["rating_update"]

    return json.dumps(data, indent=2)


def format_rating_markdown(player_id: str) -> str:
    """Format player rating and statistics as Markdown."""
    # Check if player exists
    if player_id not in player_ratings:
        return f"# Player Rating\n\nPlayer '{player_id}' not found. Play a game to initialize your rating."

    rating = player_ratings[player_id]
    stats = player_stats.get(player_id, {"wins": 0, "losses": 0, "draws": 0})
    history = game_history.get(player_id, [])

    result = f"# Player Rating: {player_id}\n\n"
    result += f"## Current Rating\n\n"
    result += f"**{rating}** ELO\n\n"

    # Statistics
    total_games = stats["wins"] + stats["losses"] + stats["draws"]
    result += f"## Statistics\n\n"
    result += f"- **Games Played**: {total_games}\n"
    result += f"- **Wins**: {stats['wins']}\n"
    result += f"- **Losses**: {stats['losses']}\n"
    result += f"- **Draws**: {stats['draws']}\n"

    if total_games > 0:
        win_rate = (stats["wins"] / total_games) * 100
        result += f"- **Win Rate**: {win_rate:.1f}%\n"

    # Recent game history (last 10 games)
    if history:
        result += f"\n## Recent Games (Last {min(10, len(history))})\n\n"
        result += "| Game ID | Difficulty | Result | Rating Change | New Rating |\n"
        result += "|---------|------------|--------|---------------|------------|\n"

        for game in reversed(history[-10:]):
            game_id = game["game_id"]
            difficulty = game["difficulty"].capitalize()
            result_val = game["result"]
            result_str = "Win" if result_val == 1.0 else ("Draw" if result_val == 0.5 else "Loss")
            rating_change = game["rating_change"]
            new_rating = game["new_rating"]
            change_str = f"+{rating_change}" if rating_change >= 0 else str(rating_change)

            result += f"| {game_id} | {difficulty} | {result_str} | {change_str} | {new_rating} |\n"

        result += "\n"

    return result


def format_rating_json(player_id: str) -> str:
    """Format player rating and statistics as JSON."""
    # Check if player exists
    if player_id not in player_ratings:
        return json.dumps({"error": f"Player '{player_id}' not found"}, indent=2)

    rating = player_ratings[player_id]
    stats = player_stats.get(player_id, {"wins": 0, "losses": 0, "draws": 0})
    history = game_history.get(player_id, [])

    total_games = stats["wins"] + stats["losses"] + stats["draws"]
    win_rate = (stats["wins"] / total_games * 100) if total_games > 0 else 0.0

    data = {
        "player_id": player_id,
        "rating": rating,
        "statistics": {
            "games_played": total_games,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "draws": stats["draws"],
            "win_rate": round(win_rate, 1)
        },
        "recent_games": history[-10:]  # Last 10 games
    }

    return json.dumps(data, indent=2)


# MCP Tools

@mcp.tool(
    name="chess_new_game",
    annotations={
        "title": "Start New Chess Game",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def chess_new_game(params: NewGameInput) -> str:
    """Start a new chess game where the agent plays white and AI plays black.

    This tool initializes a new chess game with standard starting position.
    The agent will play white (moving first) and the server's AI will play black.

    Args:
        params (NewGameInput): Input parameters containing:
            - game_id (Optional[str]): Unique identifier for the game (default: "default")
            - player_id (Optional[str]): Player identifier for ELO tracking (default: "default")
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
    try:
        # Initialize player rating if this is a new player
        if params.player_id not in player_ratings:
            player_ratings[params.player_id] = DEFAULT_PLAYER_RATING
            player_stats[params.player_id] = {"wins": 0, "losses": 0, "draws": 0}
            game_history[params.player_id] = []

        # Create new game
        board = chess.Board()
        game_state[params.game_id] = board

        # Store game metadata (difficulty, player_id, and clear resignation state)
        game_metadata[params.game_id] = {
            "difficulty": params.difficulty,
            "player_id": params.player_id,
            "resigned": False
        }

        # Format response
        if params.response_format == ResponseFormat.JSON:
            return format_board_json(board, params.game_id, include_legal_moves=True)
        else:
            return format_board_markdown(board, params.game_id, include_legal_moves=True)
    
    except Exception as e:
        error_msg = f"Error starting new game: {str(e)}"
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"error": error_msg}, indent=2)
        else:
            return f"❌ {error_msg}"


@mcp.tool(
    name="chess_submit_move",
    annotations={
        "title": "Submit Chess Move",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def chess_submit_move(params: SubmitMoveInput) -> str:
    """Submit a move for white and get the AI's response move for black.
    
    This tool validates the submitted move for white, applies it to the board,
    then calculates and applies the AI's response move for black. Both moves
    are validated to ensure they are legal.
    
    Args:
        params (SubmitMoveInput): Input parameters containing:
            - move (str): Move in standard algebraic notation (e.g., 'e4', 'Nf3') or UCI (e.g., 'e2e4')
            - game_id (Optional[str]): Game identifier (default: "default")
            - response_format (ResponseFormat): Output format - 'markdown' or 'json'
    
    Returns:
        str: Updated game state after both white's and black's moves.
            Shows the current board position, move history, and legal moves if game continues.
            Returns error message if move is illegal or game doesn't exist.
    
    Error Handling:
        - If game_id doesn't exist: Returns error suggesting to start a new game
        - If move is illegal: Returns error with list of legal moves
        - If it's not white's turn: Returns error message
        - If game is already over: Returns game result
    """
    try:
        # Check if game exists
        if params.game_id not in game_state:
            error_msg = f"Game '{params.game_id}' not found. Use chess_new_game to start a new game."
            if params.response_format == ResponseFormat.JSON:
                return json.dumps({"error": error_msg}, indent=2)
            else:
                return f"❌ {error_msg}"
        
        board = game_state[params.game_id]

        # Check if game is over (including resignations)
        is_resigned = params.game_id in game_metadata and game_metadata[params.game_id].get("resigned", False)
        if board.is_game_over() or is_resigned:
            error_msg = "Game is already over. Use chess_new_game to start a new game."
            if params.response_format == ResponseFormat.JSON:
                result = {
                    "error": error_msg,
                    "game_over": True,
                    "result": "checkmate" if board.is_checkmate() else "draw"
                }
                return json.dumps(result, indent=2)
            else:
                return f"❌ {error_msg}\n\n{format_board_markdown(board, params.game_id, False)}"
        
        # Check if it's white's turn
        if board.turn != chess.WHITE:
            error_msg = "It's black's turn (AI is thinking). Current state:"
            if params.response_format == ResponseFormat.JSON:
                return json.dumps({"error": error_msg}, indent=2)
            else:
                return f"❌ {error_msg}\n\n{format_board_markdown(board, params.game_id, False)}"
        
        # Try to parse and apply white's move
        try:
            # Try SAN notation first
            try:
                white_move = board.parse_san(params.move)
            except:
                # Try UCI notation
                white_move = chess.Move.from_uci(params.move)
            
            # Validate move is legal
            if white_move not in board.legal_moves:
                legal_moves = sorted([board.san(m) for m in board.legal_moves])
                error_msg = f"Illegal move: '{params.move}'"
                if params.response_format == ResponseFormat.JSON:
                    return json.dumps({
                        "error": error_msg,
                        "legal_moves": legal_moves
                    }, indent=2)
                else:
                    return f"❌ {error_msg}\n\nLegal moves: {', '.join(legal_moves)}"
            
            # Get SAN notation before applying the move
            white_move_san = board.san(white_move)
            
            # Apply white's move
            board.push(white_move)
            
        except Exception as e:
            legal_moves = sorted([board.san(m) for m in board.legal_moves])
            error_msg = f"Invalid move format: '{params.move}'. Error: {str(e)}"
            if params.response_format == ResponseFormat.JSON:
                return json.dumps({
                    "error": error_msg,
                    "legal_moves": legal_moves
                }, indent=2)
            else:
                return f"❌ {error_msg}\n\nLegal moves: {', '.join(legal_moves)}"
        
        # Check if game ended after white's move
        if board.is_game_over():
            # Calculate and update ELO rating
            player_id = game_metadata.get(params.game_id, {}).get("player_id", "default")
            game_result = get_game_result(board, params.game_id)

            if game_result is not None and player_id in player_ratings:
                # Get AI rating based on difficulty
                difficulty = game_metadata.get(params.game_id, {}).get("difficulty", Difficulty.MEDIUM)
                difficulty_value = difficulty.value if isinstance(difficulty, Difficulty) else difficulty
                ai_rating = AI_RATING_MAP.get(difficulty_value, 1200)

                # Calculate rating change
                old_rating = player_ratings[player_id]
                expected_score = calculate_expected_score(old_rating, ai_rating)
                new_rating = update_elo_rating(old_rating, expected_score, game_result)
                player_ratings[player_id] = new_rating

                # Update stats
                if game_result == 1.0:
                    player_stats[player_id]["wins"] += 1
                elif game_result == 0.0:
                    player_stats[player_id]["losses"] += 1
                else:
                    player_stats[player_id]["draws"] += 1

                # Record game in history
                game_history[player_id].append({
                    "game_id": params.game_id,
                    "difficulty": difficulty_value,
                    "result": game_result,
                    "old_rating": old_rating,
                    "new_rating": new_rating,
                    "rating_change": new_rating - old_rating,
                    "timestamp": datetime.now().isoformat()
                })

                # Store rating update in metadata for formatting
                game_metadata[params.game_id]["rating_update"] = {
                    "old_rating": old_rating,
                    "new_rating": new_rating,
                    "rating_change": new_rating - old_rating,
                    "expected_score": expected_score,
                    "actual_score": game_result
                }

            if params.response_format == ResponseFormat.JSON:
                return format_board_json(board, params.game_id, include_legal_moves=False)
            else:
                result = format_board_markdown(board, params.game_id, include_legal_moves=False)
                return f"White played: **{white_move_san}**\n\n{result}"

        # AI evaluates position and decides whether to resign or move
        # Get difficulty level for this game (default to MEDIUM if not set)
        game_difficulty = game_metadata.get(params.game_id, {}).get("difficulty", Difficulty.MEDIUM)
        ai_depth = DIFFICULTY_DEPTH_MAP[game_difficulty.value if isinstance(game_difficulty, Difficulty) else game_difficulty]
        eval_score, black_move = minimax(board, ai_depth, float('-inf'), float('inf'), True)

        # AI auto-resign logic: if position is hopeless (down more than 1500 centipawns), resign
        AI_RESIGN_THRESHOLD = -1500  # Roughly 1.5 pawns or a minor piece down with bad position
        if eval_score < AI_RESIGN_THRESHOLD:
            # AI resigns
            difficulty = game_metadata.get(params.game_id, {}).get("difficulty", Difficulty.MEDIUM)
            player_id = game_metadata.get(params.game_id, {}).get("player_id", "default")

            game_metadata[params.game_id]["resigned"] = True
            game_metadata[params.game_id]["resigned_side"] = "black"
            game_metadata[params.game_id]["resign_reason"] = "Position is hopeless"

            # Calculate and update ELO rating
            game_result = get_game_result(board, params.game_id)

            if game_result is not None and player_id in player_ratings:
                # Get AI rating based on difficulty
                difficulty_value = difficulty.value if isinstance(difficulty, Difficulty) else difficulty
                ai_rating = AI_RATING_MAP.get(difficulty_value, 1200)

                # Calculate rating change
                old_rating = player_ratings[player_id]
                expected_score = calculate_expected_score(old_rating, ai_rating)
                new_rating = update_elo_rating(old_rating, expected_score, game_result)
                player_ratings[player_id] = new_rating

                # Update stats
                if game_result == 1.0:
                    player_stats[player_id]["wins"] += 1
                elif game_result == 0.0:
                    player_stats[player_id]["losses"] += 1
                else:
                    player_stats[player_id]["draws"] += 1

                # Record game in history
                game_history[player_id].append({
                    "game_id": params.game_id,
                    "difficulty": difficulty_value,
                    "result": game_result,
                    "old_rating": old_rating,
                    "new_rating": new_rating,
                    "rating_change": new_rating - old_rating,
                    "timestamp": datetime.now().isoformat()
                })

                # Store rating update in metadata for formatting
                game_metadata[params.game_id]["rating_update"] = {
                    "old_rating": old_rating,
                    "new_rating": new_rating,
                    "rating_change": new_rating - old_rating,
                    "expected_score": expected_score,
                    "actual_score": game_result
                }

            # Format resignation response
            if params.response_format == ResponseFormat.JSON:
                response = format_board_json(board, params.game_id, include_legal_moves=False)
                data = json.loads(response)
                data["last_moves"] = {
                    "white": white_move_san,
                    "black": "resigned"
                }
                data["ai_evaluation"] = eval_score
                return json.dumps(data, indent=2)
            else:
                result = format_board_markdown(board, params.game_id, include_legal_moves=False)
                return f"**Moves played:**\n- White: {white_move_san}\n- Black: **Resigned** (Position is hopeless)\n\n{result}"

        # AI makes black's move
        if not black_move:
            black_move = list(board.legal_moves)[0]  # Fallback if minimax fails
        black_move_san = board.san(black_move)
        board.push(black_move)

        # Check if game ended after black's move
        if board.is_game_over():
            # Calculate and update ELO rating
            player_id = game_metadata.get(params.game_id, {}).get("player_id", "default")
            game_result = get_game_result(board, params.game_id)

            if game_result is not None and player_id in player_ratings:
                # Get AI rating based on difficulty
                difficulty = game_metadata.get(params.game_id, {}).get("difficulty", Difficulty.MEDIUM)
                difficulty_value = difficulty.value if isinstance(difficulty, Difficulty) else difficulty
                ai_rating = AI_RATING_MAP.get(difficulty_value, 1200)

                # Calculate rating change
                old_rating = player_ratings[player_id]
                expected_score = calculate_expected_score(old_rating, ai_rating)
                new_rating = update_elo_rating(old_rating, expected_score, game_result)
                player_ratings[player_id] = new_rating

                # Update stats
                if game_result == 1.0:
                    player_stats[player_id]["wins"] += 1
                elif game_result == 0.0:
                    player_stats[player_id]["losses"] += 1
                else:
                    player_stats[player_id]["draws"] += 1

                # Record game in history
                game_history[player_id].append({
                    "game_id": params.game_id,
                    "difficulty": difficulty_value,
                    "result": game_result,
                    "old_rating": old_rating,
                    "new_rating": new_rating,
                    "rating_change": new_rating - old_rating,
                    "timestamp": datetime.now().isoformat()
                })

                # Store rating update in metadata for formatting
                game_metadata[params.game_id]["rating_update"] = {
                    "old_rating": old_rating,
                    "new_rating": new_rating,
                    "rating_change": new_rating - old_rating,
                    "expected_score": expected_score,
                    "actual_score": game_result
                }

        # Format response
        if params.response_format == ResponseFormat.JSON:
            response = format_board_json(board, params.game_id, include_legal_moves=not board.is_game_over())
            data = json.loads(response)
            data["last_moves"] = {
                "white": white_move_san,
                "black": black_move_san
            }
            return json.dumps(data, indent=2)
        else:
            result = format_board_markdown(board, params.game_id, include_legal_moves=not board.is_game_over())
            return f"**Moves played:**\n- White: {white_move_san}\n- Black: {black_move_san}\n\n{result}"
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"error": error_msg}, indent=2)
        else:
            return f"❌ {error_msg}"


@mcp.tool(
    name="chess_resign",
    annotations={
        "title": "Resign Chess Game",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def chess_resign(params: ResignInput) -> str:
    """Resign the current chess game, ending it immediately.

    This tool allows white (the user) to resign the game, giving black (AI) the win.
    The game state is preserved in memory but marked as ended by resignation.

    Args:
        params (ResignInput): Input parameters containing:
            - game_id (Optional[str]): Game identifier (default: "default")
            - reason (Optional[str]): Reason for resigning (e.g., "Lost too much material")
            - response_format (ResponseFormat): Output format - 'markdown' or 'json'

    Returns:
        str: Final game state showing the resignation result and final board position.
            In JSON format: Contains game_id, result="resignation", winner, and resign_reason
            In Markdown format: Formatted board with resignation message

    Error Handling:
        - If game_id doesn't exist: Returns error suggesting to start a new game
        - If game is already over: Returns error indicating game has ended
    """
    try:
        # Check if game exists
        if params.game_id not in game_state:
            error_msg = f"Game '{params.game_id}' not found. Use chess_new_game to start a new game."
            if params.response_format == ResponseFormat.JSON:
                return json.dumps({"error": error_msg}, indent=2)
            else:
                return f"❌ {error_msg}"

        board = game_state[params.game_id]

        # Check if game is already over
        is_already_resigned = params.game_id in game_metadata and game_metadata[params.game_id].get("resigned", False)
        if board.is_game_over() or is_already_resigned:
            error_msg = "Game is already over. Use chess_new_game to start a new game."
            if params.response_format == ResponseFormat.JSON:
                result = {
                    "error": error_msg,
                    "game_over": True
                }
                return json.dumps(result, indent=2)
            else:
                return f"❌ {error_msg}\n\n{format_board_markdown(board, params.game_id, False)}"

        # Mark game as resigned (white resigns by default)
        difficulty = game_metadata.get(params.game_id, {}).get("difficulty", Difficulty.MEDIUM)
        player_id = game_metadata.get(params.game_id, {}).get("player_id", "default")

        game_metadata[params.game_id]["resigned"] = True
        game_metadata[params.game_id]["resigned_side"] = "white"
        game_metadata[params.game_id]["resign_reason"] = params.reason or ""

        # Calculate and update ELO rating
        game_result = get_game_result(board, params.game_id)

        if game_result is not None and player_id in player_ratings:
            # Get AI rating based on difficulty
            difficulty_value = difficulty.value if isinstance(difficulty, Difficulty) else difficulty
            ai_rating = AI_RATING_MAP.get(difficulty_value, 1200)

            # Calculate rating change
            old_rating = player_ratings[player_id]
            expected_score = calculate_expected_score(old_rating, ai_rating)
            new_rating = update_elo_rating(old_rating, expected_score, game_result)
            player_ratings[player_id] = new_rating

            # Update stats
            if game_result == 1.0:
                player_stats[player_id]["wins"] += 1
            elif game_result == 0.0:
                player_stats[player_id]["losses"] += 1
            else:
                player_stats[player_id]["draws"] += 1

            # Record game in history
            game_history[player_id].append({
                "game_id": params.game_id,
                "difficulty": difficulty_value,
                "result": game_result,
                "old_rating": old_rating,
                "new_rating": new_rating,
                "rating_change": new_rating - old_rating,
                "timestamp": datetime.now().isoformat()
            })

            # Store rating update in metadata for formatting
            game_metadata[params.game_id]["rating_update"] = {
                "old_rating": old_rating,
                "new_rating": new_rating,
                "rating_change": new_rating - old_rating,
                "expected_score": expected_score,
                "actual_score": game_result
            }

        # Format response
        if params.response_format == ResponseFormat.JSON:
            return format_board_json(board, params.game_id, include_legal_moves=False)
        else:
            result = format_board_markdown(board, params.game_id, include_legal_moves=False)
            return f"**White has resigned.**\n\n{result}"

    except Exception as e:
        error_msg = f"Error processing resignation: {str(e)}"
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"error": error_msg}, indent=2)
        else:
            return f"❌ {error_msg}"


@mcp.tool(
    name="chess_get_board",
    annotations={
        "title": "Get Current Board State",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def chess_get_board(params: GetBoardInput) -> str:
    """Get the current state of the chess board.
    
    This tool returns the current board position, game status, and optionally
    the list of legal moves for white. Use this to check the current state
    without making any moves.
    
    Args:
        params (GetBoardInput): Input parameters containing:
            - game_id (Optional[str]): Game identifier (default: "default")
            - include_legal_moves (bool): Whether to include legal moves for white (default: True)
            - response_format (ResponseFormat): Output format - 'markdown' or 'json'
    
    Returns:
        str: Current board state with position, game status, and optional legal moves.
            In JSON format: Contains game_id, fen, turn, board state, and optional legal moves
            In Markdown format: Formatted board with game status and optional legal moves
    
    Error Handling:
        - If game_id doesn't exist: Returns error suggesting to start a new game
    """
    try:
        # Check if game exists
        if params.game_id not in game_state:
            error_msg = f"Game '{params.game_id}' not found. Use chess_new_game to start a new game."
            if params.response_format == ResponseFormat.JSON:
                return json.dumps({"error": error_msg}, indent=2)
            else:
                return f"❌ {error_msg}"
        
        board = game_state[params.game_id]
        
        # Format response
        if params.response_format == ResponseFormat.JSON:
            return format_board_json(board, params.game_id, params.include_legal_moves)
        else:
            return format_board_markdown(board, params.game_id, params.include_legal_moves)
    
    except Exception as e:
        error_msg = f"Error retrieving board state: {str(e)}"
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"error": error_msg}, indent=2)
        else:
            return f"❌ {error_msg}"


@mcp.tool(
    name="chess_get_rating",
    annotations={
        "title": "Get Player ELO Rating",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def chess_get_rating(params: GetRatingInput) -> str:
    """Get the current ELO rating and statistics for a player.

    This tool retrieves a player's current ELO rating, win/loss/draw statistics,
    and recent game history. Use this to track player performance over time.

    Args:
        params (GetRatingInput): Input parameters containing:
            - player_id (str): Player identifier to look up (default: "default")
            - response_format (ResponseFormat): Output format - 'markdown' or 'json'

    Returns:
        str: Player rating information including current rating, statistics, and game history.
            In JSON format: Contains player_id, rating, statistics object, and recent_games array
            In Markdown format: Formatted table with rating, stats, and recent games

    Error Handling:
        - If player_id doesn't exist: Returns message indicating player not found
    """
    try:
        # Format response
        if params.response_format == ResponseFormat.JSON:
            return format_rating_json(params.player_id)
        else:
            return format_rating_markdown(params.player_id)

    except Exception as e:
        error_msg = f"Error retrieving player rating: {str(e)}"
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"error": error_msg}, indent=2)
        else:
            return f"❌ {error_msg}"


if __name__ == "__main__":
    # Run the MCP server with stdio transport
    mcp.run()
