# Simple 64x64 representation: index = from_square * 64 + to_square
# from_square and to_square are integers 0..63 (python-chess convention)
# Example usage:
#   idx = uci_to_index_4096("e2e4")
#   uci = index_to_uci_4096(idx)

import chess

def uci_to_index_4096(uci_move: str) -> int:
    mv = chess.Move.from_uci(uci_move)
    return mv.from_square * 64 + mv.to_square

def index_to_uci_4096(index: int) -> str:
    from_sq = index // 64
    to_sq = index % 64
    mv = chess.Move(from_sq, to_sq)
    return mv.uci()
