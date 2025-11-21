# scripts/mcts.py
# A minimal MCTS implementation that collects leaf nodes (batch_size),
# builds a batch, calls model(batch) once, and backpropagates the results.
# Assumes model returns (values, policy_logits) where policy_logits is size [batch, 4096].
# This skeleton intentionally omits many optimizations (transposition table, virtual loss, etc.)
# but shows the batch-eval integration pattern.

import math
import random
import numpy as np
import torch
from collections import defaultdict
import chess
from torch_geometric.data import Batch
from .fen_to_graph_data_v2 import fen_to_graph_data_v2
from .uci_index import uci_to_index_4096

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # the move that led to this node (None for root)
        self.children = {}  # move -> MCTSNode
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0

    def q_value(self):
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def ucb_score(self, c_puct=1.25):
        if self.visits == 0:
            # give unvisited children a high exploration score
            return float('inf')
        parent_visits = self.parent.visits if self.parent is not None else 1
        return self.q_value() + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)

    def expand(self):
        # expand legal moves into children (but do not evaluate)
        for move in self.board.legal_moves:
            if move not in self.children:
                nb = self.board.copy()
                nb.push(move)
                self.children[move] = MCTSNode(nb, parent=self, move=move)

    def is_leaf(self):
        return len(self.children) == 0 or self.board.is_game_over()

class BatchMCTS:
    def __init__(self, model, device='cpu', batch_size=16):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def search(self, root_board: chess.Board, num_simulations=512):
        root = MCTSNode(root_board)
        # one-time expand root for priors
        root.expand()

        for _ in range((num_simulations + self.batch_size - 1) // self.batch_size):
            batch_nodes = []
            # SELECT + EXPAND (collect up to batch_size leaf nodes)
            for _ in range(self.batch_size):
                node = root
                # selection
                while not node.is_leaf() and not node.board.is_game_over():
                    # choose child with max ucb
                    if not node.children: # Should be handled by expand, but safety check
                        break
                    best = max(node.children.values(), key=lambda c: c.ucb_score())
                    node = best

                if node.board.is_game_over():
                    # terminal leaf - we'll handle value at backprop
                    batch_nodes.append((node, None))
                    continue

                # expand (create children for this node)
                if not node.children:
                    node.expand()

                batch_nodes.append((node, list(node.children.keys())))  # store move list for prior assignment

            if not batch_nodes:
                break

            # BUILD batch graphs for evaluation
            graphs = []
            nodes_metadata = []

            for node, moves in batch_nodes:
                try:
                    # FIX: node.board is already a Board object!
                    g = fen_to_graph_data_v2(node.board)
                    graphs.append(g)
                    nodes_metadata.append((node, moves))
                except Exception as e:
                    print(f"ERROR in graph construction: {e}")
                    # Skip this node
                    continue

            if not graphs:
                continue

            # EVALUATE BATCH
            batch = Batch.from_data_list(graphs).to(self.device)

            with torch.no_grad():
                values, policy_logits, _, _ = self.model(batch)

            # ASSIGN PRIORS & BACKPROPAGATE
            for i, (node, moves) in enumerate(nodes_metadata):
                value = values[i].item()
                policy = torch.softmax(policy_logits[i], dim=0).cpu().numpy()

                # Assign priors to children
                if moves is not None:
                    for mv in node.children.keys():
                        try:
                            idx = uci_to_index_4096(mv.uci())
                            node.children[mv].prior = float(policy[idx])
                        except Exception as e:
                            print(f"ERROR assigning prior for move {mv.uci()}: {e}")
                            node.children[mv].prior = 1e-8

                # Backpropagate value up the path
                cur = node
                cur_value = value
                while cur is not None:
                    cur.visits += 1
                    cur.value_sum += cur_value
                    cur_value = -cur_value  # perspective flip
                    cur = cur.parent

        # Return best move by visits
        if not root.children:
            # Fallback: return random legal move
            legal_moves = list(root.board.legal_moves)
            return legal_moves[0] if legal_moves else None

        best_move, best_child = max(root.children.items(), key=lambda x: x[1].visits)
        return best_move
