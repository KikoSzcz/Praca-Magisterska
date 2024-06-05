import chess
import chess.svg
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QMessageBox
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QMouseEvent
from xml.etree import ElementTree as ET
import torch
import torch.nn as nn
import re
import numpy as np
import chess
import pandas as pd
import gc
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
import random
from PyQt5.QtWidgets import QScrollArea

class module(nn.Module):

    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input
        x = self.activation2(x)
        return x
    
class ChessNet(nn.Module):
    def __init__(self, hidden_layers=6, hidden_size=300):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)

        return x

def load_model(model_path):
    model = ChessNet(hidden_layers=6, hidden_size=300)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def is_book_move(board):
    try:
        bookMoves = []
        with chess.polyglot.open_reader("data/bookfish.bin") as reader:
            for entry in reader.find_all(board):
                bookMoves.append(entry.move)
        return random.choice(bookMoves)
    except:
        return None

def check_mate_single(board):
    board = board.copy()
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push_uci(str(move))
        if board.is_checkmate():
            move = board.pop()
            return move
        _ = board.pop()

def handlePawnPromotion(board, move):
    if move.promotion is not None:
        board.push(move)  # Wykonaj ruch promocji
        promotion_square = chess.square_name(move.to_square)
        promotion_piece = chess.piece_name(move.promotion).upper()

def choose_move(model_temp, board):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_temp.to(device)

    color = None
    if board.turn:
        color = chess.WHITE
    else:
        color = chess.BLACK
    legal_moves = list(board.legal_moves)

    x = torch.Tensor(board_2_rep(board)).float().to('cuda')
    if color == chess.BLACK:
        x *= -1
    x = x.unsqueeze(0)
    move = model_temp(x.to(device))
    vals = []
    for legal_move in legal_moves:
        from_ = str(legal_move)[:2]
        to = str(legal_move)[2:]
        val_temp = move[0,:,:][0][8 - int(from_[1]), letter_2_num[from_[0]]]
        val = move[0,:,:][1][8 - int(to[1]), letter_2_num[to[0]]] #możliwe że ma być move[1,:,:]
        vals.append(val.cpu().detach() + val_temp.cpu().detach())
    
    try:
        choosen_move = legal_moves[np.argmax(vals)]
    except:
        choosen_move = np.random.choice(legal_moves)

    return choosen_move

def get_best_move(model_temp, board):
    book_move = None
    book_move = is_book_move(board)
    if book_move is not None:
        return book_move

    mate_move = check_mate_single(board)
    if mate_move is not None:
        return mate_move
    
    return choose_move(model_temp, board)

letter_2_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
num_2_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

def create_rep_layer(board, piece_type):
    s = str(board)
    s = re.sub(f'[^{piece_type}{piece_type.upper()} \n]', '.', s)
    s = re.sub(f'{piece_type}', '-1', s)
    s = re.sub(f'{piece_type.upper()}', '1', s)
    s = re.sub(f'\.', '0', s)

    board_mat = []
    for row in s.split('\n'):
        row = row.split(' ')
        row = [int(x) for x in row]
        board_mat.append(row)

    return np.array(board_mat)

def board_2_rep(board):
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    return board_rep

def move_2_rep(move, board):
    board.push_san(move).uci()
    move = str(board.pop())

    from_output_layer = np.zeros((8,8))
    from_row = 8 - int(move[1])
    from_column = letter_2_num[move[0]]
    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8,8))
    to_row = 8 - int(move[3])
    to_column = letter_2_num[move[2]]
    to_output_layer[to_row, to_column] = 1

    return np.stack([from_output_layer, to_output_layer])

def create_move_list(s):
    return re.sub('\d*\. ', '', s).split(' ')[:-1]
