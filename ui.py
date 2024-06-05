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
import requests
import chess.engine
import math
from chess_engine import load_model, choose_move
from stockfish import Stockfish

engine_path = "/stockfish/stockfish-windows-x86-64-avx2.exe"  

global engine
engine = chess.engine.SimpleEngine.popen_uci(engine_path)
stockfish = Stockfish(engine_path)
global old_w_acc
old_w_acc = 0
global old_b_acc
old_b_acc = 0

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 900, 600)
        self.selected_square = None
        self.legal_moves = []
        self.model_selected_name = ""
        layout = QHBoxLayout(self)
        self.setLayout(layout)
        self.move_history = []
        self.leftWidget = QSvgWidget()
        self.leftWidget.installEventFilter(self)  # Install event filter to capture mouse events
        self.rightWidget = QWidget()
        self.rightWidget.setStyleSheet("background-color: #f0f0f0;")

        layout.addWidget(self.leftWidget, 2)
        layout.addWidget(self.rightWidget, 1)

        self.leftWidget.setGeometry(0, 0, 300, 600)
        self.rightWidget.setGeometry(0, 0, 600, 300)

        self.initRightWidget()

        self.chessboard = chess.Board()
        self.updateBoard()
        self.selected_model = None
        

    def initRightWidget(self):
        self.rightLayout = QVBoxLayout(self.rightWidget)
        self.rightLayout.setAlignment(Qt.AlignCenter)

        self.humanVsHumanButton = QPushButton("Zagraj z kimś innym", self.rightWidget)
        self.humanVsHumanButton.setMinimumHeight(50)
        self.humanVsHumanButton.setMaximumWidth(250)
        self.humanVsHumanButton.setStyleSheet("QPushButton { font-size: 16px; background-color: #4CAF50; color: white; border: 2px solid #4CAF50; border-radius: 10px; } QPushButton:hover { background-color: #45a049; }")
        self.humanVsHumanButton.clicked.connect(self.playAganitsPlayer)
        self.humanVsBotButton = QPushButton("Zagraj z botem", self.rightWidget)
        self.humanVsBotButton.setMinimumHeight(50)
        self.humanVsBotButton.setMaximumWidth(250)
        self.humanVsBotButton.setStyleSheet("QPushButton { font-size: 16px; background-color: #008CBA; color: white; border: 2px solid #008CBA; border-radius: 10px; } QPushButton:hover { background-color: #0077A3; }")

        self.humanVsBotButton.clicked.connect(self.showBotOptions)

        self.labelChooseBot = QLabel("Wybierz model:", self.rightWidget)
        self.labelChooseBot.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.labelChooseBot.hide()

        self.botOptions = QComboBox(self.rightWidget)
        self.botOptions.hide()


        self.movesLabel = QLabel("", self.rightWidget)
        self.humanVsBotButton.setMaximumHeight(250)
        self.movesLabel.setStyleSheet("font-size: 14px;")
        self.movesLabel.setAlignment(Qt.AlignTop)  # Wyrównanie tekstu do góry
        self.movesLabel.setWordWrap(True)  # Umożliwia zawijanie tekstu

        self.player_color = None
        self.rightLayout.addWidget(self.humanVsHumanButton)
        self.rightLayout.addWidget(self.humanVsBotButton)
        self.rightLayout.addWidget(self.labelChooseBot)
        self.rightLayout.addWidget(self.botOptions)
        self.rightLayout.addWidget(self.movesLabel)

    def playAganitsPlayer(self):
        self.rightLayout.setAlignment(Qt.AlignCenter)
        for widget in self.rightWidget.findChildren(QWidget):
            widget.hide()
        self.displayMoveHistory()

    def showBotOptions(self):
        self.rightLayout.setAlignment(Qt.AlignCenter)
        self.labelChooseBot.show()
        self.humanVsHumanButton.hide()
        self.humanVsBotButton.hide()
        self.movesLabel.hide()

        self.botOptions = QComboBox(self.rightWidget)
        self.botOptions.addItems(["", "Model_1_50_Acc_8312", "Model_2_100_Acc_8918", "Model_3_200_Acc_9533"])
        self.botOptions.setStyleSheet("font-size: 16px;")
        self.botOptions.currentIndexChanged.connect(self.onModelSelected)  # Connect signal to the new method
        self.botOptions.show()
        self.rightLayout.addWidget(self.botOptions)


    def onModelSelected(self):
        selected_model = self.botOptions.currentText()
        self.performActionBasedOnModel(selected_model)
        self.displayMoveHistory()

    def performActionBasedOnModel(self, model):
        print(f"Selected model: {model}")
        if model == "Model_1_50_Acc_8312":
            self.selected_model = load_model("D:/Studia/Semestr IV/Praca magisterska/moje testy i inne/GameDesktop/Models/Model1.pth")
        elif model == "Model_2_100_Acc_8918":
            self.selected_model = load_model("D:/Studia/Semestr IV/Praca magisterska/moje testy i inne/GameDesktop/Models/Model2.pth")
        elif model == "Model_3_200_Acc_9533":
            self.selected_model = load_model("D:/Studia/Semestr IV/Praca magisterska/moje testy i inne/GameDesktop/Models/Model3.pth")
        if self.selected_model is not None:
            self.player_color = chess.WHITE if random.choice([True, False]) else chess.BLACK
            self.model_selected_name = model
            self.botOptions.hide()
            self.labelChooseBot.hide()
            if self.player_color:
                computer_move = choose_move(self.selected_model, self.chessboard)
                self.chessboard.push(computer_move)
                self.updateMoveHistory(computer_move)
                self.updateBoard()

    def hideAll(self, result_message):
        self.rightLayout.setAlignment(Qt.AlignCenter)
        self.labelChooseBot.hide()
        self.botOptions.hide()
        self.humanVsHumanButton.hide()
        self.humanVsBotButton.hide()
        self.movesLabel.hide()

        win_message = QLabel(result_message, self.rightWidget)
        win_message.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.rightLayout.addWidget(win_message, alignment=Qt.AlignCenter)

        self.playAgainButton = QPushButton("Zagraj ponownie!", self.rightWidget)
        self.playAgainButton.setMinimumHeight(50)
        self.playAgainButton.setMaximumWidth(250)
        self.playAgainButton.setStyleSheet("QPushButton { font-size: 16px; background-color: #008CBA; color: white; border: 2px solid #008CBA; border-radius: 10px; } QPushButton:hover { background-color: #0077A3; }")
        self.playAgainButton.clicked.connect(self.resetWindow)
        self.rightLayout.addWidget(self.playAgainButton, alignment=Qt.AlignCenter)

    def resetWindow(self):
        self.rightLayout.setAlignment(Qt.AlignCenter)
        for widget in self.rightWidget.findChildren(QWidget):
            widget.hide()
        self.move_history = []
        self.humanVsBotButton.show()
        self.humanVsHumanButton.show()

        self.chessboard = chess.Board()
        self.updateBoard()
        global old_w_acc, old_b_acc
        old_w_acc = 0
        old_b_acc = 0

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and source == self.leftWidget:
            mouse_event = QMouseEvent(event)
            if mouse_event.button() == Qt.LeftButton:
                self.handleMouseClick(mouse_event.pos())
        return super().eventFilter(source, event)

    def handleMouseClick(self, position):
        board_size = min(self.leftWidget.width(), self.leftWidget.height())
        square_size = board_size / 8
        col = int(position.x() / square_size)
        row = 7 - int(position.y() / square_size)
        square = chess.square(col, row)
        print(f"Clicked on square: {chess.square_name(square)}")

        if self.selected_square is None:
            if self.chessboard.piece_at(square):
                self.selected_square = square
                self.legal_moves = [move for move in self.chessboard.legal_moves if move.from_square == square]
                print(f"Selected square: {chess.square_name(square)}")
                print(f"Legal moves: {[chess.square_name(move.to_square) for move in self.legal_moves]}")
                self.updateBoard()
        else:
            move = chess.Move(self.selected_square, square)
            moved = False
            if move in self.legal_moves:
                self.chessboard.push(move)
                moved = True
            else:
                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                if move in self.legal_moves:
                    self.chessboard.push(move)
                    moved = True
            self.selected_square = None
            self.legal_moves = []
            self.updateBoard()

            if moved:
                self.updateMoveHistory(move)

            if self.chessboard.is_game_over():
                moved = False
                result = self.chessboard.result()
                if result == "1-0":
                    self.hideAll("Biały wygrywa!")
                elif result == "0-1":
                    self.hideAll("Czarny wygrywa!")
                else:
                    self.hideAll("Remis!")

            if moved and self.player_color is not None:
                computer_move = choose_move(self.selected_model, self.chessboard)
                self.chessboard.push(computer_move)
                self.updateBoard()

                if self.chessboard.is_game_over():
                    result = self.chessboard.result()
                    if result == "1-0":
                        self.hideAll("Biały wygrywa!")
                    elif result == "0-1":
                        self.hideAll("Czarny wygrywa!")
                    else:
                        self.hideAll("Remis!")

                self.updateMoveHistory(computer_move)

    def updateMoveHistory(self, move):
        self.move_history.append(move)

        if len(self.move_history) > 50:
            del self.move_history[0]
            del self.move_history[0]
        get_board_accuracy(self.chessboard, move)
        self.displayMoveHistory()

    def displayMoveHistory(self):
        self.rightLayout.setAlignment(Qt.AlignTop)
        self.movesLabel.clear()
        move_text = ""
        if self.player_color is not None:
            move_text += "Bot: " + self.model_selected_name + "\n\nHistoria ruchów:\n"
        else:
            move_text += "Bot: Not selected\n\nHistoria ruchów:\n"

        moves_count = 1
        for idx, move in enumerate(self.move_history):
            if idx % 2 == 0:
                # Ruch białych
                move_text += f"{moves_count}.\t{move}\t\t"
                moves_count += 1
            else:
                # Ruch czarnych
                move_text += f"{move}\n"

            

        num_of_lines = 31 - moves_count - 2
        for i in range(num_of_lines):
            move_text += "\n"
        white_acc_temp = (old_w_acc / moves_count)
        black_acc_temp = (old_b_acc / moves_count)
        move_text += f"Dokładność białego:\t\t{white_acc_temp:.1f}%"
        move_text += f"\nDokładność czarnego:\t{black_acc_temp:.1f}%"

        self.movesLabel.setText(move_text)
        self.movesLabel.show()

    def updateBoard(self):
        if len(self.legal_moves) > 0: 
            squares = self.chessboard.attacks(chess.Square(self.selected_square))
            self.chessboardSvg = chess.svg.board(self.chessboard, squares=squares).encode("UTF-8")
        else:
            self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.leftWidget.load(self.chessboardSvg)

def compute_win_percent(centipawns):
    win_percent = 50 + 50 * (2 / (1 + math.exp(-0.00368208 * centipawns)) - 1)
    return win_percent

def compute_accuracy(win_percent_before, win_percent_after):
    accuracy = 103.1668 * math.exp(-0.04354 * (win_percent_before - win_percent_after)) - 3.1669
    return accuracy

def get_centipawn_evaluation(position):
    stockfish.set_fen_position(position)
    evaluation = stockfish.get_evaluation()
    if evaluation['type'] == 'cp':
        return evaluation['value']
    elif evaluation['type'] == 'mate':
        return 100000 if evaluation['value'] > 0 else -100000

def get_board_accuracy(board, move):
    global old_w_acc
    global old_b_acc

    board.pop()
    position_before = board.fen()
    board.push(move)
    position_after = board.fen()

    centipawns_before = get_centipawn_evaluation(position_before)
    centipawns_after = get_centipawn_evaluation(position_after)

    win_percent_before = compute_win_percent(centipawns_before)
    win_percent_after = compute_win_percent(centipawns_after)

    accuracy = compute_accuracy(win_percent_before, win_percent_after)
    
    if accuracy > 100:
        accuracy = 100

    if board.turn is False:
        old_b_acc += accuracy
    else:
        old_w_acc += accuracy


def lichess_white_expected_score(cp,p=-0.00368208):
    return 1.0/(1.0+np.exp(p*cp))

def lichess_move_accuracy(score_100_games_diff,par = [103.1668,-0.04354,-3.1669]):
    accuracy = par[0] * np.exp(par[1] * score_100_games_diff) + par[2]
    return accuracy