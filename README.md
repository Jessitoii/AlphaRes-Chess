♟️ AlphaZero-Inspired Chess AI (PyTorch)
========================================

A modern, dual-headed Deep Learning Chess Engine implemented in PyTorch. This project evolves from a simple MLP approach to a robust **ResNet (Residual Network)** architecture inspired by DeepMind's AlphaZero.

It learns to play chess by imitating Grandmaster games and Stockfish evaluations, predicting both the **best move (Policy)** and the **game outcome (Value)** simultaneously.

🚀 Key Features
---------------

*   **Dual-Headed Architecture:** A single neural network backbone that splits into two heads:
    
    *   **Policy Head:** Predicts the probability distribution over all possible moves ($4096$ output space).
        
    *   **Value Head:** Predicts the evaluation of the current position ($-1$ to $+1$).
        
*   **ResNet Backbone:** Utilizes Residual Blocks with Skip Connections, Batch Normalization, and CNNs to capture deep spatial relationships on the board without vanishing gradients.
    
*   **12-Channel Bitboard Input:** Instead of simple integers, the board is represented as a $12 \\times 8 \\times 8$ tensor (One-hot encoded channels for each piece type: P, N, B, R, Q, K for both colors).
    
*   **Legal Move Masking:** The inference engine filters raw network predictions against legal moves, ensuring the AI never attempts illegal actions.
    
*   **Multiprocessed Data Pipeline:** A high-performance preprocessing script that utilizes all CPU cores to parse PGN files and extract Stockfish evaluations (Policy/Value) into compressed NumPy arrays.
    

🧠 Model Architecture
---------------------

The network processes the board state as an image-like tensor and outputs two distinct predictions.

```text
   Input (12x8x8 Bitboard)         │  [Convolution + Batch Norm + ReLU]         │  [ Residual Block x 5 ] <--- Deep Feature Extraction         │     ┌───┴────────────────────────┐     ▼                            ▼  [Policy Head]              [Value Head]     │                            │  [Conv 1x1]                 [Conv 1x1]     │                            │  [Fully Connected]          [Fully Connected]     │                            │  [LogSoftmax]               [Tanh]     ▼                            ▼  Move Probabilities         Win Probability  (Shape: 4096)              (Shape: 1, Range: -1 to 1)
   ```

### Loss Function

The model optimizes a combined loss function:

$$Loss\_{total} = (Loss\_{policy} \\times \\text{NLL}) + (Loss\_{value} \\times \\text{MSE})$$

🛠️ Installation
----------------

1.  Bashgit clone https://github.com/yourusername/chess-ai-pytorch.gitcd chess-ai-pytorch
    
2.  You need PyTorch, python-chess, and numpy.Bashpip install torch numpy python-chess tqdm
    
3.  Download the Stockfish Engine and place the executable in the project directory (required for data generation).
    

📊 Workflow
-----------

### 1\. Data Generation (Preprocessing)

Convert raw PGN game files into a format the model can understand (Bitboards, Policy Indices, Score Values). This script uses **multiprocessing** to speed up analysis.

```bash
python ai/preprocess.py
```

*   _Input:_ data.pgn (PGN file with games)
    
*   _Output:_ alphazero\_data.npz (Compressed NumPy arrays)
    

### 2\. Training

Train the Dual-Headed ResNet model. The script automatically saves the best model based on validation loss.

```bash
python ai/train.py
```

*   _Configuration:_ You can adjust BATCH\_SIZE, EPOCHS, and LEARNING\_RATE inside the script.
    
*   _Output:_ saved\_model/alphazero\_chess.pth
    

### 3\. Playing

Run the main game interface to play against the AI. The AI uses a "Mirror Board" to track the game state and masks illegal moves during inference.

```bash
python src/chess/main_window.py
```

📂 Project Structure
--------------------

```text
.  ├── ai/  │   ├── preprocess_alphazero.py # Multiprocessed data generator  │   ├── train.py                # Dual-headed training loop  │   ├── model.py                # PyTorch ResNet Architecture  │   ├── dataset.py              # Custom PyTorch Dataset Loader  │   ├── ai.py                   # Inference wrapper & Board state manager  │   └── ai_player.py            # Move prediction logic  ├── src/  │   └── chess/                  # Game GUI and Logic (PyQt/Custom)  ├── models/                # Directory for trained weights (.pth)  └── README.md
```

📈 Results
----------

Even with a small dataset (~3,000 games / 200k positions) and 7 epochs of training, the model demonstrates:

*   **Opening Theory:** It recognizes standard openings (e.g., e4, d4, Nf3).
    
*   **Tactical Awareness:** It avoids obvious blunders by filtering illegal moves and prioritizing high-probability actions learned from Stockfish.
    
*   **Positional Understanding:** The Value Head successfully differentiates between winning and losing positions.
    

🤝 Credits
----------

*   **Engine:** Logic inspired by [AlphaZero](https://www.google.com/search?q=https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go).
    
*   **Libraries:** Built with [PyTorch](https://pytorch.org/) and [python-chess](https://python-chess.readthedocs.io/).
    
*   **Teacher:** Supervised learning targets generated by [Stockfish](https://stockfishchess.org/).