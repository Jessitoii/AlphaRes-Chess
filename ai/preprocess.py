import chess
import chess.pgn
import chess.engine
import numpy as np
import os
import multiprocessing
from tqdm import tqdm

# --- AYARLAR ---
PGN_PATH = "ai/data/data2.pgn"
ENGINE_PATH = r"D:\Software\Python\ChessApp\ChessApp\stockfish\stockfish-windows-x86-64-avx2.exe"
OUTPUT_DIR = "ai/processed_data"
NUM_GAMES = 20000 
CHUNK_SIZE = 100

def encode_move(move):
    """UCI hamlesini 0-4095 arası bir indekse çevirir."""
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.int8)
    piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel = piece_map[piece.symbol()]
        tensor[channel, row, col] = 1
    return tensor

def process_game_batch(games_pgn):
    # Her process kendi motorunu açar
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    
    batch_x = []       
    batch_policy = []  
    batch_value = []   
    
    for pgn_text in games_pgn:
        import io
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None: continue
        
        board = game.board()
        
        for move in game.mainline_moves():
            # 1. Giriş Verisi (X)
            batch_x.append(board_to_tensor(board))
            
            # 2. Stockfish Analizi
            # analyse fonksiyonu LİSTE döndürür! Fix burada:
            info_list = engine.analyse(board, chess.engine.Limit(depth=10), multipv=1)
            
            if not info_list: # Eğer boş dönerse (çok nadir)
                batch_x.pop()
                board.push(move)
                continue

            info = info_list[0] # Listenin ilk elemanını (sözlüğü) alıyoruz!

            # A. Policy (En iyi hamle hangisi?)
            if "pv" in info and len(info["pv"]) > 0:
                best_move = info["pv"][0]
                batch_policy.append(encode_move(best_move))
            else:
                batch_x.pop()
                board.push(move) # Hamleyi yapıp devam et, ama bu pozisyonu kaydetme
                continue

            # B. Value (Kim kazanıyor?)
            score = info["score"].relative.score(mate_score=10000)
            norm_score = np.tanh(score / 1000.0)
            
            # Bakış açısını ayarla (Beyaz'a göre skor)
            if board.turn == chess.BLACK:
                norm_score = -norm_score
            
            batch_value.append(norm_score)
            
            # Hamleyi yap ve devam et
            board.push(move)

    engine.quit()
    return batch_x, batch_policy, batch_value

def run_alphazero_preprocessing():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    num_cores = multiprocessing.cpu_count()
    workers = max(1, num_cores - 2) 
    print(f"{workers} çekirdek ile AlphaZero verisi hazırlanıyor...")

    all_x = []
    all_p = []
    all_v = []

    with open(PGN_PATH) as pgn:
        with multiprocessing.Pool(processes=workers) as pool:
            pbar = tqdm(total=NUM_GAMES, desc="İşleniyor")
            games_processed = 0
            
            while games_processed < NUM_GAMES:
                batch = []
                for _ in range(workers * 4): 
                    game = chess.pgn.read_game(pgn)
                    if game:
                        batch.append(str(game))
                    else:
                        break 
                
                if not batch: break

                chunked_batch = [batch[i::workers] for i in range(workers)]
                results = pool.map(process_game_batch, chunked_batch)

                for x, p, v in results:
                    all_x.extend(x)
                    all_p.extend(p)
                    all_v.extend(v)
                
                count = len(batch)
                games_processed += count
                pbar.update(count)

            pbar.close()

    print("\nVeriler kaydediliyor...")
    np.savez_compressed(
        f"{OUTPUT_DIR}/alphazero_data.npz", 
        x=np.array(all_x, dtype=np.int8), 
        y_policy=np.array(all_p, dtype=np.int16),
        y_value=np.array(all_v, dtype=np.float32)
    )
    print(f"İşlem Tamam! Toplam {len(all_x)} pozisyon kaydedildi.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_alphazero_preprocessing()