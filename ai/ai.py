import torch
import numpy as np
import chess
from .model import AlphaZeroNet

class AlphaZeroPlayer:
    def __init__(self, model_path="models/alphazero_chess.pth", device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device:
            self.device = device
            
        print(f"AI: {self.device} üzerinde başlatılıyor...")
        
        # Modeli Yükle
        self.model = AlphaZeroNet().to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"AI: Model '{model_path}' başarıyla yüklendi.")
        except FileNotFoundError:
            print("HATA: Model dosyası bulunamadı! Rastgele ağırlıklarla çalışıyor.")
        
        self.model.eval() # Inference modu (Dropout/BatchNorm'u dondurur)

    def board_to_tensor(self, board):
        """Tahtayı modelin anlayacağı 12x8x8 tensöre çevirir."""
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                     "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
        
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            channel = piece_map[piece.symbol()]
            tensor[channel, row, col] = 1.0
            
        return torch.from_numpy(tensor).unsqueeze(0) # (1, 12, 8, 8) Batch boyutu ekle

    def get_best_move(self, board):
        """Mevcut tahta için en iyi hamleyi seçer."""
        
        # 1. Yasal Hamleleri Al
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None # Mat veya Pat durumu

        # 2. Modeli Çalıştır
        tensor = self.board_to_tensor(board).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.model(tensor)
        
        # 3. Sonuçları İşle
        # Logits -> Olasılıklar (Gerekirse exp alabilirsin ama sıralama değişmez)
        policy_logits = policy_logits.squeeze().cpu().numpy() # (4096,)
        current_value = value.item() # -1 (Siyah) ile 1 (Beyaz) arası
        
        # --- MASKELEME (MASKING) ---
        # Sadece yasal hamlelerin skorlarına bakacağız.
        # Modelin "Vezirle şah çek" dediği ama vezirin arada kaldığı durumları eliyoruz.
        
        best_move = None
        best_score = -float('inf')
        
        # Debug için olasılıkları görelim
        move_scores = []

        for move in legal_moves:
            # Hamleyi index'e çevir (Preprocessing ile aynı mantık!)
            from_sq = move.from_square
            to_sq = move.to_square
            idx = from_sq * 64 + to_sq
            
            # Modelin bu index için verdiği puanı al
            score = policy_logits[idx]
            
            move_scores.append((move.uci(), score))
            
            if score > best_score:
                best_score = score
                best_move = move
        
        # Konsola Bilgi Bas (Debug)
        print(f"AI Değerlendirmesi (Beyaz Gözünden): {current_value:.4f}")
        # En yüksek puanlı 3 hamleyi göster
        move_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"Aday Hamleler: {move_scores[:3]}")
        print(f"Seçilen Hamle: {best_move.uci()}")

        return best_move

class Computer:
    def __init__(self):
        # 1. Arka planda resmi bir satranç tahtası tutuyoruz
        self.mirror_board = chess.Board()
        
        # 2. Eğittiğimiz AlphaZero modelini yüklüyoruz
        self.ai = AlphaZeroPlayer(model_path="saved_model/alphazero_chess.pth")

    def predict_best_move(self):
        """
        AI, kendi içindeki ayna tahtaya bakarak hamle seçer.
        Dışarıdan parametre almasına gerek yoktur.
        """
        # AlphaZeroPlayer'ın beklediği chess.Board objesini veriyoruz
        best_move = self.ai.get_best_move(self.mirror_board)
        return best_move # chess.Move objesi döner

    def pushMove(self, uci_move_string):
        """
        Oyunda yapılan hamleleri (hem senin hem AI'ın) buraya işlemeliyiz
        ki AI tahtanın son halini bilsin.
        """
        try:
            move = chess.Move.from_uci(uci_move_string)
            if move in self.mirror_board.legal_moves:
                self.mirror_board.push(move)
            else:
                print(f"HATA: AI senkronizasyonu bozuldu! Geçersiz hamle: {uci_move_string}")
                print("AI Tahtası:\n", self.mirror_board)
        except Exception as e:
            print(f"Hamle işlenirken hata: {e}")

# --- TEST KODU ---
if __name__ == "__main__":
    ai = AlphaZeroPlayer()
    board = chess.Board() # Başlangıç pozisyonu
    
    # AI hamle yapsın
    move = ai.get_best_move(board)
    print("Yapılan Hamle:", move)