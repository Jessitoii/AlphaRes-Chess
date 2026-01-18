import torch
from torch.utils.data import Dataset
import numpy as np
import os

class AlphaZeroDataset(Dataset):
    def __init__(self, npz_path):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {npz_path}")
            
        print(f"Veri yükleniyor: {npz_path} ...")
        data = np.load(npz_path)
        
        # Verileri belleğe al
        self.x = data['x']          # Giriş: (N, 12, 8, 8) int8
        self.p = data['y_policy']   # Policy Hedef: (N,) int16 (Sınıf İndeksi)
        self.v = data['y_value']    # Value Hedef: (N,) float32 (Skor)
        
        print(f"AlphaZero Dataset Hazır! Toplam Pozisyon: {len(self.x)}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 1. Giriş Verisi (Board)
        # Model float32 bekler. int8'den çeviriyoruz.
        board = torch.from_numpy(self.x[idx]).float()
        
        # 2. Policy Hedefi (Best Move Index)
        # CrossEntropyLoss için hedef veri tipi 'Long' (int64) olmak zorundadır.
        policy_target = torch.tensor(self.p[idx], dtype=torch.long)
        
        # 3. Value Hedefi (Score)
        # MSELoss için hedef veri tipi 'Float32' olmalıdır.
        value_target = torch.tensor([self.v[idx]], dtype=torch.float32)
        
        return board, policy_target, value_target