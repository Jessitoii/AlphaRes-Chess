import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Kendi dosyalarımızdan import ediyoruz
from model import AlphaZeroNet
from dataset import AlphaZeroDataset

# --- AYARLAR ---
NPZ_FILE = "ai/processed_data/alphazero_data.npz"
BATCH_SIZE = 64        # GPU belleğin yetmezse 32 yap
EPOCHS = 50            # AlphaZero uzun sürer, sabırlı ol
LEARNING_RATE = 1e-3   # Adam için ideal başlangıç

def train_alphazero():
    # 1. Cihaz Ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim {device} üzerinde başlayacak...")

    # 2. Veri Seti ve Loader
    full_dataset = AlphaZeroDataset(NPZ_FILE)
    
    # %90 Eğitim, %10 Doğrulama
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model Başlatma
    model = AlphaZeroNet().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler: Loss düşmezse learning rate'i kıs (Daha hassas ayar için)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # 4. Kayıp Fonksiyonları (Loss Functions)
    # Policy Head için: Negative Log Likelihood (Çünkü modelde LogSoftmax kullandık)
    policy_criterion = nn.NLLLoss()
    
    # Value Head için: Mean Squared Error (Çünkü -1 ile 1 arası sayı tahmin ediyoruz)
    value_criterion = nn.MSELoss()

    # --- EĞİTİM DÖNGÜSÜ ---
    best_total_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_policy_loss = 0
        total_value_loss = 0
        total_loss_accum = 0

        for batch_idx, (boards, target_policies, target_values) in enumerate(train_loader):
            boards = boards.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad()

            # Forward Pass (Model iki şey döndürür: p ve v)
            pred_policies, pred_values = model(boards)

            # Kayıpları Hesapla
            # 1. Policy Loss: Modelin hamle tahmini ne kadar yanlış?
            loss_policy = policy_criterion(pred_policies, target_policies)
            
            # 2. Value Loss: Modelin skor tahmini ne kadar yanlış?
            loss_value = value_criterion(pred_values, target_values)

            # 3. Toplam Kayıp (Total Loss)
            # İkisini toplayarak backprop yapıyoruz. Böylece model ikisini de öğreniyor.
            loss = loss_policy + loss_value

            loss.backward()
            optimizer.step()

            # İstatistik
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()
            total_loss_accum += loss.item()

        # Ortalamalar
        avg_policy_loss = total_policy_loss / len(train_loader)
        avg_value_loss = total_value_loss / len(train_loader)
        avg_train_loss = total_loss_accum / len(train_loader)

        # --- VALIDATION (DOĞRULAMA) ---
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for boards, target_policies, target_values in val_loader:
                boards = boards.to(device)
                target_policies = target_policies.to(device)
                target_values = target_values.to(device)

                p, v = model(boards)
                l_p = policy_criterion(p, target_policies)
                l_v = value_criterion(v, target_values)
                val_loss_accum += (l_p + l_v).item()

        avg_val_loss = val_loss_accum / len(val_loader)

        # --- RAPORLAMA ---
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f} (Policy: {avg_policy_loss:.4f} + Value: {avg_value_loss:.4f})")
        print(f"Val Loss  : {avg_val_loss:.4f}")

        # Learning Rate güncelleme
        scheduler.step(avg_val_loss)

        # En iyi modeli kaydet
        if avg_val_loss < best_total_loss:
            best_total_loss = avg_val_loss
            if not os.path.exists("saved_model"):
                os.makedirs("saved_model")
            torch.save(model.state_dict(), "saved_model/alphazero_chess.pth")
            print("--> Model Kaydedildi! (Yeni en iyi skor)")

if __name__ == "__main__":
    train_alphazero()