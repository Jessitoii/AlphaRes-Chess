import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    ResNet'in yapı taşı. Bilginin kaybolmadan derinlere inmesini sağlar.
    Giriş -> Conv -> BN -> ReLU -> Conv -> BN -> + Giriş -> ReLU
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip Connection (Sihir burada)
        return F.relu(out)

class AlphaZeroNet(nn.Module):
    def __init__(self, num_res_blocks=5, num_channels=256):
        super(AlphaZeroNet, self).__init__()
        
        # --- GÖVDE (BACKBONE) ---
        # Giriş: 12 kanal (Taşlar) -> 256 kanala genişlet
        self.conv_input = nn.Conv2d(12, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual Bloklar (Derinlik)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # --- KAFA 1: POLICY HEAD (Hamle Seçici) ---
        # Çıktı: 4096 (64 kare * 64 kare) olasılık dağılımı
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096) 

        # --- KAFA 2: VALUE HEAD (Değerlendirici) ---
        # Çıktı: -1 ile 1 arası tek bir skaler değer
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Gövdeyi çalıştır
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy Head (Hamleler)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # Flatten
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1) # Olasılıklar için LogSoftmax (CrossEntropyLoss ile çalışır)

        # Value Head (Skor)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1) # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # -1 ile 1 arasına sıkıştır

        return p, v