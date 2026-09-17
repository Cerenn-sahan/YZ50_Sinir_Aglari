import torch
import torch.nn.functional as F

# ==========================================
# 0. BÖLÜM: VERİ SETİ VE SÖZLÜK HAZIRLIĞI
# ==========================================
words = open('names.txt', 'r').read().splitlines()
words = [w.lower() for w in words]

chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

vocab_size = len(stoi) 
print(f"Alfabemizdeki harf sayısı: {vocab_size}")

# ==========================================
# 0.1. BÖLÜM: 3 HARFLİK BAĞLAM (X VE Y MATRİSLERİ)
# ==========================================
block_size = 3 # 3 harflik hafıza (kayan pencere)
X, Y = [], []

for w in words:
    context = [0] * block_size # Kelime başına geldiğimizde hafızayı [0, 0, 0] yapıyoruz
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context) # Girdiyi X'e ekle
        Y.append(ix)      # Hedefi Y'ye ekle
        
        # Pencereyi kaydır: En baştaki harfi at, yeni harfi sona ekle
        context = context[1:] + [ix]

# Listeleri PyTorch Tensörüne çeviriyoruz
X = torch.tensor(X)
Y = torch.tensor(Y)
print(f"X matrisinin boyutu (Girdiler): {X.shape}")
print(f"Y matrisinin boyutu (Hedefler): {Y.shape}")

# ==========================================
# 1. BEYNİ (AĞIRLIKLARI) İNŞA ETME
# ==========================================

# C: Embedding matrisi (Her harf için 2 boyutlu koordinat)
C = torch.randn((vocab_size, 2), requires_grad=True)

# Gizli Katman (W1, b1): 3 harf * 2 koordinat = 6 girdi -> 100 dedektif nöron
W1 = torch.randn((6, 100), requires_grad=True)
b1 = torch.randn(100, requires_grad=True)

# Çıkış Katmanı (W2, b2): 100 dedektifin fikri -> 27 harflik alfabe seçimi
W2 = torch.randn((100, vocab_size), requires_grad=True)
b2 = torch.randn(vocab_size, requires_grad=True)

# Tüm parametreleri topluyoruz
parameters = [C, W1, b1, W2, b2]
print(f"Ağın toplam parametre (öğrenilebilir sayı) adedi: {sum(p.nelement() for p in parameters)}")

# ==========================================
# 2. İLERİ YÖNLÜ ÇALIŞMA (FORWARD PASS)
# ==========================================

# 1. Adım: X'teki her harfi 2 boyutlu uzaya çevir (Embedding)
emb = C[X] 

# 2. Adım: emb matrisini düz bir şeride çevirip (view), W1 ile çarpıp tanh'den geçiriyoruz.
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) 

# 3. Adım: 100 fikri harf olasılıklarına (logits) çeviriyoruz.
logits = h @ W2 + b2 

# ==========================================
# 3. HATA HESAPLAMA (LOSS)
# ==========================================

# PyTorch'un en güvenli ve hızlı fonksiyonu ile hatayı hesaplıyoruz
loss = F.cross_entropy(logits, Y)
print(f"İlk rastgele tahminin Loss Puanı: {loss.item():.4f}")