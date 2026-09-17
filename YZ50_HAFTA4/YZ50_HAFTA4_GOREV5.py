import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. VERİ SETİNİ OKUMA VE SÖZLÜK OLUŞTURMA
# ==========================================
words = open('names.txt', 'r').read().splitlines()
words = [w.lower() for w in words]

chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

vocab_size = len(stoi) # 27
block_size = 3 # 3 harfe bakarak tahminde bulunacağız

# ==========================================
# 2. VERİYİ ÜÇE BÖLME (TRAIN, DEV, TEST)
# ==========================================
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])   # %80 Eğitim Verisi
Xdev, Ydev = build_dataset(words[n1:n2]) # %10 Geliştirme (Test) Verisi
Xte, Yte = build_dataset(words[n2:])   # %10 Final Test Verisi

print(f"Eğitim Seti: {Xtr.shape}")
print(f"Dev Seti: {Xdev.shape}")

# ==========================================
# 3. MATRİSLERİ OLUŞTURMA VE OPTİMİZASYON (KAIMING)
# ==========================================
# Harfleri 10 boyutlu uzaya taşıyoruz
C = torch.randn((vocab_size, 10)) 

# Gizli Katman (W1, b1) - Tanh Doymasını Engellemek İçin Kaiming Init
girdi_sayisi = 30 # (3 harf * 10 boyut)
kaiming_olcegi = (5 / 3) / (girdi_sayisi ** 0.5) 

W1 = torch.randn((30, 200)) * kaiming_olcegi
b1 = torch.randn(200) * 0.01 

# Çıkış Katmanı (W2, b2) - Yüksek Başlangıç Loss'unu Engellemek İçin Sıfırlama
W2 = torch.randn((200, vocab_size)) * 0.01
b2 = torch.zeros(vocab_size) 

parameters = [C, W1, b1, W2, b2]

# Türev (Gradient) takibini aktif ediyoruz
for p in parameters:
    p.requires_grad = True

# Toplam parametre sayısını görelim
toplam_param = sum(p.nelement() for p in parameters)
print(f"Toplam Öğrenilebilir Parametre (Ağırlık) Sayısı: {toplam_param}")

# ==========================================
# 4. EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# ==========================================
# Toplam 20.000 adım atacağız.
for i in range(20000):
    
    # 1. Minibatch: Eğitim setinden (Xtr) rastgele 32 satır seç
    ix = torch.randint(0, Xtr.shape[0], (32,))
    
    # 2. İleri Yönlü Hesaplama (Forward Pass)
    emb = C[Xtr[ix]] # (32, 3, 10)
    hpreact = emb.view(-1, 30) @ W1 + b1 # Tanh öncesi ham matris
    h = torch.tanh(hpreact) # (32, 200)
    logits = h @ W2 + b2 # (32, 27)
    
    # Loss Hesaplama
    loss = F.cross_entropy(logits, Ytr[ix])
    
    # 3. Geriye Yayılım (Backward Pass)
    for p in parameters:
        p.grad = None
        
    loss.backward()
    
    # 4. Parametre Güncelleme (Gradient Descent)
    # İlk 10.000 adımda büyük (0.1), sonraki adımlarda küçük (0.01) adımlarla ilerle
    lr = 0.1 if i < 10000 else 0.01 
    
    for p in parameters:
        p.data += -lr * p.grad

print(f"Eğitim tamamlandı. Son Minibatch Loss Puanı: {loss.item():.4f}")

# ==========================================
# 5. GELİŞTİRME (DEV) SETİNDE PERFORMANS ÖLÇÜMÜ
# ==========================================
# Burada ağırlıklar sabittir, türev alınmaz
emb = C[Xdev] 
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) 
logits = h @ W2 + b2 
dev_loss = F.cross_entropy(logits, Ydev)

print(f"Hiç Görülmeyen Veride (Dev Set) Loss Puanı: {dev_loss.item():.4f}")

# ==========================================
# 6. YENİ İSİMLER ÜRETME (SAMPLING)
# ==========================================
print("\n--- Modelin Ürettiği Yeni İsimler ---")
for _ in range(20):
    out = []
    context = [0, 0, 0] # Başlangıç: "..."
    
    while True:
        emb = C[torch.tensor([context])] # (1, 3, 10)
        h = torch.tanh(emb.view(1, 30) @ W1 + b1)
        logits = h @ W2 + b2
        
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        
        if ix == 0:
            break
            
        out.append(itos[ix])
        context = context[1:] + [ix] # Kaydırma
        
    print(''.join(out))
