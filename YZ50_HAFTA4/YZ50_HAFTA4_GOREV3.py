import torch
import torch.nn.functional as F
import random # Veriyi karıştırmak (shuffle) için eklendi

# ==========================================
# 1. VERİ SETİNİ OKUMA VE SÖZLÜK OLUŞTURMA
# ==========================================
words = open('names.txt', 'r').read().splitlines()
words = [w.lower() for w in words]

chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi) 
block_size = 3 

# ==========================================
# 2. VERİYİ ÜÇE BÖLME (TRAIN, DEV, TEST)
# ==========================================

# X ve Y matrislerini oluşturan işlemi bir fonksiyona çeviriyoruz.
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

# Modeli eğitmeden önce kelime listesini rastgele karıştırıyoruz
random.seed(42)
random.shuffle(words)

# Toplam kelime sayısının %80'ini ve %90'ını buluyoruz
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

# %80 Eğitim (Train) Verisi: Ağırlıkları güncellemek için kullanılacak
Xtr, Ytr = build_dataset(words[:n1])

# %10 Geliştirme (Dev/Validation) Verisi: Ağırlıklar güncellenmez, sadece loss ölçülür
Xdev, Ydev = build_dataset(words[n1:n2])

# %10 Test Verisi: Sistemin en son performansını ölçmek için saklanır
Xte, Yte = build_dataset(words[n2:])

print(f"Eğitim Seti Boyutu: {Xtr.shape}")
print(f"Geliştirme Seti Boyutu: {Xdev.shape}")
print(f"Test Seti Boyutu: {Xte.shape}")

# ==========================================
# 3. PARAMETRELERİ (MATRİSLERİ) İLKLEME
# ==========================================
C = torch.randn((vocab_size, 2))
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
W2 = torch.randn((100, vocab_size))
b2 = torch.randn(vocab_size)

parameters = [C, W1, b1, W2, b2]

# Tüm matrislerin "türev hesaplamasına" dahil olması gerektiğini PyTorch'a bildiriyoruz
for p in parameters:
    p.requires_grad = True

# ==========================================
# 4. EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# ==========================================
# Optimizasyon işlemini 10.000 adım boyunca tekrar ediyoruz
for i in range(10000):
    
    # 1. Minibatch: Eğitim setinden (Xtr) rastgele 32 satır seçiyoruz
    ix = torch.randint(0, Xtr.shape[0], (32,))
    
    # 2. İleri Yönlü Hesaplama (Forward Pass)
    emb = C[Xtr[ix]] # (32, 3, 2)
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
    logits = h @ W2 + b2 # (32, 27)
    
    # Hata hesaplama (Sadece minibatch'teki 32 satır için)
    loss = F.cross_entropy(logits, Ytr[ix])
    
    # 3. Geriye Yayılım (Backward Pass)
    # Bir önceki döngüden kalan türevleri (grad) sıfırlıyoruz ki birbirine karışmasın
    for p in parameters:
        p.grad = None
        
    # Her bir parametre için türev değerlerini hesaplıyoruz
    loss.backward()
    
    # 4. Parametreleri Güncelleme (Gradient Descent)
    lr = 0.1 # Öğrenme oranı
    for p in parameters:
        # Mevcut veriyi (p.data), türev yönünde (p.grad) değiştiriyoruz
        p.data += -lr * p.grad

print(f"Eğitim tamamlandı. Son Minibatch Loss Puanı: {loss.item():.4f}")

# ==========================================
# 5. GELİŞTİRME SETİNDE (DEV SET) PERFORMANS ÖLÇÜMÜ
# ==========================================
# Burada `backward()` veya `p.data +=` işlemleri YOKTUR. Ağırlıklar sabittir.
# Bütün Geliştirme matrisini (Xdev) aynı anda ileri yönlü hesaplamaya sokuyoruz.

emb = C[Xdev] 
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) 
logits = h @ W2 + b2 
dev_loss = F.cross_entropy(logits, Ydev)

print(f"Hiç Görülmeyen Veride (Dev Set) Loss Puanı: {dev_loss.item():.4f}")
