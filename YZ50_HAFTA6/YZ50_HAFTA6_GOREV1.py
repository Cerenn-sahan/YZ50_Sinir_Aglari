import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. VERİ HAZIRLIĞI
# ==========================================
words = open('turkceisimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower() for w in words]
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

block_size = 3 # Her tahmin için kullanılacak geçmiş harf sayısı

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
n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# ==========================================
# 2. KATMAN (LAYER) SINIFLARI (LEGO PARÇALARI)
# ==========================================
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # Kaiming init
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Öğrenilebilir parametreler
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # Hareketli ortalamalar (Eğitim dışı kullanım için)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        # Hareketli ortalamaları güncelle
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]

class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
  
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
  
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# ==========================================
# 3. MODELİ İNŞA ETME
# ==========================================
n_embd = 10
n_hidden = 200

# Görev 1: MLP + BatchNorm kodunu sınıflara topla ve model tek bir Sequential olsun[cite: 2]
model = Sequential([
  Embedding(vocab_size, n_embd),
  Flatten(),
  Linear(n_embd * block_size, n_hidden, bias=False),
  BatchNorm1d(n_hidden),
  Tanh(),
  Linear(n_hidden, vocab_size),
])

# Son katmanın aşırı güvenli (confident) başlamaması için ağırlıklarını küçültüyoruz
with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters = model.parameters()
print(f"Toplam Parametre Sayısı: {sum(p.nelement() for p in parameters)}")

for p in parameters:
    p.requires_grad = True

# ==========================================
# 4. EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# ==========================================
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # Batch oluştur
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]
    
    # İleri Yayılım (Forward Pass)
    # Eğitim döngüsü içindeki katmanların adını bilmesin kuralı[cite: 2]
    logits = model(Xb) #işte bu kadar modeli çağırmamız yeterli o bize logits i direkt verecek (model de kendi içinde ilgili class çağırımı yaparak çabucak bitiyor)
    loss = F.cross_entropy(logits, Yb)
    
    # Geri Yayılım (Backward Pass)
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # Parametre Güncelleme (Update)
    lr = 0.1 if i < 100000 else 0.01 # Learning rate decay
    for p in parameters:
        p.data -= lr * p.grad
    
    # İstatistikleri Takip Et
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.item())

# ==========================================
# 5. LOSS EĞRİSİ ÇİZİMİ
# ==========================================
# Loss eğrisi çizimini videodaki gibi düzelt kuralı[cite: 2]
loss_tensor = torch.tensor(lossi)
plt.plot(loss_tensor.view(-1, 1000).mean(1))
plt.title("Eğitim Kaybı (1000 iterasyonluk hareketli ortalama)")
plt.xlabel("Adım (x1000)")
plt.ylabel("Loss")
plt.show()

# ==========================================
# 6. MODELİ TEST ETME (EVALUATION)
# ==========================================
# Test aşamasına geçmeden önce BatchNorm katmanlarının eğitim modunu kapatıyoruz
for layer in model.layers:
    layer.training = False

# Validation (Geliştirme) seti üzerinde performansı ölç
@torch.no_grad() # Test yaparken gradyan hesaplamayı kapat
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(f'{split} loss: {loss.item():.4f}')

split_loss('train')
split_loss('val')
