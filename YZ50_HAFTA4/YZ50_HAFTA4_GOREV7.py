import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
# Veri Setini Hazırlama
words = open('turkceisimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower() for w in words]

#chars: Veri setindeki tüm benzersiz harflerin alfabetik olarak dizilmiş listesidir
chars = sorted(list(set(''.join(words)))) 
stoi = {s: i+1 for i, s in enumerate(chars)} # {'a':1, 'b':2, 'c':3, ...} örn stoi['c'] = 3
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

vocab_size = len(stoi) #böylece hangi alfabe oolursaa ollsun otomatik olarak harf sayısını bulur
block_size = 3 # Veriyi üçe bölme train dev set 

#Önce 3 lü kümeleri yanii X matrislerini hazırlayacağız ve sonuç harfi olan .em ---> a   olan Y matrislini hazırlayacağız
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0, 0, 0]
        for ch in w + '.': # pythonda stringler zaten karakter dizisi olarak kabul edilir ve direkt o kelimenin içindekii karakterlerde dönebilir
            ix = stoi[ch] # üçlü küümeler harflerle değil innteger karşılıklarıyla tutulur 
            X.append(context) # append[] değil append() kullanılır
            Y.append(ix) #loss hesaplamada kullanacağız
            context = context[1:] + [ix] # ix bir sayı olduğu için listeye çevirip eklemeliyiz
    
    return torch.tensor(X), torch.tensor(Y) 

#train dev set hazırlayalım
random.seed(42)
random.shuffle(words)

n1 = int(len(words) * 0.8) #  kelime sayısının %80'ini alıyoruz
n2 = int(len(words) * 0.9)

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])

# Matrisleri Hazırlama
# ==========================================
C = torch.randn((vocab_size, 10)) # her harfe random 10 konum atanduı

girdi_sayisi = 30 
kaiming_olcegi = (5 / 3) / (girdi_sayisi ** 0.5) 
W1 = torch.randn((30, 200)) * kaiming_olcegi

# b1'i tamamen sildik!
# BOŞLUK 1: Gamma matrisi (Çarpma işlemi). Başlangıçta 1'lerle (etkisiz) dolu olmalı.
bngain = torch.ones((1, 200)) 
# BOŞLUK 2: Beta matrisi (Toplama işlemi). Başlangıçta 0'larla (etkisiz) dolu olmalı.
bnbias = torch.zeros((1, 200)) 
# BOŞLUK 3: Running Mean (Hareketli Ortalama). Başlangıçta 0'larla dolu olmalı.
bnmean_running = torch.zeros((1, 200)) 
# BOŞLUK 4: Running Std (Hareketli Standart Sapma). Başlangıçta 1'lerle dolu olmalı (Sıfıra bölme hatası vermemesi için).
bnstd_running = torch.ones((1, 200)) 

W2 = torch.randn((200, vocab_size)) * 0.01 #vocabsize alfabedeki harf sayısı
b2 = torch.zeros(vocab_size) 

# b1'i çıkarıp Gamma ve Beta'yı ekledik
parameters = [C, W1, W2, b2, bngain, bnbias] 
for p in parameters:
    p.requires_grad = True



#Training yani eğitim kısmı (Önce modeli eğitmeliyiz ki sonra Dev Set ile test edelim)
# ==========================================
#hazır üçlükümeler barındıran X ile kktmanlatı eğitip hazırlayacağız yanii proramın öğrenmesini sağlayacağız
for i in range(20000):
    # Minibatch
    ix = torch.randint(0, Xtr.shape[0], (32,))
    hpreact = C[Xtr[ix]].view(-1, 30) @ W1  # w1 değil W1 olmalı. boyutu (32,3,10) idi ama düzleştirdik
    
    # O anki batch'in ortalama ve sapması
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)

    #BatchN formülü ile tanh ye sokmadsn önce
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

    #Hareketli ortallama ve hareketli sapma oluşturalım
    with torch.no_grad(): # İstatistikleri güncellerken gradyan takibini kapatmalıyız
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi 

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2 # w2 değil W2 olmalı

    loss = F.cross_entropy(logits, Ytr[ix]) #seçilen 32 satırın sonnuç matrisi y yi alır ve son haö sonuçlar matrisi logitsi alır loss hesaplar

    #backward ile gard bulunur ve datalar güncellenir
    for p in parameters:
        p.grad = None #önceki grad bilgilerini temizliyoruz 
    
    loss.backward() # backward for döngüsünün dışında tek bir kez çalışmalı

    lr = 0.1 if i < 10000 else 0.01 
    for p in parameters:
        p.data += -lr * p.grad


# 5. DEV SETİ İLE TEST (INFERENCE)
# ==========================================
with torch.no_grad():
    emb = C[Xdev] 
    hpreact = emb.view(-1, 30) @ W1 
    
    # BOŞLUK 8: Test aşamasındayız! Artık o anki batch'i değil, hazır koşan istatistikleri kullanmalısın!
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias #Burada artık bnmean yok eğitimden gelen bnmean_running ve bnstd_running var!!!!!!!
    
    h = torch.tanh(hpreact) 
    logits = h @ W2 + b2 
    dev_loss = F.cross_entropy(logits, Ydev)
    
print(f"Dev Set Loss Puanı: {dev_loss.item():.4f}")


#Yeni isim üretme 
# ==========================================
print("\n--- Modelin Ürettiği İsimler ---")
for _ in range(20):
    out = []
    context = [0, 0, 0]

    while True:
        emb = C[torch.tensor([context])]
        hpreact = emb.view(1, 30) @ W1

        #isim üretirken running istatistikleri kullanılır
        hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias

        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        
        if ix == 0:
            break
            
        out.append(itos[ix])
        context = context[1:] + [ix] 
        
    print(''.join(out))