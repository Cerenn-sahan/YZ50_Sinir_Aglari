import torch.nn.functional as F
import torch

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}




xs, ys = [],[]
for w in words:
    chs =['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]): #zip ikili listeler oluşturuyor
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs) # Girdiler (Örn: 0, 5, 13...)
ys = torch.tensor(ys) # Beklenen Çıktılar (Örn: 5, 13, 13...)

# Girdi sayımızı (num) belirliyoruz ki aşağıda loss hesaplarken torch.arange(num) hata vermesin
num = xs.nelement()

# xs listesindeki tüm sayıları One-Hot vektörlere çevir çünkü e nin 5 m nin 13 olması büyüklük ilişkisi değil stuationdır yani 0,0,0,0,0,0,0,0,0,1,0,0,0,0 gibi bir vektör olacak
xenc = F.one_hot(xs, num_classes=27).float()

# 27x27 boyutunda, içi tamamen rastgele sayılarla dolu bir Ağırlık (Weight) matrisi
W = torch.randn((27, 27), requires_grad=True)

# girdileri ağırlıklarla çarpıyoruz
logits = xenc @ W

counts = logits.exp() # logits içindeki sayıları pozitif hale getiriyoruz 
probs = counts / counts.sum(1, keepdims=True) # Yüzdeliğe (olasılıklara) dönüştü yani ağırlığının olaslıklarını buldu (örn e yani 5 için 5. satırı çekersin ağırlık matrisinden
# ve onun ağırlık puanlarını direkt alıp ys ile karşılaştırma yapmak yerine önce olasılığa dönüştüürüsün

loss = -probs[torch.arange(num), ys].log().mean()

W.grad = None   # Önceki hesaplamaların hafızasını sıfırla 
loss.backward() # Geriye yayılımı başlat

W.data += -50.0 * W.grad

# 100 kere eğitim yap (Gradient Descent Döngüsü)
for k in range(100):
    
    # 1. İleri Yönlü Çalışma (Forward Pass)
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    
    # 2. Hata Hesaplama (Loss)
    loss = -probs[torch.arange(num), ys].log().mean()
    print(loss.item()) # Her adımda hatanın düştüğünü gör
    
    # 3. Geriye Yayılım (Backward Pass)
    W.grad = None
    loss.backward()
    
    # 4. Ağırlıkları Güncelle (Öğrenme)
    W.data += -50.0 * W.grad