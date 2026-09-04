import torch
import torch.nn.functional as F

#VERİYİ HAZIRLAMA
words = open('turkceisimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower() for w in words] #Bütün kelimeleri küçük harfe çevir

chars = sorted(list(set(''.join(words)))) #her bir boşluğa göre toplam karakter sayısı
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


vocab_size = len(stoi) #alfabedeki toplam harf sayısı

#İKİLİLERİN OLASILIK MATRİSİ (Biagram Counting)
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32) #vocabsize*vocabsize boş matriis
for w in words:
    chs = ['.'] + list(w) + ['.'] #kelimelerin başına ve sonuna nokta koyuyoruz ve kelimeyi harflerine ayırıyoruz yani    . c e r e n.  
    for ch1,ch2 in zip(chs, chs[1:]): # chs leri ch1 ve ch2 olacak şekilde soldan sağa ikili listelere ayırdık
        N[stoi[ch1], stoi[ch2]] +=1 #string to integer yaptık çünkü e örneğin 5 e denk gelir ve bu aynı zamanda onun satır numarası olur diğeri de sütun numarası kesişime += 1 

P = (N+1).float() #smoothing 
P = P / P.sum (1, keepdim = True)

log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        # Olabilirlik yani likelihood yüksek çıkmalı gerçek bir isim için ve tüm olasılıklar çarpılarak tüm listenin lossu bulunur
    #hesaplanırken önce tüm olasılıkların logu alnır ardından toplanır ve - eksi ile çarpılıp (hatanın artmasını sayının pozitif oalrak büyümesi)  toplam ikili sayısına bölünür
        log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
        n += 1
print(f"Model 1 (Sayım Modeli) Loss Puanı: {(-log_likelihood / n).item():.4f}")

#YAPAY SİNİR AĞI (NEURAL NETWORK)

xs,ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip (chs, chs[1:]):
        xs.append(stoi[ch1])  #girdileri xs listesine ekliyoruz ve integera çevirerek ekliyoruz yani harflerin sayı karşılığını ekliyoruz
        ys.append(stoi[ch2]) # çıktıları ys listesine ekliyoruz bu listeye göre lossu hesaplayıp gradlerimizi ve ağırlığı düzeleteceğiz

xs = torch.tensor(xs)
ys = torch.tensor(ys)
#girdi sayımızı belirliyoruz loss da lazım olacak
num = xs.nelement()


for k in range(100):

    # xs listesindeki tüm sayıları One-Hot vektörlere çevir çünkü e nin 5 m nin 13 olması büyüklük ilişkisi değil stuationdır yani 0,0,0,0,0,0,0,0,0,1,0,0,0,0 gibi bir vektör olacak
    xenc = F.one_hot(xs, num_classes=vocab_size).float()

    # 27x27 boyutunda, içi tamamen rastgele sayılarla dolu bir Ağırlık (Weight) matrisi
    W = torch.randn((vocab_size, vocab_size), requires_grad=True)

    # girdileri ağırlıklarla çarpıyoruz ortaya yine bir matris çıkar
    logits = xenc @ W

    counts = logits.exp() # logits içindeki sayıları pozitif hale getiriyoruz 
    probs = counts / counts.sum(1, keepdims=True) # Yüzdeliğe (olasılıklara) dönüştü yani ağırlığının olaslıklarını buldu (örn e yani 5 için 5. satırı çekersin ağırlık matrisinden
    # ve onun ağırlık puanlarını direkt alıp ys ile karşılaştırma yapmak yerine önce olasılığa dönüştüürüsün

    loss = -probs[torch.arange(num), ys].log().mean()

    W.grad = None   # Önceki hesaplamaların hafızasını sıfırla 
    loss.backward() # Geriye yayılımı başlat

    W.data += -50.0 * W.grad



# İSİM ÜRETMe
g = torch.Generator().manual_seed(2147483647)
print("--- TÜRKÇE ÜRETİLEN İSİMLER ---")
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))







