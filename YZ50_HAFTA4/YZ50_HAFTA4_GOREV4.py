import torch
import torch.nn.functional as F
# Grafiği çizdirebilmek için matplotlib kütüphanesini import ediyoruz
import matplotlib.pyplot as plt


# AYARLAMALAR 
words = open('names.txt').read().splitlines() #read komutu \n yanş kelimeler alt alta dizldiği için boşluk komutuyla birlikte bir kütle olarak okur
#splitlines ise aradai \n yani newlines ları siler ve artık liste cerenşahandilaramert gibi tek durur

chars = sorted(list(set(''.join(words)))) #joinwords diziyi metne çevirir yani  arallarında hiç boşluk bırakmadan dizer
#set ise her karakterin benzerini silerr ve benzersiz harf listesi haline getirir karışıktır
#list ise bu karışık listeyi diziye çevirir çünkü listenin, kümenin indexi olmaz
#sorted ise alfabetik sıraya dizer

stoi = {s: i+1 for i, s in enumerate(chars)} #her stringi integer karşılığı ile eşler
stoi['.'] = 0 #noktanın integer karşılığını 0 olarak beelirleyelim
itos = {i: s for s, i in stoi.items()} #integerın string karşılığğını  getirir
vocab_size = len(stoi) #toplam harf sayımız
block_size = 3 #üçlü küme sayımız 3

# VERİYİ ÜÇE BÖLME: TRAİN, DEV, TEST

def build_dataset(words):
    X,Y = [], [] # X olan üçlü kümeler matrisi Y olanise üçlü kümenin sıradaki harflerini tutan matrisi yani sonuç matrisi olacak hatta loss hesaplama da kullanacağız
    for w in words:
        context = [0]* block_size #en başta [0,0,0] olarak ayarlıyoruz yani ... olarak ayarlıyoruz
        for ch in w + '.':
            ix = stoi [ch]
            X.append(context)
            Y.append(ix) #Yani yeni  hharfi  Y ye ekliyoruz üçlü kümeden osnra gelen yeni harfi yani
            context = context[1:] + [ix] # :1 işlemi kümeyi bir sağa kaydırır boşluğa ise + olan kısmı ekler
    return torch.tensor(X), torch.tensor(Y) # X ve Y matrislerini torch tensoorlerine çeviriypruz 


#toplam kelime sayının yüzde 80 iini eğitim veriisne vereceğiz 
n1 = int (0.8 * len(words))
n2 = int(0.9 * len(words))

#yüzde 80 eğitim yanii train verisi olacak ağırlıkları güncellemek için kullanılacak
Xtr, Ytr = build_dataset(words[:n1]) 
#buillddataset zaten sonuç olarak X üçlü küme matrisi ve Y sonuç matrisi döndürüyordu ve kelimelerin yüüzde 80 ini kullanarak oluştur bunları dedik

# %10 Geliştirme (Dev/Validation) Verisi: Ağırlıklar güncellenmez, sadece loss ölçülür
Xdev, Ydev = build_dataset(words[n1:n2])

# %10 Test Verisi: Sistemin en son performansını ölçmek için saklanır
Xte, Yte = build_dataset(words[n2:])

# PARAMETRELERİ, MATRİSLERİ OLUŞTURMA 
C = torch.randn((vocab_size, 10)) #artık her harfe integera uzayda  10 konumlu olarak tanımlıyoruz
W1 = torch.randn ((30,200)) #her üçlü kümeden toplam 30 konum gelir artık  ayrıca nöronn sayısını iki katın açıkardıkyani
#her nörona 30 girdi gireceği için her nöronon 30 ağırlılıpı var 
b1 = torch.randn (200) 
W2 = torch.randn ((200, vocab_size)) #toplam 200 sonuç raporu gelir 200 nörondan ve her biri için vocab_size yanii toplam harften bir harf sonucuna ulaşmış olacak
b2 = torch.randn(27) #son çıkış matrisinin bias, w2 nin çıkışı 27 olduğu için

#bunlarıı tek biir parametres listesinde toplayalım ve .data ya da. grad diyerek bilgileine erişelim tek bir parametre ismi ile
parameters = [C, W1,  b1, W2, b2]

# Tüm matrislerin "türev hesaplamasına" dahil olması gerektiğini PyTorch'a bildiriyoruz
for p in parameters:
    p.requires_grad = True


# EĞİTİM DÖNGÜSÜ 

for i in range(10000):
    # Minibatch: Eğitim setinden (Xtr) rastgele 32 satır seçiyoruz
    ix = torch.randint(0, Xtr.shape[0], (32,))

    #İleri yönlü hesaplaam (forward)
    emb = C[Xtr[ix]] # (32,3,10)
    h = torch.tanh(emb.view (-1,30) @ W1 + b1) #önce küp matris olmaktan çıkarıp her 30 luyu yanyana dizdik ve bu girdi matrisini aırlıklarla çarpıp bias ekleyip -1+1 arasına sıkıştırdık
    logits = h @ W2 +  b2 # (32,200) lük geçiş matrisini sonuçları alabilmek için w2 olan (200,27 yani vocab_size) lık son çıkış matrisi ile çarptık yani (32,27)

    #şimdi loss hesaplayalım f.cross.entropty ile
    loss = F.cross_entropy (logits, Ytr[ix]) # loss hesaplarkan sonuç ve tahmin matrislerini aldı

    # 3. Geriye Yayılım (Backward Pass)
    # Bir önceki döngüden kalan türevleri (grad) sıfırlıyoruz ki birbirine karışmasın
    for p in parameters:
        p.grad = None

    #her bir parametre için tüürev değerii hesaplıyoruz
    loss.backward()

    #Paramaterleri güncelleeme
    lr = 0.1 #öğrenme oranı

    for p in parameters:

        #MEVCUT VERİYİ (P.DATA) TÜREV YÖNÜNDE DEĞİŞTİRYORUZ
        p.data += -lr * p.grad   # p.data = yyerine += diyoruz

print(f"Eğitim tamamlandı. Son Minibatch Loss Puanı: {loss.item():.4f}")

#Geliştirme (Dev set) ile test
# Burada `backward()` veya `p.data +=` işlemleri YOKTUR. Ağırlıklar sabittir.
# Bütün Geliştirme matrisini (Xdev) aynı anda ileri yönlü hesaplamaya sokuyoruz.


emb =C[Xdev]
h = torch.tanh(emb.view(-1,30) @ W1 + b1)
logits = h @ W2 + b2
dev_loss = F.cross_entropy (logits, Ydev)

print(f"Hiç Görülmeyen Veride (Dev Set) Loss Puanı: {dev_loss.item():.4f}")


# 20 farklı isim üretmek için ana döngü
for _ in range(20):
    out = [] 
    context = [0, 0, 0] # Başlangıç: 3 tane "." karakteri
    
    while True:
        # 1. Mevcut bağlamın matris hesaplamaları
        emb = C[torch.tensor([context])] 
        
        # BOŞLUK 1: emb matrisini düzleştir
        h = torch.tanh(emb.view(1,30) @ W1 + b1) 
        
        logits = h @ W2 + b2
        
        # BOŞLUK 2: Ham puanları olasılığa çevir
        probs = F.softmax(logits, dim=1) #olasılığa çevirmek için ham puanlar logitsde
        
        # 3. Olasılıklara göre rastgele bir harf seç
        ix = torch.multinomial(probs, num_samples=1).item()
        
        # BOŞLUK 3: Harf 0 (nokta) ise döngüyü kır
        if ix == 0:
            break #döngüyü kırmak için kullanılır 
            
        out.append(itos[ix])
        
        # BOŞLUK 4: Pencereyi kaydır (yeni harfi ekle)
        context = context[1:] + [ix] #Yeni seçilen harfin numarasını listenin sonuna ekliyoruz
        
    print(''.join(out))




#iism üretme

# 8x8 inç boyutunda boş bir tuval (figür) oluşturuyoruz
plt.figure(figsize=(8,8))

# C matrisinin 0. sütununu X ekseni, 1. sütununu Y ekseni olarak alıp noktaları yerleştiriyoruz
# .data komutunu kullanıyoruz çünkü PyTorch'un türev (grad) takip mekanizmasını grafiğe sokmak istemiyoruz
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)

# Döngü ile her bir noktanın tam üzerine o noktaya denk gelen harfi (itos[i]) yazdırıyoruz
for i in range(C.shape[0]):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color='white')

# Arka plana ızgara (grid) ekliyoruz
plt.grid('minor')

# Grafiği ekranda göster
plt.show()












