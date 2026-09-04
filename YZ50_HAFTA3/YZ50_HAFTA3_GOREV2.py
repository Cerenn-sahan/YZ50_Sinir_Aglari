import torch

# ==========================================
# 1. BÖLÜM: N MATRİSİNİN VE SÖZLÜKLERİN İNŞASI
# ==========================================
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1







p = N[0].float() #sadece 0. satırı alır ve flosata çevirir içindeki tüm kutunun içindeki rakaamları
p = p / p.sum() #o satırın toplamını o satırdaki tüm farklı değer olan rakamlara bölüp net toplamı yüz olan olasılık elde ettik manuel olarak

# Her seferinde aynı rastgele sayıları almak için
g = torch.Generator().manual_seed(2147483647)
#  1 tane indeks numarası çek diyoruz
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

P = N.float()
# 27*27 ye 27*1 verir yani sütun sütun hizalayarak bölme yapar, satırı alıp o satırdaki her değeri o toplama bölmek yerine # 27*27 den önce 1. sütunu alır ve 27*1 lik dikey sütun ile karşılık gelen satıları böler ve böyle gider yani 27 farklı satırı kendi toplam değerlerine bölüyoruz
P = P / P.sum(1, keepdim=True)  


g = torch.Generator().manual_seed(2147483647)
# 10 tane isim uydurmasını istiyoruz
for i in range(10):
    out = []
    ix = 0 # Her zaman '.' (0) ile başlıyoruz yanii ilk harfi bulacağız önce onlarında yüzdesi hazırdı zaten örneğin kelimelerin yüzde 15 i a ile başlar gibi
    
    while True:
        # P matrisinden, o anki harfimizin satırını (olasılıkları) çek örneğin şuan 0. satırın değerlerini yani her sütundaki olaslık değerlerini çekecek
        p = P[ix] 
        
        # bu ixlerden en yüksek olanı en yüksek olasılıkla çekilmek üzere bir ix seçilir rastgele ve while döngüsü en başata artık örneğin 5.  satır için çalışacak 
        #ve daha snra 5. satırın en yüksek olaslıklı olanını çekebilmek ihtimaline göre örneğin 3 çekecek sonra 3. satır için işlem başa dönecek
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        
        # Çıkan indeksi harfe çevir ve kelimeye ekle
        out.append(itos[ix])
        
        # Eğer çektiğimiz harf '.' (0) ise kelime bitmiş demektir, döngüyü kır
        if ix == 0:
            break
            
    # Listeyi birleştirip üretilen ismi ekrana yazdır
    print(''.join(out))