import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

b = {}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs, chs[1:]):
        biagram = (ch1, ch2)
        b[biagram] = b.get(biagram,0) + 1

# 27x27 boyutlarında, içi sıfırlarla dolu bir tamsayı (int32) matrisi oluştur
N = torch.zeros((27, 27), dtype=torch.int32)

#kelimedeki tüm farklı harfleri bulalım ve alfabetik sıraya dizelim
chars = sorted (list(set(''.join(words))))

#stoi yani harfleri sayılarla eşleştir, string to integer
stoi = {s:i+1 for i,s in enumerate(chars)}

#  noktayı (.) 0. sıraya koyuyoruz
stoi['.'] = 0

# itos (Integer to String): Tam tersi çeviri (1=a, 2=b...)
itos = {i:s for s,i in stoi.items()}

#ikili siistemde satırlar ilk harfi sütunlar ikinci harfi temsil edecek
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        # İlk harfin sayısını bul çünkü ilk harfin sayısı zaten o harfin hangi satırda olduğu oluyor
        ix1 = stoi[ch1]
        # İkinci harfin sayısını (sütun) bul
        ix2 = stoi[ch2]
        
        # Matriste o satır ve sütunun kesiştiği kutuya git ve sayıyı 1 artır böylece olaslık verilerini olyuşturmuş oluyoruz
        N[ix1, ix2] += 1

plt.figure(figsize=(16,16))
# Matrisi ekrana çiz
plt.imshow(N, cmap='Blues')

for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j] #i satırı temsil ediyor her satırın numarası da o satırın hangi harfe ait olduğunu sölüyor itos zaten harfi strine çeviriyordu en son chrs ab tutar
        # Kutunun içine harfleri yaz
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        # Kutunun içine o ikilinin sayısını yaz (N matrisinden çekerek)
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')

plt.axis('off')
plt.show() # Bu komutla dev tabloyu açarız