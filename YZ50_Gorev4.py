import math
import matplotlib.pyplot as plt # İŞTE EKSİK OLAN SATIR BU!
class Neuron:
    #def __init__ (self, gelen_kargo, gelen_baraj_degeri):
        #burada her nöron için örneğin A ve B, kendi ağırlık ve bias değerleri atanır.
        # Gelen kargoyu al, nöronun "benim_hafizam" adlı cebine koy.
        #self.benim_hafizam = gelen_kargo
        #self.benim_barajim = gelen_baraj_degeri

    def __init__ (self, agirliklar, bias):
        self.agirliklar = agirliklar
        self.bias = bias
        #gelen değerleri 0 ile 1 arasına sıkıştırmak için sigmoid fonksiyonu yazarız: 1/1+e^-x
        
    def sigmoid(self, x):
        return 1/ (1 + math.exp(-x))
        #Asıl olay forward pass yani ileri beslemedir. Dışarıdan piksellerin parlaklık değerleri gelecek bunlar girdileridir.
        #Nöron, bu girdileri kendi ağırlıklaryla çarpacak biasını ekleyecek (daha kesin tahminler için) ve sigmoid ile sıkıştırıp cevp ver.
        
        #bİR sınıfın yani lassın içine fonk yazdığında parametre olarak selfi yazmak zorundasın çünkü içindeki işlemleri self yani o anki 
        #verilen,gelen,gönderilen,bulunan parametre için yapar. Örn benim_nöronum.forward(gelen_pikseller)

    def forward(self, girdiler):
        toplam_puan = 0;

        #Gelen her girdiui kendi sırasındaki ağırlıkla çarp ve toplama ekle 
        for i in range(len(girdiler)):
            toplam_puan += girdiler[i] * self.agirliklar[i]

        #2. aşama olarak bu toplama nörronun kendi barajj yani bias değerini eklenir.
        toplam_puan += self.bias

        #3. aşama olarak ise çıkan sonucu 0 iile 1 arasına sıkıştırırız
        nihai_cevap = self.sigmoid(toplam_puan) #self.sigmoid yapıyoruz çünkü o anki nöron için bu işlemler yapılıyor
        return nihai_cevap

#-----------------------cEZA HESAPLAYICIS----------------------------------------------------------------------
def Loss_hesapla( nihai, tahmin):
    #türev yerine daha kaba/genel bir düzeltmede bulunmaya çallışıyoruz
    uzaklik = nihai - tahmin
    ceza_loss = uzaklik**2
    return ceza_loss

#---------------------------------------Parametre Değişmesi ile Loss değişimi---------------------------------------------

gercek_hedef = 1.0
gelen_piksel = [1.0] #bu dışarıdan nörone giren veridir!

denenecek_agirliklar = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
# YENİ: Grafik çizerken kullanmak için boş listeler (X ve Y ekseni verileri) oluşturuyoruz
cizim_icin_agirliklar = []
cizim_icin_loss_degerleri = []

print("Ağırlık testi başllıyor...\n")

for agirlik in denenecek_agirliklar:
    # O anki deneme ağırlığıyla YENİ BİR NÖRON üretiyoruz (Bias şimdilik 0 kalsın)
    test_noronu = Neuron(agirliklar = [agirlik], bias=0.0)

    #Nörona al bu pikseli ve bana bir tahmin ver diyoruz
    uretilen_tahmin = test_noronu.forward(gelen_piksel)

    #şimdi lossu hesaplaiyoruz
    ceza_puani = Loss_hesapla(nihai = gercek_hedef, tahmin = uretilen_tahmin)

    cizim_icin_agirliklar.append(agirlik)
    cizim_icin_loss_degerleri.append(ceza_puani)

    # Sonuçları ekrana yazdırıp izliyoruz
    print(f"Denenen Ağırlık: {agirlik} | Nöronun Tahmini: {uretilen_tahmin:.4f} | Ceza (Loss): {ceza_puani:.4f}")

    # YENİ: DÖNGÜ BİTTİKTEN SONRA GRAFİĞİ ÇİZDİRİYORUZ
plt.plot(cizim_icin_agirliklar, cizim_icin_loss_degerleri, marker='o', color='red') # marker='o' verilerin yerini nokta ile belli eder
plt.title("Ağırlık ve Ceza (Loss) Eğrisi")
plt.xlabel("Denenen Ağırlıklar")
plt.ylabel("Loss (Ceza Puanı)")
plt.grid(True) # Arka plana okumayı kolaylaştıran ızgaralar ekler
plt.show() # Grafiği ekranda göster!