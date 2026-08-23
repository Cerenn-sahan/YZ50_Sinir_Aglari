import math 
class Neuron:
    def __init__ (self, gelen_kargo, gelen_baraj_degeri):
        #burada her nöron için örneğin A ve B, kendi ağırlık ve bias değerleri atanır.
        # Gelen kargoyu al, nöronun "benim_hafizam" adlı cebine koy.
        #self.benim_hafizam = gelen_kargo
        #self.benim_barajim = gelen_baraj_degeri
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

 #"Şu an dünyada bir sürü nöron üretilmiş olabilir ama sen bu işlemi ve bu bilgileri sadece o an üzerinde çalıştığın nöron için kullan, diğerlerine sakın dokunma."       

#Nöronu çalıştıralım ve test edelim
#İlk olarak kendimize bir nöron yaratalım
#şuan Neuron classının dışındayız o yüzden nesne üretelim
#benim_noronum = Neuron    demen yeterli new'e gerek yok.
benim_noronum = Neuron(agirliklar = [0.2, 0.8, -0.5], bias= 2.0)

#şimdi dışarıdan gelen piksellerin parlaklık değerini verelim
gelen_pikseller = [1.0, 2.0, 3.0]

# 3. Nöronumuza "Hadi bu pikselleri incele ve kararını ver (forward pass yap)" diyelim.
sonuc = benim_noronum.forward(gelen_pikseller)

#son olarak çıkan sonucu ekrana yazdıralım
print("Nöronun verdiği karar (0,1 arası):", sonuc)



class Layer:
    def __init__(self, noron_sayisi, her_norona_gelen_kablo_sayisi):
        # Katmanımızın içinde nöronları tutacağımız boş bir liste (cep) açıyoruz
        self.noronlar = []

        for i in range(noron_sayisi):
            # Not: Gerçekte ağırlıklar rastgele atanır ama şimdilik test için 
            # hepsine aynı örnek ağırlıkları ve bias'ı veriyoruz.
            ornek_agirliklar = [0.5]* her_norona_gelen_kablo_sayisi
            ornek_bias = 0.0

            yeni_noron = Neuron(agirliklar=ornek_agirliklar, bias=ornek_bias) #bu noronları zaten eğitmiştik 
            self.noronlar.append(yeni_noron) # Üretilen nöronu listeye ekle

     def forward(self, girdiler):
         katmanin_cevaplari = [] # Nöronların vereceği cevapları tutacağımız boş liste

         # Listemizdeki her bir nöronu tek tek dolaş
         for noron in self.noronlar:

             # İşçiye "Al bu girdileri, kendi 'forward' yeteneğinle hesabını yap" diyoruz.
             cevap = noron.forward(girdiler)

            # toplam_puani = 0
            # toplam_puani += girdiler[i]*self.agirliklar
            #toplam_puani += self.bias
            # katmanin_cevaplari[i] = toplam_puani   BUNLARI ZATEN YAPMIŞTK

             return katmanin_cevaplari


#Test aşaması
# İçinde 3 tane nöron olan bir katman (layer) üret.
# (Her bir nörona dışarıdan 2 tane girdi gelecek varsayalım)
# şuan o classta olmadığğımız için nesnesini üretip parametrelerini göndereceğiz
benim_katmanim = Layer(noron_sayisi = 3, her_norona_gelen_kablo_sayisi=2)
test_girdileri = [1.0, 2.0]

sonuclar = benim_katmanim.forward(test_girdileri)

print("Katmandaki nöronların ürettiği cevaplar:", sonuclar)



