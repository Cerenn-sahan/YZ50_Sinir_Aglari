#Dördüncü görevde ağırlıkları kendimiz randoom olarak tek tek verip forwarddan çıkan tahminle gerçek tahmin arası  farkı bulup loss yapıyorduk. 
#5. Görevde, Şu anki hatan (loss) bu. Şimdi sayısal türev kullanarak eğimi hesapla ve ağırlığını çukura (0'a) doğru minik bir adım (learning rate) kaydır diyeceğiz.
 #Görev: Sayısal türev (numerical derivative) ile basit bir gradient descent döngüsü kur: parametreyi küçük adımlarla güncelleyerek loss'u düşür.

import math
import matplotlib.pyplot as plt # İŞTE EKSİK OLAN SATIR BU!
class Neuron:
   

    def __init__ (self, agirliklar, bias):
        self.agirliklar = agirliklar
        self.bias = bias
        
    def sigmoid(self, x):
        return 1/ (1 + math.exp(-x))
       
    def forward(self, girdiler):
        toplam_puan = 0;

        for i in range(len(girdiler)):
            toplam_puan += girdiler[i] * self.agirliklar[i]

        toplam_puan += self.bias
        nihai_cevap = self.sigmoid(toplam_puan) 
        return nihai_cevap

#-----------------------cEZA HESAPLAYICIS----------------------------------------------------------------------
def Loss_hesapla( nihai, tahmin):
    #türev yerine daha kaba/genel bir düzeltmede bulunmaya çallışıyoruz
    uzaklik = nihai - tahmin
    ceza_loss = uzaklik**2
    return ceza_loss

    #----------------------------5. gÖREV------------------------------------------------------------------------------------------------------------------------------
def sayisal_turev_hesapla(eski_agirlik, gercek_cevap, gelen_veri):
    h=0.0001

    # Adım 1: Nöronu mevcut (eski) ağırlığıyla kur ve cezasını (loss) ölç
    noron_eski = Neuron(agirliklar = [eski_agirlik], bias=0.0)
    tahmin_eski= noron_eski.forward(gelen_veri)
    loss_eski = Loss_hesapla (nihai = gercek_cevap, tahmin= tahmin_eski)

    # Adım 2: Ağırlığı 'h' kadar dürt (artır)
    durtulmus_agirlik = eski_agirlik + h

    # Adım 3: Nöronu bu DÜRTÜLMÜŞ ağırlıkla kur ve yeni cezasını (loss) ölç
    noron_yeni = Neuron(agirliklar=[durtulmus_agirlik], bias=0.0)
    tahmin_yeni = noron_yeni.forward(gelen_veri)
    loss_yeni = Loss_hesapla(nihai=gercek_cevap, tahmin=tahmin_yeni)

    # Adım 4: Karpathy'nin o meşhur eğim formülünü uygula ve sonucu fırlat

    egim= (loss_yeni - loss_eski) / h
    return egim

#------------------------------------Akıllı Öğrenma (Gradent Discent ) --------------------------------------------------------------------------------------
gercek_hedef= 1.0
gelen_piksel= [1.0]

# Nörona "kötü" bir başlangıç ağırlığı veriyoruz. Bakalım düzeltebilecek mi?
suanki_agirlik = -2.0 
ogrenme_hizi = 0.5 # Her adımda eğim yönünde ne kadar güçlü zıplayacağını belirler

print(f"--- ÖĞRENME KAMPI BAŞLIYOR ---\nBaşlangıç Ağırlığı: {suanki_agirlik}")

# Nöronu 100 kere eğitimden geçirelim
for adim in range(100):

    # 1. Gözlemciye sor: "Şu anki ağırlığımla yokuşun eğimi ne durumda?"
    yokusun_egimi = sayisal_turev_hesapla (eski_agirlik=suanki_agirlik, gercek_cevap=gercek_hedef, gelen_veri=gelen_piksel)

    # 2. Gradient Descent Kurallı: Ağırlığı eğimin TERSİNE doğru güncelle (yokuş aşağı in)
    suanki_agirlik = suanki_agirlik - (ogrenme_hizi * yokusun_egimi)

    #böylece yeni ağırlık 100 adımda her seferinde h eklenmiş hali olur eski ise eski 1 eksik h eklenmiş hali olur tahmin hesaplanır lossu hesaplanır, gercek cevap sabit
    #en son ise yeni loss - bir öncekiadımdaki loss yapar, artırdığına böler ve egimi bulur daha sonrasıda agirligi bulunan egime gore guneller

    # Gelişimi görebilmek için sadece her 10 adımda bir ekrana yazdıralım
    if adim % 10 == 0:
        # O anki güncel durumun loss'unu hesaplayıp ekrana veriyoruz
        test_noronu = Neuron(agirliklar=[suanki_agirlik], bias=0.0)
        guncel_tahmin = test_noronu.forward(gelen_piksel)
        guncel_loss = Loss_hesapla(nihai=gercek_hedef, tahmin=guncel_tahmin)
        
        print(f"Adım {adim} | Güncel Ağırlık: {suanki_agirlik:.4f} | Loss: {guncel_loss:.4f} | Eğim: {yokusun_egimi:.4f}")

print("\n--- KAMP BİTTİ ---")
print(f"Nöronun Bulduğu En Mükemmel Ağırlık: {suanki_agirlik:.4f}")
   

