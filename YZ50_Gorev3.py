
def Loss_hesapla( nihai, tahmin):
    #türev yerine daha kaba/genel bir düzeltmede bulunmaya çallışıyoruz
    uzaklik = nihai - tahmin
    ceza_loss = uzaklik**2
    return ceza_loss


