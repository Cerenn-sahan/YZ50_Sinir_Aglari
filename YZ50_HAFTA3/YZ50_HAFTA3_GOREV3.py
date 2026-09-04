#smoothing sahte sayım ekliyoruzkii hiçbir zman içeride 0 olasılığına ulaşmasın çünkü 0/toplam 0 verir ve log0 infinetdir. her kutucuğun değerini 1 artırıız.
import torch
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


#  SMOOTHING (SAHTE SAYIM) VE NLL HESABI

P = (N + 1).float() 
P = P / P.sum(1, keepdim=True)


log_likelihood = 0.0
n = 0 # Toplam kaç tane ikili (bigram) saydığımızı tutacak

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1] #striingi ntegera çevirelimki hangi satır hangi sütuna gidilip içindeki olaslık değeriine bakılacağı anlaşılsın
        ix2 = stoi[ch2]
        
        # 1. Modelin bu ikiliye verdiği olasılığı çek yani o satır ve o sütunun içindeki sayıyı çek
        prob = P[ix1, ix2] 
        
        # 2. Olasılığın logaritmasını al
        logprob = torch.log(prob) 
        
        # 3. artık çarpma işleminden kurtulduk ve toplama yapıyoruz böylece sayımız çok fazla 0 ın altında çıkmıyor
        log_likelihood += logprob 
        
        # İkili (bigram) sayacını 1 artır
        n += 1 

# 4. Negatif yap (Eksi ile çarp)
nll = -log_likelihood #hataarttıkça pozitif olarak artmasını bekleriz 

# 5. Ortalamayı al
print(f'Average NLL (Loss): {nll / n}') #toplam ikili sayısına böl
