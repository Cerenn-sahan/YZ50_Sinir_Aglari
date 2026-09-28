import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))

stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0

itos = {i: s for s, i in stoi.items()}

vocab_size = len(itos) #harf sayısı nokta ile beraber 27

print(stoi)
print("Vocabulary size:", vocab_size)

random.seed(42)
random.shuffle(words) #kelimeleri karıştırırız

def build_dataset(words, block_size): #Burayı fonksiyon yaptık çünkü block size boyutunu ayarlayıp önce 3 sonra 8 deneyip loss ölçeğiz
    X = []
    Y = []

    for w in words:

        # İlk başta context tamamen "." karakterlerinden oluşuyor.
        context = [0] * block_size

        for ch in w + ".":
            ix = stoi[ch]

            X.append(context)
            Y.append(ix)

            # En eski karakteri çıkar,
            # yeni karakteri sona ekle.
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

#Dataseti bölüyoruz
def create_splits(block_size):

    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size)
    Xte, Yte = build_dataset(words[n2:], block_size)

    print("Train:", Xtr.shape, Ytr.shape)
    print("Validation:", Xdev.shape, Ydev.shape)
    print("Test:", Xte.shape, Yte.shape)

    return Xtr, Ytr, Xdev, Ydev, Xte, Yte



#Linear katmanımız girdi matrisini w1 ile çarpıp b ekler
class Linear:

    def __init__(self, fan_in, fan_out, bias=True):

        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5

        self.bias = torch.zeros(fan_out) if bias else None #w1 ve b matrislerimi oluşturuyoruz

    def __call__(self, x):

        self.out = x @ self.weight

        if self.bias is not None:
            self.out += self.bias #işlemleri yapıyoruz 

        return self.out

    def parameters(self):

        return [self.weight] + ([] if self.bias is None else [self.bias])

#BATCHNORM KATMANI
class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):

        self.eps = eps
        self.momentum = momentum

        self.training = True

        # Öğrenilen parametreler
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # Eğitim sırasında güncellenen istatistikler
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):

        if self.training:

            # Normal MLP:
            # (B, C)
            if x.ndim == 2:
                dim = 0

            # WaveNet biçimi:
            # (B, T, C)
            elif x.ndim == 3:
                dim = (0, 1)

            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)

        else:

            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)

        self.out = self.gamma * xhat + self.beta

        if self.training:

            with torch.no_grad():

                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * xmean
                )

                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * xvar
                )

        return self.out

    def parameters(self):

        return [self.gamma, self.beta]


#Tanh katmanı
class Tanh:

    def __call__(self, x):

        self.out = torch.tanh(x)

        return self.out

    def parameters(self):

        return []

#Embedding classı
class Embedding:

    def __init__(self, num_embeddings, embedding_dim):

        self.weight = torch.randn(
            (num_embeddings, embedding_dim)
        )

    def __call__(self, IX):

        self.out = self.weight[IX]

        return self.out

    def parameters(self):

        return [self.weight]


#FlattenConsecutive ile B,8,10 luk matrisi B,80 yapmak yerine önce 4 erli gruplara daha sonra 2 li gruplara ayırırız ve daha sonra toparlarız tekrar tek gruba.
# öneğn 4,8,10 luk düşünelim artık 4,4,20 (her sayınnın iki iki sayı içeriyor artık 10*2 den 20 özelliği olur daha sonra ise 4,2,40) yani (B, T/2, 2C)
#4,2,40 olmaz çünkü ağırlık matrisi 20,200 ile çarpıyoruz (nöronlarla) ve çıkan sonuç 4,4,200 olur 
#İkinci FlattenConsecutive(2) bundan sonra çalıştığı için: (4,2,400) olur.
class FlattenConsecutive:

    def __init__(self, n):

        self.n = n

    def __call__(self, x):

        B, T, C = x.shape

        x = x.view(
            B,
            T // self.n,
            C * self.n
        )

        # Eğer ortadaki boyut 1 olduysa kaldır.
        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x

        return self.out

    def parameters(self):

        return []


#BÜTÜN BU KATMANLARI TEK TEK ÇAPIRMAK YERİNE model(x) DİYECEĞİZ BUNUN İÇİN:
class Sequential:

    def __init__(self, layers):

        self.layers = layers

    def __call__(self, x):

        for layer in self.layers:
            x = layer(x)

        self.out = x

        return self.out

    def parameters(self):

        return [
            p
            for layer in self.layers
            for p in layer.parameters()
        ]

#MODELİ TRAİN MODA ALMA
def set_mode(model, training):

    for layer in model.layers:

        if hasattr(layer, "training"):
            layer.training = training


#Model eğitme
def train_model(
    model,
    Xtr,
    Ytr,
    max_steps=200000,
    batch_size=32
):

    parameters = model.parameters()

    for p in parameters:
        p.requires_grad = True

    set_mode(model, True)

    lossi = []

    for i in range(max_steps):

        # -----------------------------
        # MINIBATCH
        # -----------------------------

        ix = torch.randint(
            0,
            Xtr.shape[0],
            (batch_size,)
        )

        Xb = Xtr[ix]
        Yb = Ytr[ix]

        # -----------------------------
        # FORWARD
        # -----------------------------

        logits = model(Xb)

        loss = F.cross_entropy(
            logits,
            Yb
        )

        # -----------------------------
        # BACKPROPAGATION
        # -----------------------------

        for p in parameters:
            p.grad = None

        loss.backward()

        # -----------------------------
        # PARAMETRE UPDATE
        # -----------------------------

        lr = 0.1 if i < 150000 else 0.01

        for p in parameters:
            p.data += -lr * p.grad

        # -----------------------------
        # LOSS KAYDI
        # -----------------------------

        lossi.append(loss.log10().item())

        if i % 10000 == 0:

            print(
                f"{i:7d}/{max_steps:7d} "
                f"loss: {loss.item():.4f}"
            )

    return lossi


#loss hesaplama
@torch.no_grad()
def calculate_loss(model, X, Y):

    set_mode(model, False)

    logits = model(X)

    loss = F.cross_entropy(
        logits,
        Y
    )

    return loss.item()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ============================================================
# GÖREVİN DEVAMI
# ============================================================


# ------------------------------------------------------------
# 1) ÖNCE ESKİ MODEL: BLOCK SIZE = 3
# ------------------------------------------------------------

block_size = 3

Xtr3, Ytr3, Xdev3, Ydev3, Xte3, Yte3 = create_splits(block_size)

torch.manual_seed(42)

n_embd = 10
n_hidden = 200


# Eski modelde 3 karakterin embeddinglerini direkt birleştiriyoruz.
#
# Input:
# (B, 3)
#
# Embedding sonrası:
# (B, 3, 10)
#
# FlattenConsecutive(3):
# (B, 30)
#
# Linear:
# (B, 200)
#
# Output:
# (B, 27)

model3 = Sequential([

    Embedding(
        vocab_size,
        n_embd
    ),

    FlattenConsecutive(3),

    Linear(
        n_embd * 3,
        n_hidden,
        bias=False
    ),

    BatchNorm1d(
        n_hidden
    ),

    Tanh(),

    Linear(
        n_hidden,
        vocab_size
    )
])


# Son katmanın başlangıçtaki değerlerini biraz küçültüyoruz.
# Bunun sebebi başlangıçta modelin aşırı büyük logitler üretmesini engellemek.

with torch.no_grad():
    model3.layers[-1].weight *= 0.1


print("\n========================================")
print("3 CONTEXT MODELİ")
print("========================================")

print(
    "Parametre sayısı:",
    sum(p.nelement() for p in model3.parameters())
)


# Modeli eğitiyoruz.

lossi3 = train_model(
    model3,
    Xtr3,
    Ytr3,
    max_steps=200000,
    batch_size=32
)


# Eğitim ve validation loss hesaplıyoruz.

train_loss_3 = calculate_loss(
    model3,
    Xtr3,
    Ytr3
)

val_loss_3 = calculate_loss(
    model3,
    Xdev3,
    Ydev3
)


print("\n3 CONTEXT SONUÇLARI")

print(
    "Train loss:",
    train_loss_3
)

print(
    "Validation loss:",
    val_loss_3
)



# ------------------------------------------------------------
# 2) BLOCK SIZE'I 3'TEN 8'E ÇIKARIYORUZ
# ------------------------------------------------------------

block_size = 8

Xtr8, Ytr8, Xdev8, Ydev8, Xte8, Yte8 = create_splits(block_size)

torch.manual_seed(42)


# Burada henüz WaveNet yapmıyoruz.
#
# Sadece eski MLP modeline 3 yerine 8 karakter veriyoruz.
#
# Input:
# (B, 8)
#
# Embedding:
# (B, 8, 10)
#
# FlattenConsecutive(8):
#
# 8 tane 10 boyutlu embedding tek seferde birleşir.
#
# (B, 8, 10)
#
# ->
#
# (B, 80)

model8_flat = Sequential([

    Embedding(
        vocab_size,
        n_embd
    ),

    FlattenConsecutive(8),

    Linear(
        n_embd * 8,
        n_hidden,
        bias=False
    ),

    BatchNorm1d(
        n_hidden
    ),

    Tanh(),

    Linear(
        n_hidden,
        vocab_size
    )
])


with torch.no_grad():
    model8_flat.layers[-1].weight *= 0.1


print("\n========================================")
print("8 CONTEXT DÜZ MODEL")
print("========================================")

print(
    "Parametre sayısı:",
    sum(p.nelement() for p in model8_flat.parameters())
)


lossi8 = train_model(
    model8_flat,
    Xtr8,
    Ytr8,
    max_steps=200000,
    batch_size=32
)


train_loss_8 = calculate_loss(
    model8_flat,
    Xtr8,
    Ytr8
)

val_loss_8 = calculate_loss(
    model8_flat,
    Xdev8,
    Ydev8
)


print("\n8 CONTEXT SONUÇLARI")

print(
    "Train loss:",
    train_loss_8
)

print(
    "Validation loss:",
    val_loss_8
)



# ------------------------------------------------------------
# 3) 3 CONTEXT VE 8 CONTEXT LOSS KARŞILAŞTIRMASI
# ------------------------------------------------------------

loss_drop = val_loss_3 - val_loss_8


print("\n========================================")
print("LOSS KARŞILAŞTIRMASI")
print("========================================")

print(
    "3 karakter Validation Loss:",
    val_loss_3
)

print(
    "8 karakter Validation Loss:",
    val_loss_8
)

print(
    "Validation Loss düşüşü:",
    loss_drop
)


# Eğer sonuç pozitifse loss düşmüştür.

if loss_drop > 0:

    print(
        "Sonuç: Context 3'ten 8'e çıkarılınca validation loss",
        loss_drop,
        "kadar düştü."
    )

else:

    print(
        "Bu eğitim çalıştırmasında validation loss düşmedi."
    )



# ------------------------------------------------------------
# 4) ŞİMDİ WAVENET BENZERİ MODELİ KURUYORUZ
# ------------------------------------------------------------

torch.manual_seed(42)


# Burada artık 8 karakteri TEK SEFERDE:
#
# (B,8,10) -> (B,80)
#
# yapmıyoruz.
#
#
# Bunun yerine:
#
# 8 grup
# ↓
# 4 grup
# ↓
# 2 grup
# ↓
# 1 grup
#
# şeklinde kademeli birleştiriyoruz.


wavenet_model = Sequential([


    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    Embedding(
        vocab_size,
        n_embd
    ),

    # Input:
    #
    # (B,8)
    #
    # Embedding sonrası:
    #
    # (B,8,10)



    # --------------------------------------------------------
    # 1. KATMAN
    #
    # 8 GRUP -> 4 GRUP
    # --------------------------------------------------------

    FlattenConsecutive(2),

    # (B,8,10)
    #
    # iki iki birleşir
    #
    # 8 / 2 = 4
    #
    # 10 * 2 = 20
    #
    # sonuç:
    #
    # (B,4,20)


    Linear(
        n_embd * 2,
        n_hidden,
        bias=False
    ),

    # Linear(20,200)
    #
    # Son boyutu 20'den 200'e dönüştürür.
    #
    # (B,4,20)
    #
    # ->
    #
    # (B,4,200)


    BatchNorm1d(
        n_hidden
    ),

    # Shape değişmez:
    #
    # (B,4,200)


    Tanh(),

    # Shape değişmez:
    #
    # (B,4,200)



    # --------------------------------------------------------
    # 2. KATMAN
    #
    # 4 GRUP -> 2 GRUP
    # --------------------------------------------------------

    FlattenConsecutive(2),

    # İki tane 200 boyutlu temsil birleşir.
    #
    # 200 + 200 = 400
    #
    # (B,4,200)
    #
    # ->
    #
    # (B,2,400)


    Linear(
        n_hidden * 2,
        n_hidden,
        bias=False
    ),

    # Linear(400,200)
    #
    # 400 tane bilgiyi kullanarak
    # 200 yeni özellik üretir.
    #
    # (B,2,400)
    #
    # ->
    #
    # (B,2,200)


    BatchNorm1d(
        n_hidden
    ),

    # (B,2,200)


    Tanh(),

    # (B,2,200)



    # --------------------------------------------------------
    # 3. KATMAN
    #
    # 2 GRUP -> 1 GRUP
    # --------------------------------------------------------

    FlattenConsecutive(2),

    # İki tane 200 boyutlu temsil birleşir.
    #
    # (B,2,200)
    #
    # ->
    #
    # (B,1,400)
    #
    # Ama FlattenConsecutive içerisinde
    #
    # squeeze(1)
    #
    # yaptığımız için:
    #
    # (B,400)


    Linear(
        n_hidden * 2,
        n_hidden,
        bias=False
    ),

    # Linear(400,200)
    #
    # (B,400)
    #
    # ->
    #
    # (B,200)


    BatchNorm1d(
        n_hidden
    ),

    # (B,200)


    Tanh(),

    # (B,200)



    # --------------------------------------------------------
    # OUTPUT LAYER
    # --------------------------------------------------------

    Linear(
        n_hidden,
        vocab_size
    )

    # Linear(200,27)
    #
    # (B,200)
    #
    # ->
    #
    # (B,27)
    #
    # Çünkü 27 tane olası çıktı karakterimiz var.
])


with torch.no_grad():
    wavenet_model.layers[-1].weight *= 0.1



print("\n========================================")
print("WAVENET MODELİ")
print("========================================")

print(
    "WaveNet parametre sayısı:",
    sum(
        p.nelement()
        for p in wavenet_model.parameters()
    )
)



# ------------------------------------------------------------
# 5) SHAPE'LERİ GÖRMEK İÇİN 4 ÖRNEK SEÇİYORUZ
# ------------------------------------------------------------

ix = torch.randint(
    0,
    Xtr8.shape[0],
    (4,)
)

Xb = Xtr8[ix]
Yb = Ytr8[ix]


print("\nSeçilen inputların shape'i:")

print(
    Xb.shape
)


print("\nSeçilen inputlar:")

print(
    Xb
)



# ------------------------------------------------------------
# 6) MODELİ FORWARD PASS'TEN GEÇİRİYORUZ
# ------------------------------------------------------------

set_mode(
    wavenet_model,
    True
)

logits = wavenet_model(
    Xb
)



# ------------------------------------------------------------
# 7) HER KATMANIN ÇIKTI SHAPE'İNİ YAZDIRIYORUZ
# ------------------------------------------------------------

print("\n========================================")
print("HER KATMANIN SHAPE ÇIKTISI")
print("========================================\n")


print(
    "Input                    :",
    tuple(Xb.shape)
)


for i, layer in enumerate(wavenet_model.layers):

    print(
        f"{i:2d} "
        f"{layer.__class__.__name__:24s}"
        f" -> {tuple(layer.out.shape)}"
    )



# ------------------------------------------------------------
# 8) BEKLENEN SHAPE AKIŞINI AÇIKÇA YAZDIRALIM
# ------------------------------------------------------------

print("\n========================================")
print("WAVENET SHAPE ÖZETİ")
print("========================================")


print(
    """
Input                    : (4, 8)

Embedding                : (4, 8, 10)

FlattenConsecutive(2)    : (4, 4, 20)

Linear(20 -> 200)        : (4, 4, 200)

BatchNorm                : (4, 4, 200)

Tanh                     : (4, 4, 200)

FlattenConsecutive(2)    : (4, 2, 400)

Linear(400 -> 200)       : (4, 2, 200)

BatchNorm                : (4, 2, 200)

Tanh                     : (4, 2, 200)

FlattenConsecutive(2)    : (4, 400)

Linear(400 -> 200)       : (4, 200)

BatchNorm                : (4, 200)

Tanh                     : (4, 200)

Linear(200 -> 27)        : (4, 27)
"""
)



# ------------------------------------------------------------
# 9) WAVENET MODELİNİ DE EĞİTELİM
# ------------------------------------------------------------

print("\n========================================")
print("WAVENET EĞİTİMİ BAŞLIYOR")
print("========================================")


lossi_wavenet = train_model(
    wavenet_model,
    Xtr8,
    Ytr8,
    max_steps=200000,
    batch_size=32
)



# ------------------------------------------------------------
# 10) WAVENET LOSS DEĞERLERİ
# ------------------------------------------------------------

wave_train_loss = calculate_loss(
    wavenet_model,
    Xtr8,
    Ytr8
)

wave_val_loss = calculate_loss(
    wavenet_model,
    Xdev8,
    Ydev8
)

wave_test_loss = calculate_loss(
    wavenet_model,
    Xte8,
    Yte8
)


print("\n========================================")
print("WAVENET SONUÇLARI")
print("========================================")


print(
    "Train loss:",
    wave_train_loss
)

print(
    "Validation loss:",
    wave_val_loss
)

print(
    "Test loss:",
    wave_test_loss
)



# ------------------------------------------------------------
# 11) TÜM MODELLERİ KARŞILAŞTIR
# ------------------------------------------------------------

print("\n========================================")
print("TÜM MODELLERİN KARŞILAŞTIRMASI")
print("========================================")


print(
    "3 context validation loss  :",
    val_loss_3
)

print(
    "8 context düz validation   :",
    val_loss_8
)

print(
    "WaveNet validation loss    :",
    wave_val_loss
)



# ------------------------------------------------------------
# 12) LOSS GRAFİKLERİ
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    torch.tensor(lossi3).view(-1, 1000).mean(1),
    label="3 context"
)

plt.plot(
    torch.tensor(lossi8).view(-1, 1000).mean(1),
    label="8 context flat"
)

plt.plot(
    torch.tensor(lossi_wavenet).view(-1, 1000).mean(1),
    label="WaveNet"
)

plt.xlabel(
    "Training step (x1000)"
)

plt.ylabel(
    "log10(loss)"
)

plt.title(
    "Modellerin Eğitim Loss Karşılaştırması"
)

plt.legend()

plt.show()



# ------------------------------------------------------------
# 13) WAVENET İLE YENİ İSİMLER ÜRETME
# ------------------------------------------------------------

set_mode(
    wavenet_model,
    False
)


print("\n========================================")
print("MODELİN ÜRETTİĞİ ÖRNEK İSİMLER")
print("========================================\n")


g = torch.Generator().manual_seed(2147483647 + 10)


for _ in range(20):

    out = []

    context = [0] * 8

    while True:

        x = torch.tensor(
            [context]
        )

        logits = wavenet_model(
            x
        )

        probs = F.softmax(
            logits,
            dim=1
        )

        ix = torch.multinomial(
            probs,
            num_samples=1,
            generator=g
        ).item()

        context = context[1:] + [ix]

        out.append(ix)

        if ix == 0:
            break

    generated_name = "".join(
        itos[i]
        for i in out
    )

    print(
        generated_name
    )



# ------------------------------------------------------------
# 14) GÖREVİN SON ÖZETİNİ EKRANA YAZDIR
# ------------------------------------------------------------

print("\n========================================")
print("GÖREV SONUCU")
print("========================================")


print(
    f"""
1) İlk modelde context uzunluğu 3'tü.

   Validation Loss:
   {val_loss_3:.4f}


2) Context uzunluğunu 3'ten 8'e çıkardık.

   Validation Loss:
   {val_loss_8:.4f}


3) Validation loss değişimi:

   {val_loss_3:.4f} - {val_loss_8:.4f}
   =
   {loss_drop:.4f}


4) Düz 8-context modelinde:

   (B,8,10)
       ↓
   (B,80)

   yapılarak bütün karakterler tek seferde
   flatten ediliyordu.


5) WaveNet benzeri modelde ise:

   8
   ↓
   4
   ↓
   2
   ↓
   1

   şeklinde üç aşamada birleştirme yaptık.


6) WaveNet'in temel shape zinciri:

   (B,8,10)
       ↓
   (B,4,20)
       ↓
   (B,4,200)
       ↓
   (B,2,400)
       ↓
   (B,2,200)
       ↓
   (B,400)
       ↓
   (B,200)
       ↓
   (B,27)


7) WaveNet Validation Loss:

   {wave_val_loss:.4f}
"""
)


