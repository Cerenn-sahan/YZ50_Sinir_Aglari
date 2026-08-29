import random
import math

# ==========================================
# 1. MİMARİ: Nöron, Katman ve Ağ Sınıfları
# ==========================================

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad 
        out._backward = _backward
        return out

    # Ters toplama işlemi (0 + Value hatasını çözer)
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "sadece int/float üsleri destekliyoruz"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        
        return out

    # Diğer işlemleri temel Lego parçalarıyla kuruyoruz
    def __rmul__(self, other): # other * self için
        return self * other

    def __truediv__(self, other): # self / other için
        return self * other**-1

    def __neg__(self): # -self için
        return self * -1

    def __sub__(self, other): # self - other için
        return self + (-other)

    def backward(self):
        topo = [] 
        visited = set() 
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: 
                    build_topo(child)
                topo.append(v) 
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo): 
            v._backward()

class Neuron:
    def __init__(self, nin):
        # nin kadar rastgele ağırlık ve 1 adet rastgele bias oluştur
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        
    def __call__(self, x):
        # w * x işlemlerini topla ve bias ekle, ardından tanh filtresinden geçir
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        # Nöronun ağırlıklarını ve biasını tek listede döndür
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        # nout kadar nöron oluştur ve listeye diz
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        # Gelen veriyi katmandaki tüm nöronlara ver
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # Katmandaki tüm nöronların parametrelerini topla
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        # Girdi sayısını ve katmanların nöron sayılarını birleştir
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        # Veriyi ilk katmandan son katmana kadar sırayla geçir
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # Ağdaki bütün katmanların parametrelerini tek bir devasa listede topla
        return [p for layer in self.layers for p in layer.parameters()]

# ==========================================
# 2. VERİ SETİ VE AĞI OLUŞTURMA
# ==========================================

# Ağımızı yaratıyoruz: 3 girdi alacak, iki tane 4'lü gizli katman ve 1 çıktısı olacak
n = MLP(3, [4, 4, 1])

# Karpathy'nin mini veri seti (4 örnek, 3 özellik)
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

# Ağın bulmasını istediğimiz gerçek hedefler
ys = [1.0, -1.0, -1.0, 1.0]

# ==========================================
# 3. EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# ==========================================

print("Eğitim Başlıyor...\n")

for k in range(20):
    
    # 1. İleri Yayılım (Forward Pass): Tahminleri al ve hatayı (Loss) hesapla
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # 2. MEŞHUR BUG ÇÖZÜMÜ: Geriye yayılımdan önce tüm gradyanları sıfırla!
    for p in n.parameters():
        p.grad = 0.0
        
    # 3. Geriye Yayılım (Backward Pass): Otomatik türev motorunu çalıştır
    loss.backward()
    
    # 4. Parametre Güncelleme (Gradient Descent): Hatanın tersi yönünde ufak bir adım at
    for p in n.parameters():
        p.data += -0.05 * p.grad
        
    # Her adımda loss değerinin nasıl düştüğünü ekrana yazdır
    print(f"Adım {k+1:2} | Kayıp (Loss): {loss.data:.4f}")

print("\nEğitim Tamamlandı! Hedeflere ne kadar yaklaşmışız bakalım:")
print("Gerçek Sonuçlar :", ys)
print("Ağın Tahminleri :", [y.data for y in ypred])