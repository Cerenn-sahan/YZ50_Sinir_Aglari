import math
import torch

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

#TEST
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')

x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

# BURASI DEĞİŞTİ: tanh'ın parçalanmış, saf matematiksel hali
e = (2 * n).exp()
o = (e - 1) / (e + 1)
o.label = 'o'

o.backward()

print("Kendi Yazdığımız Motorun Sonuçları:")
print(f"w1 grad: {w1.grad}")
print(f"w2 grad: {w2.grad}")




#pytorch kısmı
x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

o.backward()

print("\nPyTorch Motorunun Sonuçları:")
print(f"w1 grad: {w1.grad.item()}")
print(f"w2 grad: {w2.grad.item()}")



#Örneğin ağırlıkla çok ufak bir değer olacak şekilde oynalayım ve bu değere göre ham değeri hesaplayıp tanh nin açılımını uygulayaıp backward ile tüm gardiantları hesaplayalım
# ==========================================
# 3. YÖNTEM: SAYISAL TÜREV 
# ==========================================
h = 0.0001 #işte bu deeri ekleyerrek yeni ve eski deer yardımıyla klasik türev yapıp eğim yani grad bulacağız

# Adım 1: Sistemin Normal (Eski) Çıktısını Hesapla
x1 = Value(2.0); x2 = Value(0.0)
w1 = Value(-3.0); w2 = Value(1.0)
b = Value(6.8813735870195432)

n_eski = (x1*w1) + (x2*w2) + b
e_eski = (2 * n_eski).exp()
out_eski = (e_eski - 1) / (e_eski + 1)

# Adım 2: w1'i 'h' kadar dürt ve Yeni Çıktıyı Hesapla
x1 = Value(2.0); x2 = Value(0.0)
w1_durtulmus = Value(-3.0 + h) # <--- İŞTE BURADA h KADAR DÜRTTÜK!
w2 = Value(1.0)
b = Value(6.8813735870195432)

n_yeni = (x1*w1_durtulmus) + (x2*w2) + b
e_yeni = (2 * n_yeni).exp()
out_yeni = (e_yeni - 1) / (e_yeni + 1)

# Adım 3: Klasik meşhur türev formülü: (Yeni Sonuç - Eski Sonuç) / h
w1_sayisal_grad = (out_yeni.data - out_eski.data) / h

print("\nSayısal Türev (Numerical) Sonucu:")
print(f"w1 grad: {w1_sayisal_grad}")