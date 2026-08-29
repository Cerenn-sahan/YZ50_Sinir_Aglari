import math
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        # Şimdi otomatik gradinat hesaplayan fonksiyonumuzu yazalım
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 
        self.label = label # Dışarıdan gelen etiketi objenin hafızasına kaydet

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad # += olarak ekliyoruz çünkü başşka yerde hesaplananları da aynı değişkenin üstüne eklesin üstüne yazmasın
            other.grad += out.grad #toplama işleminde yerel türev 1 gelir gelen türev ise zaaten çıktı.grad yani o.graddir.
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad #çarpma iileninin türevindde otherın data bilgisini alırız
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        # tanh'ın kendi türev kuralı (1 - t^2)
        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        
        return out

    def backward(self):

        topo = [] 
        visited = set() #Listeye zaten eklenmişle
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: #örneği c=a+b ise a ve b yi ilk olarak listeye ekler sonra c yi ekler ve böylece soldan sağa sıralanır.
                    build_topo(child)
                topo.append(v) #v şuanki nesne anlamına gelir
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo): #geriye doğru yayılım yaptığımız için snuçtan başlamaız lazım liste bileşenden sonuca doğru gidiyor,ters çevir.
            v._backward() #sonuç için listeyi childlera göre ayarlandığını bilir ve backwardı ona göre çalıştırır.

     
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label=f"{n.label} | data {n.data:.4f} | grad {n.grad:.4f}", shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot






#TEST

# 1. Nöronun Başlangıç Değişkenleri (Girdiler, Ağırlıklar ve Bias)
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

b = Value(6.8813735870195432, label='b')

# 2. İleri Yayılım (Forward Pass) - Nöronun matematiğini kuruyoruz
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'

x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

# Çıktımız
o = n.tanh(); o.label = 'o'

# 3. İŞTE BÜYÜ BURADA: Tek bir satırla tüm gradyanları otomatik hesapla!
o.backward()

# 4. Çizimi ekrana yansıt 
# (Önceki dosyandaki from graphviz import Digraph, trace ve draw_dot fonksiyonlarının 
# bu dosyada da yukarıda bir yerde ekli olduğunu varsayıyorum)
grafik = draw_dot(o)
grafik.render('otomatik_noron', view=True)

