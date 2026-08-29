class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        #objeyi oluşturan önceki onbeleri tutacak değişkenimiz
        self._prev = set(_children)
        #aralarında hangi işaret olduğunu tutar bu arada = eşitliğin karşı tarafıdakiler paramtre olarak bu fonksiyona gelen girdilerdir.
        self._op =_op
        self.label = label


    def __repr__ (self):
        return f"Value(data={self.data})" #"Value(data=..." metnini ve yanına verinin içindeki sayıyı yazdırır.

    #bu fonksiyon iki bileşenden oluşan valueları üretmemiizi  ve parametrelerini vermemizi sağlayacak
    #döndüreceği tip value objecti tipinden olacağı için out diiye boş bir değişkene eşitleyip returnleyeceğim
    def __add__(self,other):
        out = Value(self.data + other.data, (self,other),'+')
        #işlemi yapan iki aktör self ve other'dır.
        return out

    #çarpma işlemi için olanı
    def __mul__(self, other):
        out = Value(self.data * other.data, (self,other),'*')
        return out

from graphviz import Digraph

def trace(root):
    # Ağacı kökten başlayarak geriye doğru tarar, düğümleri (nodes) ve kenarları (edges) bulur.
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
    # Bulunan düğüm ve kenarları kullanarak görsel bir harita (Digraph) çizer.
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR: Left to Right (Soldan Sağa çiz)
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # Kutuların içine yazılacak metni (label ve data) belirliyoruz.
        # Eğer henüz grad eklemediysen " | grad {n.grad:.4f}" kısmını silebilirsin.
        dot.node(name=uid, label=f"{n.label} | data {n.data:.4f} | grad {n.grad:.4f}", shape='record')
        
        if n._op:
            # Eğer bu bir işlem sonucu oluştuysa, araya işlemi gösteren yuvarlak bir düğüm ekle
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        # Birbirine bağlı nesneler arasına ok çiz
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

# Test Aşaması
#Her yeni Value, kendisini üreten Value'ları ve hangi işlemden çıktığını saklıyor.
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = a + b
c.label = 'c'

# Çizimi oluştur
grafik = draw_dot(c)
grafik.render('hesaplama_grafigi', view=True)



