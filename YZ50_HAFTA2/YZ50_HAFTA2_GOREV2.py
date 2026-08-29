import math
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op =_op
        self.label = label


    def __repr__ (self):
        return f"Value(data={self.data})" #"Value(data=..." metnini ve yanına verinin içindeki sayıyı yazdırır.

    def __add__(self,other):
        out = Value(self.data + other.data, (self,other),'+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self,other),'*')
        return out

    def tanh(self):
        # tanh formülünün Python'daki matematiksel hesabı:
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    
        # Yeni bir Value üret, ebeveyni self olsun, işlemi 'tanh' olarak kaydet
        out = Value(t, (self, ), 'tanh') 
        return out

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

# h = 0.0001
# a = Value(2.0, label='a')
# b = Value(-3.0, label= 'b')
# c = Value(10.0, label= 'c')
# e = a * b; e. label ='e'
# d = e + c; d.label = 'd'
# f = Value(-2.0, label = 'f')
# L = d * f; L.label = 'L'
# L1 = L.data #L is a value

# a = Value(2.0 + h, label='a')
# b = Value(-3.0, label= 'b')
# c = Value(10.0, label= 'c')
# e = a * b; e. label ='e'
# d = e + c; d.label = 'd'
# f = Value(-2.0, label = 'f')
# L = d * f; L.label = 'L'
# L2 = L.data

# print((L2-L1)/h)


a = Value(2.0, label='a')
b = Value(-3.0, label= 'b')
c = Value(10.0, label= 'c')
e = a * b; e. label ='e'
d = e + c; d.label = 'd'
f = Value(-2.0, label = 'f')
L = d * f; L.label = 'L'

L.grad = 1.0
#Çarpma işleminde kural diğer sayının data'sını almaktır.
d.grad = f.data * L.grad
f.grad = d.data * L.grad

e.grad = d.grad
c.grad = d.grad

a.grad = b.data * e.grad
b.grad = a.data * e.grad


#2. Görevin 2. Kısmı

# Girdiler (Inputs)
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# Ağırlıklar (Weights)
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# Bias (Sabit değer) - Karpathy bu küsuratı bilerek seçer
b = Value(6.8813735870195432, label='b')

# 1. Aşama: Çarpmalar
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'

# 2. Aşama: Toplamalar
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

# 3. Aşama: tanh Fonksiyonu
o = n.tanh(); o.label = 'o'

# Nöron için Manuel Geriye Yayılım (Backpropagation)
o.grad = 1.0

# tanh işleminden geriye (n'ye) geçiş: 1 - o^2 kuralı
n.grad = (1.0 - o.data**2) * o.grad

# Toplama işleminden geriye (b ve x1w1x2w2'ye) geçiş: Gelen türev aynen aktarılır
b.grad = n.grad
x1w1x2w2.grad = n.grad

# Diğer toplama işleminden geriye (x1w1 ve x2w2'ye) geçiş: Gelen türev aynen aktarılır
x1w1.grad = x1w1x2w2.grad
x2w2.grad = x1w1x2w2.grad

# Çarpmalardan geriye geçiş (x1, w1, x2, w2 için diğerinin verisi kuralı):
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad

x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad

draw_dot(o).render('noron_grafigi', view=True)


