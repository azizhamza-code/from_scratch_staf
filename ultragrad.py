import math
import numpy as np

class scaler:

    def __init__(self,value,label='',child = set(),op='' ):
        self.value = value
        self.child = child
        self.op = op
        self.grad = 0
        self._backprobe =  lambda :None
        self.label=label

    def __repr__(self) -> str:
        return f'(scaler {self.label}:{self.value})'

    
    def __add__(self,other):
        other = other if isinstance(other, scaler) else scaler(other)
        out = scaler(self.value + other.value , child = (self,other),op = '+')

        def _backprobe():

            self.grad +=  out.grad
            other.grad += out.grad

        out._backprobe = _backprobe

        return out

    def tanh(self):

        out = scaler(math.tanh(self.value) , child = (self,), op = 'tanh' , label = 'tanh')

        def _backprobe(): 
            self.grad += out.grad * (1-out.value **2)

        out._backprobe = _backprobe

        return out


    def __mul__(self,other):
        other = other if isinstance(other, scaler) else scaler(other)
        out = scaler(self.value * other.value  , child = (self,other) , op = "*")

        def _backprobe():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out._backprobe = _backprobe

        return out

    def backpropev1(self):
        result = []
        def node(c):
            if c.child:
                result.append(c)
                if len(c.child)>1:
                    node(c.child[0])
                    node(c.child[1])
                else:
                    node(c.child[0])

        node(self)
        for node in (result):
            node._backprobe()


    def backpropev2(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.child:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backprobe()

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = scaler(self.value**other, child = (self,),label =  f'**{other}')

        def _backprobe():
            self.grad += other * (self.value ** (other - 1)) * out.grad
        out._backprobe = _backprobe

        return out



    def __radd__(self,other):
        return  self+other

    def __rmul__(self,other):
        return  self*other

    def __neg__(self): # -self
        return self * -1

    def __sub__(self,other):
        return self + (- other)

    def __rsub__(self,other):
        return self - other


