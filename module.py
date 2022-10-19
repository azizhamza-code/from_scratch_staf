from ultragrad import scaler
import numpy as np

class Neuron:

    def __init__(self,n_in):

        self.weight = [scaler(np.random.randn())for _ in range(n_in)]
        self.b = scaler(0)

    def __call__(self,n_value):

        act = sum((w * value for w , value in zip(self.weight , n_value)) , self.b)
        out = act.tanh()

        return out

    def parameters(self):
        return self.weight + [self.b]

    
class Layer:

    def __init__(self,n_in,n_neuron):

        self.neuron = [Neuron(n_in) for _ in range(n_neuron)]

    def __call__(self,in_):

        out = [n(in_) for n in self.neuron]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        
        return [p for n in self.neuron for p in n.parameters()]

class MLP:

    def __init__(self , in_ , dim_layer):
        layer_dim = [in_]+dim_layer
        self.layers = [Layer(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim)-1)]
        self.in_ = in_


    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        param = []
        return [p for layer in self.layers for p in layer.parameters()]