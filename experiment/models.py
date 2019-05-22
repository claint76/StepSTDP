import torch
import numpy as np
from torch.nn.modules.utils import _pair
from scipy.spatial.distance import euclidean
from typing import Optional, Union, Tuple, List, Sequence

from bindsnet.network import Network
from bindsnet.network.topology import Connection, LocallyConnectedConnection
from bindsnet.network.nodes import Input, RealInput, LIFNodes
from experiment.nodes import DafengNodes,DafengInput
from experiment.learning import Step_STDP

class Dafeng2019(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from Diehl&Cook 2015, with
    a new STDP training method from Dafeng2019
    """
    def __init__(self, n_inpt: int, n_neurons: int = 100, exc: float = 22.5, inh: float = 17.5, dt: float = 1.0,
                 nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2), wmin: float = 0.0, wmax: float = 1.0,
                 norm: float = 78.4, theta_plus: float = 0.05, theta_decay: float = 1e-7,
                 Non_linear: bool= True,
                 X_Ae_decay: Optional[float] = None, Ae_Ai_decay: Optional[float] = None,
                 Ai_Ae_decay: Optional[float] = None
                 ) -> None:
        # language=rst
        """
        Constructor for class ``Dafeng2019``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization constant.
        :param theta_plus: On-spike increment of ``DafengNodes`` membrane threshold potential.
        :param theta_decay: Time constant of ``DafengNodes`` threshold potential decay.
        :param X_Ae_decay: Decay of activation of connection from input to excitatory neurons.
        :param Ae_Ai_decay: Decay of activation of connection from excitatory to inhibitory neurons.
        :param Ai_Ae_decay: Decay of activation of connection from inhibitory to excitatory neurons.
        """
        super().__init__(dt=dt)

        self.n_input = n_inpt
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        self.add_layer(DafengInput(n=self.n_input,traces=True),name='X')
        self.add_layer(DafengNodes(n=self.n_neurons,traces=True,rest=-65.0,reset=-60.0,thresh=-52.0,
                                   refrac=5,decay=1e-2,theta_plus=theta_plus,theta_decay
                                   =theta_decay),name='Ae')
        self.add_layer(LIFNodes(n=self.n_neurons, traces=False, rest=-60.0, reset=-45.0, thresh=-40.0,
                                decay=1e-1,refrac=2, trace_tc=5e-2),name='Ai')

        w = 0.3*torch.rand(self.n_input,self.n_neurons)
        self.add_connection(Connection(source=self.layers['X'],target=self.layers['Ae'],w=w,update_rule
                                       =Step_STDP,Non_linear=Non_linear,nu=nu,wmin=wmin,wmax=wmax,norm=norm, decay=X_Ae_decay),
                            source='X',target='Ae')
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        self.add_connection(Connection(source=self.layers['Ae'],target=self.layers['Ai'],w=w,wmin=0,wmax=self.exc,
                                       decay=Ai_Ae_decay),
                            source='Ae', target='Ai')
        w = -self.inh * (torch.ones(self.n_neurons, self.n_neurons)-torch.diag(torch.ones(self.n_neurons)))
        self.add_connection(Connection(source=self.layers['Ai'],target=self.layers['Ae'],w=w,wmin=-self.inh,
                                       wmax=0,decay=Ai_Ae_decay),
                            source='Ai',target='Ae')
