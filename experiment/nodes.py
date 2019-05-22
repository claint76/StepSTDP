import torch
from typing import Iterable, Optional, Union
from bindsnet.network.nodes import Nodes,Input

class DafengNodes(Nodes):
    #language=rst
    """
    Based on Diehl&Cook 2015
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds(modified for Dafeng 2019)
    """
    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, traces: bool = False,
                 sum_input: bool = False,
                 thresh: Union[float, torch.Tensor] = -52.0, rest: Union[float, torch.Tensor] = -65.0,
                 reset: Union[float, torch.Tensor] = -65.0, refrac: Union[int, torch.Tensor] = 5,
                 decay: Union[float, torch.Tensor] = 1e-2, theta_plus: Union[float, torch.Tensor] = 0.05,
                 theta_decay: Union[float, torch.Tensor] = 1e-7, lbound: float = None, one_spike: bool = True) -> None:
        # language=rst
        """
        Instantiates a layer of Dafeng2019 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        :param one_spike: Whether to allow only one spike per timestep.
        """
        super().__init__(n, shape, traces, sum_input)
        # Rest voltage.
        if isinstance(rest, float):
            self.rest = torch.tensor(rest)
        else:
            self.rest = rest

        # Post-spike reset voltage.
        if isinstance(reset, float):
            self.reset = torch.tensor(reset)
        else:
            self.reset = reset

        # Spike threshold voltage.
        if isinstance(thresh, float):
            self.thresh = torch.tensor(thresh)
        else:
            self.thresh = thresh

        # Post-spike refractory period.
        if isinstance(refrac, float):
            self.refrac = torch.tensor(refrac)
        else:
            self.refrac = refrac

        # Rate of decay of neuron voltage.
        if isinstance(decay, float):
            self.decay = torch.tensor(decay)
        else:
            self.decay = decay

        # Constant threshold increase on spike.
        if isinstance(theta_plus, float):
            self.theta_plus = torch.tensor(theta_plus)
        else:
            self.theta_plus = theta_plus

        # Rate of decay of adaptive thresholds.
        if isinstance(theta_decay, float):
            self.theta_decay = torch.tensor(theta_decay)
        else:
            self.theta_decay = theta_decay

        # Lower bound of voltage.
        self.lbound = lbound

        # One spike per timestep.
        self.one_spike = one_spike

        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.theta = torch.zeros(self.shape)  # Adaptive thresholds.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        On each simulation step, set the outputs of the population equal to the inputs.

        :param x: Inputs to the layer.
        """
        # Decay voltages and adaptive thresholds.
        self.v -= self.dt * self.decay * (self.v - self.rest)
        self.theta -= self.dt * self.theta_decay * self.theta

        # Integrate inputs.
        self.v += (self.refrac_count == 0).float() * x

        # Decrement refractory counters.
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)

        # Check for spiking neurons.
        self.s = (self.v >= self.thresh + self.theta)

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        self.theta += self.theta_plus * self.s.float()

        # Choose only a single neuron to spike.
        if self.one_spike:
            if self.s.any():
                s = torch.zeros(self.n).byte()
                s[torch.multinomial(self.s.float().view(-1), 1)] = 1
                self.s = s.view(self.shape)

        # voltage clipping to lowerbound
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        if self.traces:
            # Decay and record time intervals.
            self.x += self.dt
            self.x.masked_fill_(self.s, 0)

        if self.sum_input:
            # Add current input to running sum.
            self.summed += x.float()

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

class DafengInput(Input):

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None,
                 traces: bool = False, sum_input: bool = False) -> None:
        # language=rst
        """
        Instantiates a layer of input neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying time intervals.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__(n, shape, traces, sum_input)

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        On each simulation step, set the spikes of the population equal to the inputs.

        :param x: Inputs to the layer.
        """
        # Set spike occurrences to input values.
        self.s = x.byte()
        if self.traces:
            # Decay and record time intervals.
            self.x += self.dt
            self.x.masked_fill_(self.s, 0)

        if self.sum_input:
            # Add current input to running sum.
            self.summed += x.float()


    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()