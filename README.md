# StepSTDP
StepSTDP is a new improved STDP algorithm for SNN training, which has good feasibility in hardware implementation.
## Requirements
- Python 3.6
- BindsNET package installed (To install the package, visit [BindsNET](https://github.com/Hananel-Hazan/bindsnet))
## Setting things up
Change the current directory to BindsNET installation directory.
```
cd bindsnet-master
```
Copy the "experiment" folder to this directory.

## Getting started
To run an instance with StepSTDP algorithm, you can replace the following Python statements in bindsnet-master\examples\mnist\eth_mnist.py
```
from bindsnet.models import DiehlAndCook2015
network = DiehlAndCook2015(n_inpt=784, n_neurons=n_neurons, exc=exc, inh=inh, dt=dt, norm=78.4, theta_plus=1)
```
to
```
from experiment.models import Dafeng 2019
network = Dafeng2019(n_inpt=784, n_neurons=n_neurons, exc=exc, inh=inh, dt=dt, norm=78.4, theta_plus=1,Non_linear=True)
```


