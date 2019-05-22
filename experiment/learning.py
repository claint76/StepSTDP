import torch
import numpy as np
from typing import Union, Tuple, Optional, Sequence

from bindsnet.utils import im2col_indices
from bindsnet.network.topology import AbstractConnection, Connection, Conv2dConnection, LocallyConnectedConnection
from bindsnet.learning import LearningRule

#initialize the lookuptable function
def initial(weight:np.array,time:np.array,x:np.array):
    # get ladder weight and its corresponding time point
    '''
    weight=np.linspace(start,end,num)
    time=-np.log(weight)/trace_tc
    '''
    func=weight
    t = time
    cond=[]
    for i in range(0,t.size,1):
        if(i!=t.size-1):
            cond.append((x >= t[i]) & (x < t[i+1]))
        else:
            cond.append(x >= t[i])
    return cond,func

#Generalized lookuptable function
def lookuptable(weight:np.array,time:np.array,x:np.array):
    cond,func=initial(weight,time,x);
    #get the lookuptable result through numpy piecewise function
    x = np.piecewise(x,cond,func)
    return x


class Step_STDP(LearningRule):
    # language=rst
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. The pre-synaptic update is negative, while
    the post-synpatic update is positive.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0,Non_linear :bool =True, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``Step_STDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``Ladder_STDP`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs
        )
        self.Non_linear=Non_linear
        assert self.source.traces and self.target.traces, 'Both pre- and post-synaptic nodes must record spike traces'
        if isinstance(connection,(Connection, LocallyConnectedConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )
    def _connection_update(self, **kwargs)->None:
        #language=rst
        """
        Ladder_STDP learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
        """
        super().update()

        source_s = self.source.s.view(-1).float()
        source_x = self.source.x.view(-1)

        target_s = self.target.s.view(-1).float()
        target_x = self.target.x.view(-1)

        source_x_arr=source_x.numpy()
        target_x_arr=target_x.numpy()
        trace_tc=0.05

        #Non-linear cut
        if self.Non_linear:
            ls = [1 / pow(2, i) for i in range(0, 16)]
            weight = np.array(ls)
            time = -np.log(weight) / trace_tc
        #linear cut
        else:
            start = 1
            end = 1 / 16
            num = 16
            weight = np.linspace(start, end, num)
            time = -np.log(weight) / trace_tc
        source_x_arr=lookuptable(weight,time,x=source_x_arr)
        target_x_arr=lookuptable(weight,time,x=target_x_arr)

        source_x=torch.from_numpy(source_x_arr)
        target_x=torch.from_numpy(target_x_arr)

        # Pre-synaptic update.
        if self.nu[0]:
            self.connection.w -= self.nu[0] * torch.ger(source_s, target_x)

        # Post-synaptic update.
        if self.nu[1]:
            self.connection.w += self.nu[1] * torch.ger(source_x, target_s)

#The following was the derivation of the lookuptable function
'''
t = ln(w)/(-trace_tc)=-20ln(w)
w=np.linspace(1,1/16,16)
w
Out[12]: 
array([1.    , 0.9375, 0.875 , 0.8125, 0.75  , 0.6875, 0.625 , 0.5625,
       0.5   , 0.4375, 0.375 , 0.3125, 0.25  , 0.1875, 0.125 , 0.0625])
t=-20*np.log(w)
t
Out[19]: 
array([-0.        ,  1.29077042,  2.67062785,  4.1527873 ,  5.75364145,
        7.49386899,  9.40007258, 11.5072829 , 13.86294361, 16.53357146,
       19.61658506, 23.2630162 , 27.72588722, 33.47952867, 41.58883083,
       55.45177444])
using the following python script to get the cond
for i in range(0,t.size,1):
    if(i!=t.size-1):
        print("(x >= {}) & (x < {})".format(t[i],t[i+1]),",")
    else:
        print("x >= {}".format(t[i]))
'''
  
'''
    The following is an example of the lookuptable method.
    if start=1,end=1/16,num=16,trace_tc=0.05; weight=np.linspace(start,end,num)
    time=-np.log(weight)/trace_tc; The function lookuptable(weight,time,x) will 
    be equivalent to the following function lookuptable_ex(x):
//*********************************************************//
def lookuptable_ex(x:np.array):
    cond=[
        (x >= -0.0) & (x < 1.29077042),
        (x >= 1.29077042) & (x < 2.67062785),
        (x >= 2.67062785) & (x < 4.1527873),
        (x >= 4.1527873) & (x < 5.75364145),
        (x >= 5.75364145) & (x < 7.49386899),
        (x >= 7.49386899) & (x < 9.40007258),
        (x >= 9.40007258) & (x < 11.5072829),
        (x >= 11.5072829) & (x < 13.86294361),
        (x >= 13.86294361) & (x < 16.53357146),
        (x >= 16.53357146) & (x < 19.61658506),
        (x >= 19.61658506) & (x < 23.2630162),
        (x >= 23.2630162) & (x < 27.72588722),
        (x >= 27.72588722) & (x < 33.47952867),
        (x >= 33.47952867) & (x < 41.58883083),
        (x >= 41.58883083) & (x < 55.45177444),
        x >= 55.45177444
    ]
    func=[
        1., 0.9375, 0.875, 0.8125, 0.75, 0.6875, 0.625, 0.5625,
        0.5, 0.4375, 0.375, 0.3125, 0.25, 0.1875, 0.125, 0.0625
    ]
    x = np.piecewise(x,cond,func)
    return x
'''
