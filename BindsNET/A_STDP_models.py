from typing import Optional, Union, Tuple, List, Sequence, Iterable
import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair
import torch.nn as nn
from torchvision import models
from bindsnet.learning.learning import PostPre, LearningRule
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes
from bindsnet.network.topology import Connection, LocalConnection
from A_STDP_learning import A_STDP_PostPre
from A_STDP_topology import WeightMemorizingConnection

class DiehlAndCook2015v2_A_STDP(Network):

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        inh: float = 120,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        pfault: float = 0,
        tau: float = 0.01,
        thresh: float = -52.0,
    ) -> None:
        
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.inh = inh
        self.dt = dt

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)

        prob = pfault

        weight_mask = torch.FloatTensor(self.n_inpt, self.n_neurons).uniform_() >= prob

        input_connection = WeightMemorizingConnection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            weight_mask=weight_mask,
            update_rule=A_STDP_PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            tau=tau,
        )
        self.add_connection(input_connection, source="X", target="Y")

        w = -self.inh * (
                torch.ones(self.n_neurons, self.n_neurons)
                - torch.diag(torch.ones(self.n_neurons))
        )
        recurrent_connection = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh,
            wmax=0,
        )
        self.add_connection(recurrent_connection, source="Y", target="Y")
