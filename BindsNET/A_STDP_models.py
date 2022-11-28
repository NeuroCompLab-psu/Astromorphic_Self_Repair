from typing import Optional, Union, Tuple, List, Sequence, Iterable

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair
import torch.nn as nn
from torchvision import models

from A_STDP_learning import PostPre, WeightSumDependentDivergencePostPre, LearningRule
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes
from A_STDP_topology import Connection, WeightMemorizingConnection, LocalConnection

class DiehlAndCook2015v2(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing
    the inhibitory layer and replacing it with a recurrent inhibitory connection in the
    output layer (what used to be the excitatory layer).
    """

    def __init__(
            self,
            n_inpt: int,
            n_neurons: int = 100,
            inh: float = 17.5,
            dt: float = 1.0,
            nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
            reduction: Optional[callable] = None,
            wmin: Optional[float] = 0.0,
            wmax: Optional[float] = 1.0,
            norm: float = 78.4,
            theta_plus: float = 0.05,
            tc_theta_decay: float = 1e7,
            inpt_shape: Optional[Iterable[int]] = None,
            **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
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
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)

        # TODO: include errored wheights

        # creating the indexes where weight should be clamped to 0
        # ind = np.arange(self.n_inpt * self.n_neurons)

        prob = kwargs.get("p", 0.0)

        weight_mask = torch.FloatTensor(self.n_inpt, self.n_neurons).uniform_() >= prob

        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            weight_mask=weight_mask,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
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

class DiehlAndCook2015v2_CIFAR(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing
    the inhibitory layer and replacing it with a recurrent inhibitory connection in the
    output layer (what used to be the excitatory layer).
    """

    def __init__(
            self,
            n_inpt: int,
            n_neurons: int = 100,
            inh: float = 17.5,
            dt: float = 1.0,
            nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
            reduction: Optional[callable] = None,
            wmin: Optional[float] = 0.0,
            wmax: Optional[float] = 1.0,
            norm: float = 78.4,
            theta_plus: float = 0.05,
            tc_theta_decay: float = 1e7,
            inpt_shape: Optional[Iterable[int]] = None,
            **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
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
            thresh=kwargs.get("thresh", -52.0),
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = kwargs.get("w_initial", 0.3) * torch.rand(self.n_inpt, self.n_neurons)

        # TODO: include errored wheights

        # creating the indexes where weight should be clamped to 0
        # ind = np.arange(self.n_inpt * self.n_neurons)

        prob = kwargs.get("p", 0.0)

        learning_rule = eval(kwargs.get('learning_rule', 'PostPre'))

        assert issubclass(learning_rule, LearningRule), 'Learning Rule Not Present'

        weight_mask = torch.FloatTensor(self.n_inpt, self.n_neurons).uniform_() >= prob

        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            weight_mask=weight_mask,
            update_rule=learning_rule,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            alpha=kwargs.get("alpha", 1.0),
            sigma=kwargs.get("sigma", 1.0),
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

class DiehlAndCook2015v9_Weight_Sum_Dependent_Divergence(Network):

    def __init__(
            self,
            n_inpt: int,
            n_neurons: int = 100,
            inh: float = 17.5,
            dt: float = 1.0,
            nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
            reduction: Optional[callable] = None,
            wmin: Optional[float] = 0.0,
            wmax: Optional[float] = 1.0,
            norm: Optional[float] = 78.4,
            theta_plus: float = 0.05,
            tc_theta_decay: float = 1e7,
            inpt_shape: Optional[Iterable[int]] = None,
            **kwargs,
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
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)

        

        prob = kwargs.get("p", 0.0)

        weight_mask = torch.FloatTensor(self.n_inpt, self.n_neurons).uniform_() >= prob

        input_connection = WeightMemorizingConnection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            weight_mask=weight_mask,
            update_rule=WeightSumDependentDivergencePostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            tau=kwargs.get("tau", 16.0),
            kdiv=kwargs.get("kdiv", 1.0),
            bdiv=kwargs.get("bdiv", 1.0)
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

class DiehlAndCook2015v9_Weight_Sum_Dependent_Divergence_CIFAR(Network):

    def __init__(
            self,
            n_inpt: int,
            n_neurons: int = 100,
            inh: float = 17.5,
            dt: float = 1.0,
            nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
            reduction: Optional[callable] = None,
            wmin: Optional[float] = 0.0,
            wmax: Optional[float] = 1.0,
            norm: float = 78.4,
            theta_plus: float = 0.05,
            tc_theta_decay: float = 1e7,
            inpt_shape: Optional[Iterable[int]] = None,
            **kwargs,
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
            thresh=kwargs.get("thresh", -52.0),
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = kwargs.get("w_initial", 0.3) * torch.rand(self.n_inpt, self.n_neurons)

        # TODO: include errored wheights

        # creating the indexes where weight should be clamped to 0
        # ind = np.arange(self.n_inpt * self.n_neurons)

        prob = kwargs.get("p", 0.0)

        learning_rule = eval(kwargs.get('learning_rule', 'PostPre'))

        assert issubclass(learning_rule, LearningRule), 'Learning Rule Not Present'

        weight_mask = torch.FloatTensor(self.n_inpt, self.n_neurons).uniform_() >= prob

        input_connection = WeightMemorizingConnection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            weight_mask=weight_mask,
            update_rule=learning_rule,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            tau=kwargs.get("tau", 16.0),
            kdiv=kwargs.get("kdiv", 1.0),
            bdiv=kwargs.get("bdiv", 1.0)
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