from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Sequence
import numpy as np
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from bindsnet.network.nodes import Nodes
from bindsnet.network.topology import AbstractConnection

class WeightMemorizingConnection(AbstractConnection):

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:

        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        weight_mask = kwargs.get("weight_mask", None)
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)

        if weight_mask is None:
            weight_mask = torch.full_like(w , True)

        w *= weight_mask

        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(kwargs.get("b", torch.zeros(target.n)), requires_grad=False)
        self.weight_mask = Parameter(weight_mask , requires_grad = False)

        self.q = None

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                 decaying spike activation).
        """
        # Compute multiplication of spike activations by weights and add bias.
        self.w *= self.weight_mask
        post = s.float().view(s.size(0), -1) @ self.w + self.b
        return post.view(s.size(0), *self.target.shape)

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)
        self.w *= self.weight_mask

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum
        
        # return

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

    def weight_snapshot(self) -> None:
        # language=rst
        """
        Store current weight as w0 for future use
        """
        self.w0 = torch.clone(self.w)
