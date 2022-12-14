from abc import ABC
from typing import Union, Optional, Sequence
import torch
import numpy as np
from bindsnet.network.nodes import SRM0Nodes
from bindsnet.utils import im2col_indices
from bindsnet.network.topology import (
    AbstractConnection,
    Connection,
    Conv2dConnection,
    LocalConnection,
)
from bindsnet.learning.learning import LearningRule
from A_STDP_topology import WeightMemorizingConnection

class A_STDP_PostPre(LearningRule):

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:

        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert self.source.traces, "Pre-synaptic nodes must record spike traces."
        assert (
            connection.wmin != -np.inf and connection.wmax != np.inf
        ), "Connection must define finite wmin and wmax."

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        self.tau = kwargs["tau"],

        if isinstance(connection, (WeightMemorizingConnection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )
        
        
    def _connection_update(self, **kwargs) -> None:

        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)
        
        # Collect the sum of healthy and total weights for each neuron in Y layer
        weight_original = self.connection.w0
        weight_mask = self.connection.weight_mask
        sum_total = weight_original.sum(0)
        sum_healthy = (weight_original * weight_mask).sum(0)

        n_input = weight_original.size()[0]
        
        # q: PR raising ratio. In astrocyte model, if a healthy PR is x0 before fault
        # it should become q*x0 after self repair
        if self.connection.q is None:
            self.connection.q = 1.0 / (sum_healthy/sum_total)
            q = self.connection.q
        else:
            q = self.connection.q

        tau = torch.tensor(self.tau, device=torch.device(q.device))    

        update = 0

        # Pre-synaptic update.
        if self.nu[0]:
            outer_product = self.reduction(torch.bmm(source_s, target_x), dim=0)
            update -= self.nu[0] * outer_product

        # Post-synaptic update.
        if self.nu[1]:
            outer_product = self.reduction(torch.bmm(source_x, target_s), dim=0)
            update += self.nu[1] * outer_product * (q * weight_original * weight_mask - self.connection.w) / tau

        self.connection.w += update

        super().update()
