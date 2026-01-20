import pennylane as qml
import numpy as np
from math import ceil, log2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from schmidt_decomposition import SchmidtDecomposition


# -----------------------------------------------------------------------------
# 2) Quantum circuit module - FIXED TENSOR HANDLING
# -----------------------------------------------------------------------------
class SchmidtQuantumCircuit(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        n_terms: int,
        init_terms: list = None,
        n_ancilla_layers: int = 2,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_terms = n_terms
        self.n_ancilla = ceil(log2(n_terms)) if n_terms > 1 else 0
        self.n_ancilla_layers = n_ancilla_layers

        # Ancilla rotation parameters
        if self.n_ancilla > 0:
            num_ancilla_params = self.n_ancilla + max(0, self.n_ancilla - 1)
            self.ancilla_params = nn.Parameter(
                torch.randn(self.n_ancilla_layers * num_ancilla_params)
            )
        else:
            self.ancilla_params = None

        # Data-qubit parameters - now with optional Schmidt initialization
        self.unitary_params = nn.ParameterList()

        # # Initialize from Schmidt decomposition if provided
        # # We did not observe any performance improvement from this!
        # # maybe the angle computations for also coefficients would make better!
        # if init_terms:

        #     for k in range(min(n_terms, len(init_terms))):
        #         term_angles = []
        #         for vector in init_terms[k]:
        #             angle = SchmidtDecomposition.vector_to_angle(vector)
        #             term_angles.append(angle)
        #         # Initialize parameter with Schmidt-derived angles
        #         self.unitary_params.append(
        #             nn.Parameter(torch.tensor(term_angles, dtype=torch.float32)))

        #     # Initialize remaining terms randomly
        #     for _ in range(n_terms - len(init_terms)):
        #         print("herrrrr")
        #         self.unitary_params.append(
        #             nn.Parameter(torch.randn(self.n_qubits)))
        # else:
        # All random initialization
        for _ in range(n_terms):
            self.unitary_params.append(nn.Parameter(torch.randn(self.n_qubits)))

        self.dev = qml.device("default.qubit", wires=self.n_ancilla + self.n_qubits)
        self.qnode = self.create_qnode()

    def create_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(features,output_type='expectations'):
            # Ancilla preparation
            if self.n_ancilla > 0:
                p = 0
                for l in range(self.n_ancilla_layers):
                    for i in range(self.n_ancilla):
                        qml.RY(self.ancilla_params[p], wires=i)
                        p += 1
                    for i in range(self.n_ancilla - 1):
                        qml.CRY(self.ancilla_params[p], wires=[i, i + 1])
                        p += 1

            # Embed input data
            data_wires = list(range(self.n_ancilla, self.n_ancilla + self.n_qubits))
            qml.AmplitudeEmbedding(features=features, wires=data_wires, normalize=True)

            # Apply parameterized unitaries
            ctrl_qubits = list(range(self.n_ancilla))
            if self.n_ancilla == 0:
                for q in range(self.n_qubits):
                    qml.RY(self.unitary_params[0][q], wires=q + self.n_ancilla)
            else:
                for k in range(self.n_terms):
                    ctrl_bin_values = bin(k)[2:].zfill(self.n_ancilla)
                    ctrl_int_values = [int(a) for a in list(ctrl_bin_values)]
                    for q in range(self.n_qubits):
                        qml.ctrl(qml.RY, ctrl_qubits, control_values=ctrl_int_values)(
                            self.unitary_params[k][q], wires=q + self.n_ancilla
                        )
            if output_type == 'expectations':
                return [qml.expval(qml.PauliZ(w)) for w in data_wires]
            else:  # 'probs'
                return qml.probs(wires=data_wires)

        return circuit

    def forward(self, x: torch.Tensor, output_type='expectations'):
        # x: [batch, features]
        # Process each sample individually
        ### return x.float() # for only classical
        q_out = self.qnode(x, output_type=output_type)
        if isinstance(q_out, list):
            q_out = torch.stack(q_out, dim=-1).to(torch.float32)
        return q_out

# -----------------------------------------------------------------------------
# End of quantum circuit module
# -----------------------------------------------------------------------------