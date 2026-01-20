# Federated Learning with Exponential Compression through Schmidt-Decomposition-Guided Quantum Circuits
*written with the help of DeepSeek AI*

![License](https://img.shields.io/badge/license-MIT-green)
[![DOI](https://img.shields.io/badge/doi-10.5281/zenodo.18095056-blue)](https://doi.org/10.5281/zenodo.18095056) 



This repository contains the implementation for the paper **"Federated Learning with Exponential Compression through Schmidt-Decomposition-Guided Quantum Circuits"** (accepted to IEEE CYBCONF 2026). The framework achieves exponential communication compression in federated learning using quantum circuits optimized via Schmidt decomposition.

## 🎯 Key Features

- **Exponential Compression**: Compresses $2^n$-dimensional data to $n$-dimensional features ($2^n \to n$)
- **Schmidt-Decomposition-Guided**: Quantum circuit architecture determined by data's tensor network structure
- **Federated Learning**: Multi-client setup with minimal communication overhead
- **Quantum-Classical Hybrid**: Client-side quantum compression with server-side classical training
- **Benchmark Support**: Iris, Digits, CIFAR10, and 15+ other datasets

## 📁 Repository Structure

```text
├── data_loaders.py              # Dataset loading and preprocessing  
├── federated_simulation.py      # Main federated learning orchestration  
├── schmidt_decomposition.py     # Schmidt decomposition implementation  
├── schmidt_circuit_optimization.py  # Quantum circuit optimization  
├── quantum_circuit_model.py     # Quantum circuit model definition  
├── plotting.py                  # Visualization and result plotting  
├── main.tex                     # Conference paper (IEEE format)  
└── requirements.txt             # Python dependencies  
```

### File Descriptions

| File | Purpose |
|------|---------|
| `data_loaders.py` | Loads and preprocesses datasets including Iris, Digits, CIFAR10, MNIST, etc. |
| `federated_simulation.py` | Main orchestrator for federated learning with quantum clients and classical server |
| `schmidt_decomposition.py` | Implements Schmidt decomposition for tensor network analysis |
| `schmidt_circuit_optimization.py` | Optimizes quantum circuits to match Schmidt decomposition targets |
| `quantum_circuit_model.py` | Defines the Schmidt-based quantum circuit using PennyLane |
| `plotting.py` | Generates publication-quality plots for results |
| `main.tex` | Conference paper describing the methodology and results |


## 🏃‍♂️ Quick Start

### Run Federated Learning on Iris Dataset
```python
from federated_simulation import FederatedOrchestrator

# Configure and run
orchestrator = FederatedOrchestrator(
    dataset_name="iris",
    n_clients=5,
    schmidt_threshold=0.3, # Lower threshold for complex data, 
    device='cuda'  # or 'cpu'
)

results = orchestrator.run_federated_learning(method='max')
```

### Command Line Execution
```bash
# Direct execution from federated_simulation.py
python federated_simulation.py
```

## 📊 Available Datasets

The framework supports multiple datasets including:

| Dataset | Qubits | Original Dim | Compressed Dim | Compression Ratio |
|---------|--------|--------------|----------------|-------------------|
| Iris | 2 | 4 | 2 | 2.0× |
| Digits | 6 | 64 | 6 | 10.7× |
| CIFAR10 | 10 | 1024 | 10 | 102.4× |
| MNIST | 9 | 512 | 9 | 56.9× |
| 20 Newsgroups | 10 | 1024 | 10 | 102.4× |

## 🔬 Methodology

### Four-Phase Protocol
1. **Local Schmidt Analysis**: Each client computes Schmidt decomposition of local data
2. **Global k Determination**: Server aggregates local k values to determine global circuit complexity
3. **Local Quantum Compression**: Clients optimize quantum circuits and compress data
4. **Centralized Classical Training**: Server trains classical model on aggregated compressed features

### Quantum Circuit Architecture
- **Schmidt-Guided**: Circuit terms determined by data's Schmidt spectrum
- **Parameterized Unitaries**: Each term implemented as tensor product of single-qubit rotations
- **Ancilla Encoding**: $\lceil \log_2 k \rceil$ ancilla qubits for k-term superposition

## 📈  Key Findings
- **Exponential Compression**: Achieved up to 102.4× compression on CIFAR10
- **Data-Aware Architecture**: Schmidt decomposition provides optimal low-rank approximation
- **Communication Efficiency**: Transmits only $n$ features instead of $2^n$ per sample
- **Privacy Enhancement**: Nonlinear quantum transformations complicate data reconstruction

## 🎨 Visualization

The framework generates publication-ready figures:
- Local Schmidt rank distribution
- Training convergence curves
- Per-class accuracy plots
- Compression ratio visualization

Example generation:
```python
from plotting import plot_results
plot_results(orchestrator, results)
```

## 🧪 Advanced Usage

### Custom Dataset Integration
```python
# Add custom dataset to data_loaders.py
elif name == "my_dataset":
    X, y = load_my_custom_data()
    # Ensure X is normalized and shaped appropriately
```

### Parameter Tuning
```python
# Adjust Schmidt threshold for different datasets
schmidt_thresholds = {
    "iris": 0.3,
    "digits": 0.3,
    "cifar10": 0.3
}

# Modify quantum circuit complexity
n_ancilla_layers = 2  # Increase for more expressive circuits
```

### Different k Determination Methods
```python
# Available methods: 'max', 'avg', 'percentile'
results = orchestrator.run_federated_learning(method='percentile')
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

Ammar Daskin - [adaskin25@gmail.com](mailto:adaskin25@gmail.com)

