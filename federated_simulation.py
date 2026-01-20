# federated_simulation.py
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import your existing modules
from data_loaders import load_dataset
from schmidt_decomposition import SchmidtDecomposition
from schmidt_circuit_optimization import optimize_schmidt_circuit
from quantum_circuit_model import SchmidtQuantumCircuit


class QuantumFLClient:
    """Client for Quantum Federated Learning"""
    
    def __init__(self, client_id: int, X_train, y_train, X_test, y_test, 
                 schmidt_threshold=0.1, device='cpu'):
        """
        Initialize a client with local data.
        
        Args:
            client_id: Unique client identifier
            X_train, y_train: Local training data
            X_test, y_test: Local test data
            schmidt_threshold: Threshold for Schmidt decomposition
            device: 'cpu' or 'cuda'
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.schmidt_threshold = schmidt_threshold
        self.device = device
        
        # Will be set during federated learning
        self.n_qubits = int(np.log2(X_train.shape[1]))
        self.circuit = None
        self.optimized_params = None
        self.local_k = None
        
    def compute_local_k(self) -> int:
        """Compute local k value from Schmidt decomposition of mean vector"""
        # Compute normalized mean vector
        mean_vec = self.X_train.mean(axis=0)
        mean_vec = mean_vec / np.linalg.norm(mean_vec)
        
        # Perform Schmidt decomposition
        sd = SchmidtDecomposition(threshold=self.schmidt_threshold)
        terms, coeffs = sd.flatten_decomposition(mean_vec)
        
        # Determine k: number of terms above threshold
        self.local_k = len(terms)
        
        # If no terms found, use at least 1
        if self.local_k == 0:
            self.local_k = 1
            
        return self.local_k
    
    def build_circuit(self, global_k: int):
        """Build quantum circuit with specified number of terms"""
        self.global_k = global_k
        
        # Get mean vector for initialization
        mean_vec = self.X_train.mean(axis=0)
        mean_vec = mean_vec / np.linalg.norm(mean_vec)
        
        # Get terms from Schmidt decomposition
        sd = SchmidtDecomposition(threshold=self.schmidt_threshold)
        terms, _ = sd.flatten_decomposition(mean_vec)
        
        # Build circuit with global_k terms (use min(global_k, available_terms))
        n_terms = min(global_k, len(terms) if len(terms) > 0 else 1)
        
        # Use the circuit from schmidt_circuit_optimization
        self.circuit = SchmidtQuantumCircuit(
            n_qubits=self.n_qubits,
            n_terms=n_terms,
            init_terms=terms[:n_terms] if terms else None,
            n_ancilla_layers=2
        ).to(self.device)
        
        return self.circuit
    
    def optimize_circuit(self, epochs=200, lr=0.01):
        """Optimize quantum circuit parameters locally"""
        if self.circuit is None:
            raise ValueError("Circuit not built. Call build_circuit first.")
        
        # Use a sample of data for optimization (to reduce computation)
        sample_size = min(100, len(self.X_train))
        indices = np.random.choice(len(self.X_train), sample_size, replace=False)
        X_sample = self.X_train[indices]
        
        # Optimize using the existing function
        print(f"Client {self.client_id}: Optimizing circuit with {self.circuit.n_terms} terms...")
        circuit_model, best_loss, losses = optimize_schmidt_circuit(
            data=X_sample,
            schmidt_threshold=self.schmidt_threshold,
            epochs=epochs,
            lr=lr,
            single_vector=False,
            n_terms=self.circuit.n_terms
        )

        self.circuit = circuit_model.to(self.device)
        # Store optimized parameters
        self.optimized_params = {name: param.clone() for name, param in self.circuit.named_parameters()}
       
        return circuit_model,best_loss, losses
    
    def compress_data(self, X_data=None):
        """Compress data using optimized quantum circuit"""
        if self.circuit is None:
            raise ValueError("Circuit not optimized. Call optimize_circuit first.")
        
        if X_data is None:
            X_data = self.X_train
            
        # Use expectation values as compressed features
        compressed_features = []
        
        with torch.no_grad():
            self.circuit.eval()
            for x in X_data:
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                # Get expectation values (n_qubits dimensional)
                output = self.circuit(x_tensor, output_type='expectations')
                
                if isinstance(output, list):
                    output = torch.stack(output, dim=-1)
                    
                compressed_features.append(output.cpu().numpy().flatten())
        
        return np.array(compressed_features)


class FLServer:
    """Server for Quantum Federated Learning"""
    
    def __init__(self, n_qubits, n_classes, device='cpu'):
        """
        Initialize the server.
        
        Args:
            n_qubits: Number of qubits (determines compressed feature dimension)
            n_classes: Number of output classes
            device: 'cpu' or 'cuda'
        """
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.device = device
        self.global_k = None
        self.classical_model = None
        self.clients = []
        
    def register_clients(self, clients: List[QuantumFLClient]):
        """Register clients with the server"""
        self.clients = clients
    
    def collect_local_k(self) -> List[int]:
        """Collect local k values from all clients"""
        local_ks = []
        for client in self.clients:
            k = client.compute_local_k()
            local_ks.append(k)
            print(f"Client {client.client_id}: local_k = {k}")
        return local_ks
    
    def determine_global_k(self, local_ks: List[int], method='max'):
        """Determine global k from local k values"""
        if method == 'max':
            self.global_k = max(local_ks)
        elif method == 'avg':
            self.global_k = int(np.ceil(np.mean(local_ks)))
        elif method == 'percentile':
            # 75th percentile
            self.global_k = int(np.ceil(np.percentile(local_ks, 75)))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Global k determined ({method}): {self.global_k}")
        return self.global_k
    
    def broadcast_global_k(self):
        """Broadcast global k to all clients"""
        if self.global_k is None:
            raise ValueError("Global k not determined. Call determine_global_k first.")
        
        for client in self.clients:
            client.build_circuit(self.global_k)
            
    def collect_compressed_data(self, epochs=200, lr=0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Collect compressed data from all clients"""
        all_compressed = []
        all_labels = []
        
        for client in self.clients:
            # Each client optimizes their circuit
            print(f"Client {client.client_id} optimizing circuit...")
            model_circuit, best_loss, losses = client.optimize_circuit(epochs=epochs, lr=lr)
            client.circuit = model_circuit
   
            print(f"Client {client.client_id} compressing data...")
            # Compress local data
            compressed = client.compress_data(client.X_train)
            all_compressed.append(compressed)
            all_labels.append(client.y_train)
            
        # Concatenate all data
        X_compressed = np.vstack(all_compressed)
        y_compressed = np.hstack(all_labels)
        
        return X_compressed, y_compressed
    
    def train_classical_model(self, X_train, y_train, hidden_dims=[64, 32], 
                              epochs=200, lr=0.01, batch_size=32):
        """Train classical model on aggregated compressed data"""
        
        # Define classical model (similar to hybrid_model but without quantum part)
        layers = []
        input_dim = self.n_qubits  # Compressed dimension
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.25)
            ])
            input_dim = h_dim
        
        layers.append(nn.Linear(input_dim, self.n_classes))
        self.classical_model = nn.Sequential(*layers).to(self.device)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classical_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            self.classical_model.train()
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.classical_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def evaluate(self, clients=None, test_data=None):
        """Evaluate the federated model"""
        if self.classical_model is None:
            raise ValueError("Classical model not trained")
        
        if test_data is not None:
            # If test data provided, compress it using all clients
            X_test, y_test = test_data
            all_compressed = []
            
            for client in self.clients:
                compressed = client.compress_data(X_test)
                all_compressed.append(compressed)
            
            # Average the compressed features from all clients
            X_compressed = np.mean(all_compressed, axis=0)
            
        else:
            # Use clients' test data
            all_compressed = []
            all_labels = []
            
            for client in (clients or self.clients):
                compressed = client.compress_data(client.X_test)
                all_compressed.append(compressed)
                all_labels.append(client.y_test)
            
            X_compressed = np.vstack(all_compressed)
            y_test = np.hstack(all_labels)
        
        # Evaluate
        self.classical_model.eval()
        X_tensor = torch.tensor(X_compressed, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.classical_model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        accuracy = np.mean(predictions == y_test)
        return accuracy, predictions, y_test


class FederatedOrchestrator:
    """Main orchestrator for federated learning simulation"""
    
    def __init__(self, dataset_name, n_clients=5, test_size=0.2, 
                 schmidt_threshold=0.1, device='cpu', random_state=42, max_samples=10000):
        """
        Initialize federated orchestrator.
        
        Args:
            dataset_name: Name of dataset to load
            n_clients: Number of clients
            test_size: Fraction of data for testing
            schmidt_threshold: Threshold for Schmidt decomposition
            device: 'cpu' or 'cuda'
        """
        self.dataset_name = dataset_name
        self.n_clients = n_clients
        self.schmidt_threshold = schmidt_threshold
        self.device = device
        self.random_state = random_state
        # Load dataset
        X, y, n_qubits, n_classes = load_dataset(
            dataset_name, max_samples=max_samples
        )
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state= self.random_state
        )
      
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        
        # Create clients with data splits
        self.clients = self._create_clients(X_train, y_train, X_test, y_test)
        
        # Create server
        self.server = FLServer(n_qubits, n_classes, device)
        self.server.register_clients(self.clients)
    
    def _create_clients(self, X_train, y_train, X_test, y_test):
        """Split data among clients"""
        clients = []
        
        # Split indices for each client
        n_samples = len(X_train)
        indices = np.arange(n_samples)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        
        # Calculate samples per client (stratified by label if possible)
        samples_per_client = n_samples // self.n_clients
        
        for i in range(self.n_clients):
            # Get indices for this client
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < self.n_clients - 1 else n_samples
            
            client_indices = indices[start_idx:end_idx]
            
            # Create client with its data
            client = QuantumFLClient(
                client_id=i,
                X_train=X_train[client_indices],
                y_train=y_train[client_indices],
                X_test=X_test,
                y_test=y_test,
                schmidt_threshold=self.schmidt_threshold,
                device=self.device
            )
            clients.append(client)
        
        return clients
    
    def run_federated_learning(self, method='max'):
        """Run the complete federated learning process"""
        
        print("=" * 60)
        print(f"Starting Federated Learning on {self.dataset_name}")
        print(f"Number of clients: {self.n_clients}")
        print(f"Number of qubits: {self.n_qubits}")
        print(f"Number of classes: {self.n_classes}")
        print("=" * 60)
        
        # Phase 1: Local Schmidt Analysis
        print("\nPhase 1: Local Schmidt Analysis")
        local_ks = self.server.collect_local_k()
        
        # Phase 2: Global k Determination
        print("\nPhase 2: Global k Determination")
        global_k = self.server.determine_global_k(local_ks, method=method)
        
        # Phase 3: Local Quantum Compression
        print("\nPhase 3: Local Quantum Compression")
        self.server.broadcast_global_k()
        
        # Collect compressed data from clients
        print("\nCollecting compressed data from clients...")
        X_compressed, y_compressed = self.server.collect_compressed_data()
        
        print(f"Original dimension: {2**self.n_qubits}")
        print(f"Compressed dimension: {self.n_qubits}")
        print(f"Compression ratio: {2**self.n_qubits}/{self.n_qubits} = "
              f"{(2**self.n_qubits)/self.n_qubits:.1f}x")
        
        # Phase 4: Centralized Classical Training
        print("\nPhase 4: Centralized Classical Training")
        losses = self.server.train_classical_model(
            X_compressed, y_compressed,
            epochs=500, lr=0.01, batch_size=64
        )
        
        # Evaluation
        print("\nEvaluating federated model...")
        accuracy, predictions, y_true = self.server.evaluate()
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return {
            'local_ks': local_ks,
            'global_k': global_k,
            'losses': losses,
            'accuracy': accuracy,
            'predictions': predictions,
            'y_true': y_true
        }
    






# def main():
# if __name__ == "__main__":
"""Main function to run federated learning simulation"""

# Configuration
dataset_name = "cifar10"  # Try: "iris", "cifar10", "wine", "breast_cancer", "digits"
n_clients = 5
schmidt_threshold = 0.3  # Threshold for Schmidt decomposition 
max_samples = 10000  # Maximum samples to use from dataset

# method to choose k for the number of circuit terms
method = 'max'  # 'max', 'avg', or 'percentile'

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)
# Create orchestrator
orchestrator = FederatedOrchestrator(
    dataset_name=dataset_name,
    n_clients=n_clients,
    schmidt_threshold=schmidt_threshold,
    device=device,
    random_state=random_state,
    max_samples=max_samples
)

# Run federated learning
results = orchestrator.run_federated_learning(method=method)
from plotting import plot_results
# Plot results
plot_results(orchestrator, results)

# Print summary
print("\n" + "=" * 60)
print("Federated Learning Summary")
print("=" * 60)
print(f"Dataset: {dataset_name}")
print(f"Number of clients: {n_clients}")
print(f"Local k values: {results['local_ks']}")
print(f"Global k (method={method}): {results['global_k']}")
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Compression: {2**orchestrator.n_qubits} → {orchestrator.n_qubits} "
        f"({(2**orchestrator.n_qubits)/orchestrator.n_qubits:.1f}x compression)")
overall_acc = results['accuracy']
original_dim = 2**orchestrator.n_qubits
compressed_dim = orchestrator.n_qubits
compression_ratio = original_dim / compressed_dim
metrics_data = [
        ['Dataset', orchestrator.dataset_name],
        ['Number of Clients', f'{orchestrator.n_clients}'],
        ['Number of Qubits', f'{orchestrator.n_qubits}'],
        ['Original Dimension', f'{original_dim}'],
        ['Compressed Dimension', f'{compressed_dim}'],
        ['Compression Ratio', f'{compression_ratio:.1f}x'],
        ['Global Schmidt Rank ($k_g$)', f'{results["global_k"]}'],
        ['Overall Accuracy', f'{overall_acc:.4f}'],
        ['Minimum Local $k$', f'{min(results["local_ks"])}'],
        ['Maximum Local $k$', f'{max(results["local_ks"])}'],
        ['Average Local $k$', f'{np.mean(results["local_ks"]):.2f}'],
        ['Standard Deviation', f'{np.std(results["local_ks"]):.2f}'],
]
print("\nDetailed Metrics:")
for metric, value in metrics_data:
    print(f"{metric}: {value}")
# ====================================================================