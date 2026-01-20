"""
Script to run all experiments from the paper
"""

import subprocess
import sys

def run_experiment(dataset_name, schmidt_threshold, method='max'):
    """Run federated learning experiment for a specific dataset"""
    print(f"\n{'='*60}")
    print(f"Running experiment for {dataset_name}")
    print(f"{'='*60}")
    
    # Import here to avoid circular imports
    from federated_simulation import FederatedOrchestrator
    
    orchestrator = FederatedOrchestrator(
        dataset_name=dataset_name,
        n_clients=5,
        schmidt_threshold=schmidt_threshold,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results = orchestrator.run_federated_learning(method=method)
    
    return results

if __name__ == "__main__":
    import torch
    
    # Run all experiments from the paper
    experiments = [
        ("iris", 0.3),
        ("digits", 0.3),
        ("cifar10", 0.022),
    ]
    
    for dataset, threshold in experiments:
        run_experiment(dataset, threshold)
    
    print("\n✅ All experiments completed successfully!")