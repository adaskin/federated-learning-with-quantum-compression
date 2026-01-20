
import matplotlib.pyplot as plt
import numpy as np


def plot_results(orchestrator, results):
    """Create publication-quality plots for federated learning results"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        # 'font.family': 'serif',
        # 'font.serif': ['Times New Roman', 'Computer Modern Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.8,
    })
    
    # Create figure with GridSpec for precise layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.5, wspace=1.4)
    
    # Colors
    cmap = plt.cm.viridis
    client_colors = [cmap(i / (orchestrator.n_clients - 1)) for i in range(orchestrator.n_clients)]
    main_color = '#2E86AB'
    highlight_color = '#A23B72'
    
    # ========== Plot 1: Local k Distribution (Top Left) ==========
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Create grouped bar chart with individual client values
    x = np.arange(orchestrator.n_clients)
    width = 0.6
    
    bars = ax1.bar(x, results['local_ks'], width, 
                color=client_colors, edgecolor='black', linewidth=0.8,
                alpha=0.8, label='Local $k$')
    
    # Add global k line
    ax1.axhline(y=results['global_k'], color=highlight_color, 
                linestyle='--', linewidth=2, alpha=0.8, 
                label=f'Global $k={results["global_k"]}$')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Client ID', fontweight='bold')
    ax1.set_ylabel('Schmidt Rank ($k$)', fontweight='bold')
    ax1.set_title('A. Local Schmidt Rank Distribution', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'C{i}' for i in range(orchestrator.n_clients)])
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistical annotations
    ax1.text(0.02, 0.58, 
            f'Min: {min(results["local_ks"])}, Max: {max(results["local_ks"])}, '
            f'Avg: {np.mean(results["local_ks"]):.1f}',
            transform=ax1.transAxes, fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # ========== Plot 2: Training Loss (Top Right) ==========
    ax2 = fig.add_subplot(gs[0, 2:])
    
    # Smooth loss curve
    loss = np.array(results['losses'])
    window_size = max(1, len(loss) // 20)
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        smoothed_loss = np.convolve(loss, kernel, mode='valid')
        ax2.plot(smoothed_loss, color=main_color, alpha=0.3, linewidth=1, 
                label=f'Smoothed (window={window_size})')
    
    ax2.plot(loss, color=main_color, linewidth=2, label='Training Loss')
    
    # Mark convergence point (90% of final loss)
    final_loss = loss[-1]

    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Cross-Entropy Loss', fontweight='bold')
    ax2.set_title('B. Classical Model Training Convergence', fontweight='bold', pad=15)
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add final loss annotation
    ax2.text(0.95, 0.05, f'Final Loss: {final_loss:.4f}',
            transform=ax2.transAxes, ha='right', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    

    
    # ========== Plot 4: Accuracy Distribution (Middle Right) ==========
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Compute per-class accuracy
    unique_classes = np.unique(results['y_true'])
    class_accuracies = []
    
    for cls in unique_classes:
        idx = results['y_true'] == cls
        if np.sum(idx) > 0:
            accuracy = np.mean(results['predictions'][idx] == cls)
            class_accuracies.append(accuracy)
    
    # Create bar plot
    bars2 = ax4.bar(range(len(class_accuracies)), class_accuracies,
                color=plt.cm.Set2(np.linspace(0, 1, len(class_accuracies))),
                edgecolor='black', linewidth=0.8)
    
    # Add overall accuracy line
    overall_acc = results['accuracy']
    ax4.axhline(y=overall_acc, color=highlight_color, linestyle='--',
            linewidth=2, alpha=0.8, label=f'Overall: {overall_acc:.3f}')
    
    ax4.set_xlabel('Class Index', fontweight='bold')
    ax4.set_ylabel('Accuracy', fontweight='bold')
    ax4.set_title('C. Per-Class Accuracy', fontweight='bold', pad=15)
    ax4.set_xticks(range(len(class_accuracies)))
    ax4.set_xticklabels([f'{i}' for i in range(len(class_accuracies))])
    ax4.set_ylim(0, 1.05)
    ax4.legend(loc='lower right', framealpha=0.95)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars2, class_accuracies)):
        ax4.text(bar.get_x() + bar.get_width()/2., acc + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # ========== Plot 5: Compression Summary (Bottom Left) ==========
    ax5 = fig.add_subplot(gs[1, 2:])
    
    # Create donut chart for compression ratio
    original_dim = 2**orchestrator.n_qubits
    compressed_dim = orchestrator.n_qubits
    compression_ratio = original_dim / compressed_dim
    
    wedges, texts, autotexts = ax5.pie(
        [compressed_dim, original_dim - compressed_dim],
        labels=['Compressed', 'Discarded'],
        colors=[main_color, '#E8E8E8'],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Style the autopct text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    # Add center text
    center_text = f'{original_dim}→{compressed_dim}\n{compression_ratio:.1f}x'
    ax5.text(0, 0, center_text, ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    ax5.set_title('D. Dimensionality Reduction', fontweight='bold', pad=15)
    

    # Add footer
    plt.figtext(0.5, 0.01, 
                f'Experiment: {orchestrator.dataset_name} | Method: Schmidt Decomposition | '
                f'Compression: {original_dim} → {compressed_dim} ({compression_ratio:.1f}x)',
                ha='center', fontsize=9, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f'fig_federated_results_{orchestrator.dataset_name}.pdf', 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'fig_federated_results_{orchestrator.dataset_name}.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'fig_federated_results_{orchestrator.dataset_name}.svg', 
                bbox_inches='tight', pad_inches=0.1)
    
    plt.show()
    


def create_summary_figure(orchestrator, results):
    """Create a compact summary figure suitable for paper columns"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Consistent color scheme
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))
    
    # 1. Schmidt rank comparison
    x = np.arange(orchestrator.n_clients)
    ax1.bar(x - 0.15, results['local_ks'], 0.3, label='Local $k$', 
            color=colors[0], alpha=0.8)
    ax1.bar(x + 0.15, [results['global_k']]*orchestrator.n_clients, 0.3, 
            label=f'Global $k={results["global_k"]}$', color=colors[1], alpha=0.8)
    ax1.set_xlabel('Client ID', fontweight='bold')
    ax1.set_ylabel('Schmidt Rank', fontweight='bold')
    ax1.set_title('A. Schmidt Rank Distribution', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'C{i}' for i in range(orchestrator.n_clients)])
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # 2. Training convergence
    ax2.plot(results['losses'], color=colors[2], linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_title('B. Training Convergence', fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.text(0.7, 0.9, f'Final: {results["losses"][-1]:.4f}',
            transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # 3. Compression visualization
    original_dim = 2**orchestrator.n_qubits
    compressed_dim = orchestrator.n_qubits

    sizes = [original_dim, compressed_dim]
    labels = ['Original', 'Compressed']
    explode = (0.1, 0)
    
    wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels,
                                    colors=[colors[3], colors[1]], autopct='%1.1f%%',
                                    shadow=True, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax3.set_title(f'C. Compression: {original_dim}→{compressed_dim}\n'
                f'({original_dim/compressed_dim:.1f}x)', fontweight='bold')
    
    # 4. Performance metrics
    metrics = [
        f'Accuracy: {results["accuracy"]:.4f}',
        f'Global $k$: {results["global_k"]}',
        f'Avg Local $k$: {np.mean(results["local_ks"]):.2f}',
        f'Compression: {original_dim/compressed_dim:.1f}x',
        f'Clients: {orchestrator.n_clients}',
        f'Qubits: {orchestrator.n_qubits}',
    ]
    
    ax4.axis('off')
    y_pos = 0.9
    for metric in metrics:
        ax4.text(0.1, y_pos, metric, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        y_pos -= 0.15
    
    ax4.set_title('D. Performance Summary', fontweight='bold', y=0.95)
    
    # # Overall title
    # plt.suptitle('Quantum Federated Learning with Schmidt Decomposition',
    #             fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'fig_summary_{orchestrator.dataset_name}.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'fig_summary_{orchestrator.dataset_name}.png', 
                dpi=300, bbox_inches='tight')
    
    plt.show()
