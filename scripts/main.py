"""
Main script for DDNN with Deep Offload Mechanism
Trains and evaluates the distributed deep neural network with learned offload decisions
"""

import os
import sys
import argparse
import torch
import torch.optim as optim

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import LocalFeatureExtractor, LocalClassifier, CloudCNN, OffloadMechanism
from src.data import load_data  # Don't import NUM_CLASSES here - it will be set dynamically
from src.utils import initialize_models, initialize_optimizers
from src.training import train_DDNN
from src.evaluation import (
    testing_offload_mechanism,
    test_inference_timing,
    analyze_border_noisy_misclassification,
    test_offload_overfitting
)


def main(
    epochs_DDNN=50,
    epochs_optimization=30,
    batch_size=256,
    L0=0.54,
    local_weight=0.7,
    mode='load',
    dataset_name='cifar10',
    testing_mode='offload_mechanism'
):
    """
    Initialize models, train the 2 Networks (Local and Remote), create local features and bks from the DDNN
    in order to train the Optimization Rule Network and test the DDNN with the optimized rule

    Args:
        epochs_DDNN (int): Number of epochs for DDNN training. Defaults to 50.
        epochs_optimization (int): Number of epochs for offload mechanism training. Defaults to 20.
        batch_size (int): Batch size for training. Defaults to 256.
        L0 (float): Target local percentage for single L0 test. Defaults to 0.54.
        local_weight (float): Weight for local loss in DDNN training. Defaults to 0.7.
        mode (str): 'train' or 'test'. Defaults to 'train'.
        dataset_name (str): Dataset to use. Defaults to 'cifar10'.
        testing_mode (str): Testing mode - 'offload_mechanism', 'timing', 'border_noisy', 'overfitting'. Defaults to 'offload_mechanism'.
    """
    
    DATASET_INFO = {
        'cifar10': 10,   # baseline easy
        'cifar100': 100,  # harder 100-class variant
        'cinic10': 10,   # CIFAR/ImageNet mix (32×32)
        'svhn': 10,   # street-view digits (32×32)
        'gtsrb32': 43,   # traffic signs (32×32)
        'fashion_mnist': 10  # Fashion-MNIST (28×28 → 32×32, grayscale → RGB)
    }
    
    # Set NUM_CLASSES based on dataset - must be done BEFORE importing models
    if dataset_name in DATASET_INFO:
        num_classes = DATASET_INFO[dataset_name]
        # Update the global NUM_CLASSES in src.data module

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # === Create 'models' directory if it does not exist ===
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize data loaders
    print(f"\nLoading {dataset_name} dataset...")
    train_loader, val_loader, test_loader = load_data(batch_size, dataset=dataset_name)
    
    # Initialize models with correct NUM_CLASSES
    print("\nInitializing models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_feature_extractor = LocalFeatureExtractor().to(device)
    local_classifier = LocalClassifier(num_classes=num_classes).to(device)
    cloud_cnn = CloudCNN(num_classes=num_classes).to(device)
    # OffloadMechanism needs explicit NUM_CLASSES argument
    offload_mechanism = OffloadMechanism(input_mode='logits_plus', num_classes=num_classes).to(device)
    
    cnn_optimizer, offload_optimizer = initialize_optimizers(
        local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism
    )
    
    # Initialize scheduler after optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode='min', factor=0.5, patience=10)
    print("✓ Models and optimizers initialized successfully!")
    if mode == 'train':
        # Train the DDNN network
        print(f"\nTraining DDNN for {epochs_DDNN} epochs...")
        train_DDNN(
            train_loader,
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            cnn_optimizer,
            local_weight,
            epochs_DDNN
        )
        
        # === Save models in the 'models' directory ===
        torch.save(local_feature_extractor.state_dict(), os.path.join(models_dir, f"local_feature_extractor_{dataset_name}.pth"))
        torch.save(local_classifier.state_dict(), os.path.join(models_dir, f"local_classifier_{dataset_name}.pth"))
        torch.save(cloud_cnn.state_dict(), os.path.join(models_dir, f"cloud_cnn_{dataset_name}.pth"))
        torch.save(offload_mechanism.state_dict(), os.path.join(models_dir, f"offload_mechanism_{dataset_name}.pth"))
        print(f"\n✓ Models saved successfully in 'models/' directory (dataset: {dataset_name})!")
    
    else:  # mode == 'test'
        # Load the models from the 'models' directory
        print("\nLoading pretrained models...")
        local_feature_extractor.load_state_dict(
            torch.load(os.path.join(models_dir, "local_feature_extractor.pth"))
        )
        local_classifier.load_state_dict(
            torch.load(os.path.join(models_dir, "local_classifier.pth"))
        )
        cloud_cnn.load_state_dict(
            torch.load(os.path.join(models_dir, "cloud_cnn.pth"))
        )
        # Note: offload_mechanism will be trained fresh for each L0 value in testing_offload_mechanism
        print("✓ Models loaded successfully!")
    
    # ============================================================================
    # EXPERIMENTS: Run different tests based on testing_mode
    # ============================================================================
    print("\n" + "="*80)
    print(f"EXPERIMENT: {testing_mode.upper().replace('_', ' ')}")
    print("="*80)
    
    if testing_mode == 'offload_mechanism':
        # Test offload mechanism across multiple L0 values
        L0_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.585, 0.6, 0.7, 0.9, 1]
        results = testing_offload_mechanism(
            L0_values=L0_values,
            local_feature_extractor=local_feature_extractor,
            local_classifier=local_classifier,
            cloud_cnn=cloud_cnn,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            methods_to_test=['logits',  'entropy', 'random','feat', 'logits_plus'],
            device='cuda',
            offload_epochs=epochs_optimization,
            batch_size=batch_size,
            dataset_name=dataset_name,
            evaluation_target='ddnn_overall',
            num_classes=num_classes
        )
    
    elif testing_mode == 'timing':
        # Inference timing benchmark
        results = test_inference_timing(
            L0=L0,
            local_feature_extractor=local_feature_extractor,
            local_classifier=local_classifier,
            cloud_cnn=cloud_cnn,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            methods_to_test=['logits', 'entropy', 'random'],
            device='cuda',
            offload_epochs=epochs_optimization,
            batch_size=batch_size,
            dataset_name=dataset_name,
            input_mode='logits',
            num_runs=5,
            num_classes=num_classes
        )
    
    elif testing_mode == 'border_noisy':
        # Border/noisy misclassification analysis
        results = analyze_border_noisy_misclassification(
            local_feature_extractor=local_feature_extractor,
            local_classifier=local_classifier,
            cloud_cnn=cloud_cnn,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            L0=L0,
            tau=0.01,
            input_mode='logits_plus',
            offload_epochs=epochs_optimization,
            batch_size=batch_size,
            dataset_name=dataset_name,
            device='cuda',
            plot=True,
            num_classes=num_classes
        )
    
    elif testing_mode == 'overfitting':
        # Overfitting test
        results = test_offload_overfitting(
            local_feature_extractor=local_feature_extractor,
            local_classifier=local_classifier,
            cloud_cnn=cloud_cnn,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            L0=L0,
            methods_to_test=['feat', 'logits'],
            device='cuda',
            offload_epochs=epochs_optimization,
            batch_size=batch_size,
            dataset_name=dataset_name,
            num_classes=num_classes
        )
    
    else:
        raise ValueError(f"Unknown testing_mode: {testing_mode}")
    
    # Print results summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    if testing_mode == 'offload_mechanism':
        print(f"\nResults structure:")
        for method in results:
            print(f"\n{method}:")
            for key, val in results[method].items():
                if isinstance(val, list) and len(val) > 3:
                    print(f"  {key}: {val[:3]}... (showing first 3 values)")
                else:
                    print(f"  {key}: {val}")
    elif testing_mode == 'timing':
        print(f"\nTiming Results:")
        for method in results:
            print(f"\n{method}:")
            print(f"  Total Time: {results[method]['total_time']:.3f}s")
            print(f"  Per-Sample: {results[method]['per_sample_time']:.3f}ms")
            print(f"  Accuracy: {results[method]['accuracy']:.2f}%")
    elif testing_mode == 'border_noisy':
        print(f"\nBorder/Noisy Analysis Results:")
        print(f"  Noisy Misclassification Rate: {results['noisy_misclass_rate']:.2f}%")
        print(f"  Border Misclassification Rate: {results['border_misclass_rate']:.2f}%")
        print(f"  Normal Misclassification Rate: {results['normal_misclass_rate']:.2f}%")
    elif testing_mode == 'overfitting':
        print(f"\nOverfitting Results:")
        for method in results:
            print(f"\n{method}:")
            print(f"  Epochs: {len(results[method]['epochs'])}")
            print(f"  Final Train Acc: {results[method]['train_accs'][-1]:.2f}%")
            print(f"  Final Val Acc: {results[method]['val_accs'][-1]:.2f}%")
    
    print("\n✓ Check 'plots/' directory for generated figures!")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate DDNN with Deep Offload Mechanism')
    
    # Main arguments
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'load'],
                        help='Mode: train (train DDNN then test) or load (load pretrained DDNN if it existsand test)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'cinic10', 'svhn', 'gtsrb32'],
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--testing_mode', type=str, default='offload_mechanism',
                        choices=['offload_mechanism', 'timing', 'border_noisy', 'overfitting'],
                        help='Testing mode to run (default: offload_mechanism)')
    
    # Training hyperparameters
    parser.add_argument('--epochs_ddnn', type=int, default=50,
                        help='Number of epochs for DDNN training (default: 50)')
    parser.add_argument('--epochs_offload', type=int, default=30,
                        help='Number of epochs for offload mechanism training (default: 30)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--local_weight', type=float, default=0.7,
                        help='Weight for local loss in DDNN training (default: 0.7)')
    parser.add_argument('--L0', type=float, default=0.54,
                        help='Target local percentage for single L0 tests (timing/border_noisy/overfitting) (default: 0.54)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("DDNN with Deep Offload Mechanism - Experiments")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Testing Mode: {args.testing_mode}")
    print(f"  DDNN Training Epochs: {args.epochs_ddnn}")
    print(f"  Offload Training Epochs: {args.epochs_offload}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Local Weight: {args.local_weight}")
    if args.testing_mode != 'offload_mechanism':
        print(f"  L0 (Target Local %): {args.L0}")
    
    # Run main experiment
    results = main(
        epochs_DDNN=args.epochs_ddnn,
        epochs_optimization=args.epochs_offload,
        batch_size=args.batch_size,
        L0=args.L0,
        local_weight=args.local_weight,
        mode=args.mode,
        dataset_name=args.dataset,
        testing_mode=args.testing_mode
    )
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
