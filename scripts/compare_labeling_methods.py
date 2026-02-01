"""
Script to run Cost-Based vs Oracle Labeling Comparison.
Generates plots comparing noise rate and DDNN accuracy between the two methods.
"""

import torch
import torch.optim as optim
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import compare_labeling_methods
from src.models import LocalFeatureExtractor, LocalClassifier, CloudCNN
from src.data import load_data
from src.training import train_DDNN

# Parse arguments
parser = argparse.ArgumentParser(description='Compare Cost-Based vs Oracle Labeling methods')
parser.add_argument('--mode', type=str, default='load', choices=['train', 'load'],
                   help='Mode: train DDNN from scratch or load pretrained')
parser.add_argument('--dataset', type=str, default='cifar10',
                   choices=['cifar10', 'cifar100', 'cinic10', 'svhn', 'gtsrb32', 'fashion_mnist'],
                   help='Dataset to use')
parser.add_argument('--L0', type=float, default=0.54,
                   help='Target local processing ratio')
parser.add_argument('--epochs_ddnn', type=int, default=50,
                   help='Epochs to train DDNN (only for train mode)')
parser.add_argument('--epochs_offload', type=int, default=50,
                   help='Epochs to train each offload mechanism')
parser.add_argument('--input_mode', type=str, default='logits_plus',
                   choices=['logits', 'logits_plus', 'hybrid', 'feat'],
                   help='Input mode for offload mechanism')
parser.add_argument('--batch_size', type=int, default=256,
                   help='Batch size')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = args.dataset

# Map dataset to num_classes
DATASET_INFO = {
    'cifar10': 10, 'cifar100': 100, 'cinic10': 10,
    'svhn': 10, 'gtsrb32': 43, 'fashion_mnist': 10
}
num_classes = DATASET_INFO[dataset_name]

print(f"\n{'='*80}")
print(f"COST-BASED vs ORACLE LABELING COMPARISON")
print(f"{'='*80}")
print(f"Dataset: {dataset_name.upper()}")
print(f"Mode: {args.mode}")
print(f"L0: {args.L0}")
print(f"Input Mode: {args.input_mode}")
print(f"{'='*80}\n")

# 1. Load data
print("Loading data...")
train_loader, val_loader, test_loader = load_data(batch_size=args.batch_size, dataset=dataset_name)

# 2. Initialize models
print("Initializing models...")
local_feat_extr = LocalFeatureExtractor().to(device)
local_clf = LocalClassifier(num_classes=num_classes).to(device)
cloud_cnn = CloudCNN(num_classes=num_classes).to(device)

# 3. Train or Load DDNN models
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

if args.mode == 'train':
    print(f"\n{'='*40}")
    print("TRAINING DDNN FROM SCRATCH")
    print(f"{'='*40}")
    
    # Combined optimizer for all DDNN components
    all_params = list(local_feat_extr.parameters()) + \
                 list(local_clf.parameters()) + \
                 list(cloud_cnn.parameters())
    cnn_optimizer = optim.Adam(all_params, lr=0.001)
    local_weight = 0.5  # Balance between local and cloud loss
    
    train_DDNN(
        train_loader,
        local_feat_extr, local_clf, cloud_cnn,
        cnn_optimizer,
        local_weight,
        args.epochs_ddnn
    )
    
    torch.save(local_feat_extr.state_dict(), f'{model_dir}/local_feature_extractor_{dataset_name}.pth')
    torch.save(local_clf.state_dict(), f'{model_dir}/local_classifier_{dataset_name}.pth')
    torch.save(cloud_cnn.state_dict(), f'{model_dir}/cloud_cnn_{dataset_name}.pth')
    print(f"\n✓ Models saved to {model_dir}/")
    
else:
    print("Loading pretrained DDNN models...")
    local_feat_extr.load_state_dict(torch.load(f'{model_dir}/local_feature_extractor_{dataset_name}.pth'))
    local_clf.load_state_dict(torch.load(f'{model_dir}/local_classifier_{dataset_name}.pth'))
    cloud_cnn.load_state_dict(torch.load(f'{model_dir}/cloud_cnn_{dataset_name}.pth'))
    print("✓ Models loaded")

# 4. Run comparison
print(f"\n{'='*80}")
print("RUNNING LABELING COMPARISON...")
print(f"{'='*80}")

results = compare_labeling_methods(
    local_feat_extr,
    local_clf,
    cloud_cnn,
    train_loader,
    val_loader,
    test_loader,
    L0=args.L0,
    input_mode=args.input_mode,
    offload_epochs=args.epochs_offload,
    batch_size=args.batch_size,
    device=device,
    dataset_name=dataset_name,
    plot=True,
    num_classes=num_classes
)

# 5. Print final summary
print(f"\n{'='*80}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*80}")
print(f"\n  📊 NOISE ANALYSIS:")
print(f"  ─────────────────────────────────────────────────")
print(f"  Total training samples:  {results['total_samples']:,}")
print(f"  Noise Rate:              {results['noise_rate']:.2f}%")
print(f"  Agreement Rate:          {results['agreement_rate']:.2f}%")

print(f"\n  📊 LABEL DISTRIBUTION:")
print(f"  ─────────────────────────────────────────────────")
print(f"  Cost-Based Rule: Local {results['bk_local_pct']:.1f}%, Cloud {results['bk_cloud_pct']:.1f}%")
print(f"  Oracle Rule:     Local {results['oracle_local_pct']:.1f}%, Cloud {results['oracle_cloud_pct']:.1f}%")

print(f"\n  📊 DDNN OVERALL ACCURACY (Test Set):")
print(f"  ─────────────────────────────────────────────────")
print(f"  Cost-Based Rule training: {results['bk_ddnn_accuracy']:.2f}%")
print(f"  Oracle Labeling training: {results['oracle_ddnn_accuracy']:.2f}%")
print(f"  Difference:               {results['accuracy_difference']:+.2f}%")

print(f"\n  📊 TEST LOCAL PERCENTAGES:")
print(f"  ─────────────────────────────────────────────────")
print(f"  Cost-Based: {results['bk_local_test_pct']:.1f}%")
print(f"  Oracle:     {results['oracle_local_test_pct']:.1f}%")

print(f"\n{'='*80}")
print(f"✅ ANALYSIS COMPLETE!")
print(f"Plot saved: plots/labeling_comparison_{dataset_name}_L0{int(args.L0*100)}.png")
print(f"{'='*80}\n")
