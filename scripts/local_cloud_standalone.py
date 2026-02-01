"""
Local and Cloud Standalone Testing Script

Tests local-only and cloud-only modes for baseline comparison.
"""

import sys
import os
import torch
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data
from src.models import LocalFeatureExtractor, LocalClassifier, CloudCNN, CloudLogitPredictor
from src.evaluation import testing_offload_mechanism
from src.training import train_DDNN
from src.utils import initialize_optimizers

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_NAME = 'cifar10'
BATCH_SIZE = 256

L0_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.585, 0.6, 0.7, 0.9, 1] # Target local percentages (list for testing_offload_mechanism)
OFFLOAD_EPOCHS = 30
NUM_CLASSES = 10
LOGITS_ONLY_PREDICTOR = False
FEAT_LATENT_DIM = 32

# Modes to compare
MODES_TO_TEST = [
    'logits',                       # Basic: 10-dim
    'logits_plus',                  # Enhanced: 12-dim
    'logits_with_bk_pred',          # With predicted bk: 22-dim
    'logits_with_real_bk',          # With real bk (upper bound): 22-dim
    'logits_predicted_regression',  # Regression mode: 20-dim
    'hybrid',                       # Features + logits: 44-dim
]


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Local/Cloud Standalone Testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'load'],
                        help='train: Train DDNN first, load: Use pretrained models')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        help='Dataset name (default: fashion_mnist)')
    parser.add_argument('--epochs_ddnn', type=int, default=50,
                        help='DDNN training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    
    args = parser.parse_args()
    
    # Update config from args
    DATASET_NAME = args.dataset
    BATCH_SIZE = args.batch_size
    EPOCHS_DDNN = args.epochs_ddnn
    
    print("="*80)
    print("LOCAL/CLOUD STANDALONE TESTING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  L0 values: {L0_VALUES}")
    print(f"  Offload training epochs: {OFFLOAD_EPOCHS}")
    
    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    train_loader, val_loader, test_loader = load_data(BATCH_SIZE, dataset=DATASET_NAME)
    
    # Initialize models
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    local_feature_extractor = LocalFeatureExtractor().to(device)
    local_classifier = LocalClassifier(num_classes=NUM_CLASSES).to(device)
    cloud_cnn = CloudCNN(num_classes=NUM_CLASSES).to(device)
    
    if args.mode == 'train':
        # Train DDNN models
        print(f"\n{'='*80}")
        print(f"TRAINING DDNN FOR {EPOCHS_DDNN} EPOCHS")
        print(f"{'='*80}")
        
        # Initialize optimizer for DDNN only (no offload mechanism)
        cnn_optimizer = torch.optim.Adam(
            list(local_feature_extractor.parameters()) + 
            list(local_classifier.parameters()) + 
            list(cloud_cnn.parameters()), 
            lr=0.001
        )
        
        train_DDNN(
            train_loader,
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            cnn_optimizer,
            local_weight=0.7,
            epochs_DDNN=EPOCHS_DDNN
        )
        
        # Save models
        torch.save(local_feature_extractor.state_dict(), 
                   os.path.join(models_dir, f"local_feature_extractor_{DATASET_NAME}.pth"))
        torch.save(local_classifier.state_dict(), 
                   os.path.join(models_dir, f"local_classifier_{DATASET_NAME}.pth"))
        torch.save(cloud_cnn.state_dict(), 
                   os.path.join(models_dir, f"cloud_cnn_{DATASET_NAME}.pth"))
        print(f"\n✓ DDNN models saved to 'models/' directory (dataset: {DATASET_NAME})")
    
    else:  # mode == 'load'
        # Load pretrained models
        print(f"\n{'='*80}")
        print("LOADING PRETRAINED MODELS")
        print(f"{'='*80}")
        
        local_feature_extractor.load_state_dict(
            torch.load(os.path.join(models_dir, f"local_feature_extractor_{DATASET_NAME}.pth"),
                       map_location=device)
        )
        local_classifier.load_state_dict(
            torch.load(os.path.join(models_dir, f"local_classifier_{DATASET_NAME}.pth"),
                       map_location=device)
        )
        cloud_cnn.load_state_dict(
            torch.load(os.path.join(models_dir, f"cloud_cnn_{DATASET_NAME}.pth"),
                       map_location=device)
        )
        print("✓ DDNN models loaded successfully")
    
    # Load CloudLogitPredictor (needed for some modes, optional for standalone)
    print("\n[Loading CloudLogitPredictor...]")
    cloud_predictor = CloudLogitPredictor(num_classes=NUM_CLASSES, logits_only=LOGITS_ONLY_PREDICTOR).to(device)
    predictor_path = os.path.join(models_dir, "cloud_logit_predictor_fc_only.pth" if LOGITS_ONLY_PREDICTOR else "cloud_logit_predictor.pth")
    
    if os.path.exists(predictor_path):
        cloud_predictor.load_state_dict(torch.load(predictor_path, map_location=device))
        cloud_predictor.eval()
        print(f"✓ CloudLogitPredictor loaded successfully")
    else:
        print(f"⚠ CloudLogitPredictor not found (not needed for standalone modes)")
        cloud_predictor = None
    
    # ============================================================================
    # RUN STANDALONE TESTING
    # ============================================================================
    
    print(f"\n{'='*80}")
    print("TESTING LOCAL/CLOUD STANDALONE MODES")
    print(f"{'='*80}")
    
    results = testing_offload_mechanism(
        L0_values=L0_VALUES,
        local_feature_extractor=local_feature_extractor,
        local_classifier=local_classifier,
        cloud_cnn=cloud_cnn,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        methods_to_test=['local_standalone', 'cloud_standalone','oracle'],
        device='cuda',
        offload_epochs=OFFLOAD_EPOCHS,
        batch_size=BATCH_SIZE,
        dataset_name=DATASET_NAME,
        evaluation_target='ddnn_overall',
        num_classes=NUM_CLASSES,
        cloud_predictor=cloud_predictor,
        feat_latent_dim=FEAT_LATENT_DIM
    )

    print("\n✓ LOCAL/CLOUD STANDALONE TESTING COMPLETE!")
