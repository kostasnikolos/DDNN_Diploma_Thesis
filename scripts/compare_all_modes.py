"""
Multi-Mode Comparison Script

Tests all offload mechanism input modes and compares their DDNN overall accuracy
using the unified testing_offload_mechanism function.

Modes tested:
- logits (10-dim): Raw local logits
- logits_plus (12-dim): local_logits + margin + entropy
- logits_with_bk_pred (22-dim): logits_plus + predicted_bk (softmax diff)
- logits_with_real_bk (22-dim): logits_plus + real_bk (upper bound)
- logits_predicted_regression (20-dim): local_logits + predicted_cloud_logits
- hybrid (44-dim): compressed features + logits_plus

Generates comparison plot showing Oracle accuracy and gap for each mode.
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data
from src.models import LocalFeatureExtractor, LocalClassifier, CloudCNN, CloudLogitPredictor
from src.evaluation import testing_offload_mechanism

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_NAME = 'cifar10'
BATCH_SIZE = 256

L0_VALUES = [0, 0.1, 0.2, 0.3, 0.5, 0.585, 0.6, 0.7, 0.9, 1] # Target local percentages (list for testing_offload_mechanism)
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
    print("="*80)
    print("MULTI-MODE COMPARISON: OFFLOAD MECHANISM INPUT MODES")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  L0 values: {L0_VALUES}")
    print(f"  Offload training epochs: {OFFLOAD_EPOCHS}")
    print(f"  Modes to test: {len(MODES_TO_TEST)}")
    for mode in MODES_TO_TEST:
        print(f"    - {mode}")
    
    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    train_loader, val_loader, test_loader = load_data(BATCH_SIZE, dataset=DATASET_NAME)
    
    # Load models
    print(f"\n{'='*80}")
    print("LOADING PRETRAINED MODELS")
    print(f"{'='*80}")
    
    models_dir = "models"
    local_feature_extractor = LocalFeatureExtractor().to(device)
    local_classifier = LocalClassifier(num_classes=NUM_CLASSES).to(device)
    cloud_cnn = CloudCNN(num_classes=NUM_CLASSES).to(device)
    
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
    
    # Load CloudLogitPredictor (needed for logits_with_bk_pred and regression modes)
    print("\n[Loading CloudLogitPredictor...]")
    cloud_predictor = CloudLogitPredictor(num_classes=NUM_CLASSES, logits_only=LOGITS_ONLY_PREDICTOR).to(device)
    predictor_path = os.path.join(models_dir, "cloud_logit_predictor_fc_only.pth" if LOGITS_ONLY_PREDICTOR else "cloud_logit_predictor.pth")
    cloud_predictor.load_state_dict(torch.load(predictor_path, map_location=device))
    cloud_predictor.eval()
    print(f"✓ CloudLogitPredictor loaded successfully")
    
    # ============================================================================
    # RUN TESTING_OFFLOAD_MECHANISM (handles all printing and plotting)
    # ============================================================================
    
    # Add oracle to methods for comparison
    methods = MODES_TO_TEST + ['oracle']
    
    # results = testing_offload_mechanism(
    #     L0_values=L0_VALUES,
    #     local_feature_extractor=local_feature_extractor,
    #     local_classifier=local_classifier,
    #     cloud_cnn=cloud_cnn,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     val_loader=val_loader,
    #     methods_to_test=methods,
    #     device='cuda',
    #     offload_epochs=OFFLOAD_EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     dataset_name=DATASET_NAME,
    #     evaluation_target='ddnn_overall',
    #     num_classes=NUM_CLASSES,
    #     cloud_predictor=cloud_predictor,
    #     feat_latent_dim=FEAT_LATENT_DIM
    # )
    # results = testing_offload_mechanism(
    #     L0_values=L0_VALUES,
    #     local_feature_extractor=local_feature_extractor,
    #     local_classifier=local_classifier,
    #     cloud_cnn=cloud_cnn,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     val_loader=val_loader,
    #     methods_to_test=[ 'logits', 'entropy'],
    #     device='cuda',
    #     offload_epochs=OFFLOAD_EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     dataset_name=DATASET_NAME,
    #     evaluation_target='ddnn_overall',
    #     num_classes=NUM_CLASSES,
    #     cloud_predictor=cloud_predictor,
    #     feat_latent_dim=FEAT_LATENT_DIM
    # )
    results = testing_offload_mechanism(
        L0_values=L0_VALUES,
        local_feature_extractor=local_feature_extractor,
        local_classifier=local_classifier,
        cloud_cnn=cloud_cnn,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        methods_to_test=['logits', 'oracle', 'entropy'],
        device='cuda',
        offload_epochs=OFFLOAD_EPOCHS,
        batch_size=BATCH_SIZE,
        dataset_name=DATASET_NAME,
        evaluation_target='ddnn_overall',
        num_classes=NUM_CLASSES,
        cloud_predictor=cloud_predictor,
        feat_latent_dim=FEAT_LATENT_DIM
    )

    print("\n✓ MULTI-MODE COMPARISON COMPLETE!")
