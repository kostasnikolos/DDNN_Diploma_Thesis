"""
Evaluation and Testing functions for DDNN and Offload Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils import (
    my_oracle_decision_function,
    compute_bks_input_for_deep_offload,
    calculate_b_star,
    create_3d_data_deep,
    calculate_normalized_entropy,
    calibrate_threshold
)
from src.models import OffloadMechanism, OffloadDatasetCNN
import src.data.data_loader  # Import module to access dynamic NUM_CLASSES
from src.training import train_deep_offload_mechanism


# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_logits_with_bk_pred(local_out, local_feats, cloud_predictor, num_classes):
    """
    Compute combined input for logits_with_bk_pred mode.
    
    Combines two signals:
    1. logits_plus (12-dim): local logits + uncertainty measures (margin, entropy)
    2. predicted_bk (10-dim): predicted difference between cloud and local outputs
    
    The OffloadMechanism learns to use these signals for offloading decisions.
    
    Args:
        local_out: (B, num_classes) local logits
        local_feats: (B, 32, 16, 16) local feature maps
        cloud_predictor: CloudLogitPredictor model
        num_classes: number of classes (typically 10)
    
    Returns:
        combined_input: (B, 22) concatenation of logits_plus (12) + predicted_bk (10)
    """
    # ★ SIGNAL 1: logits_plus = local logits + uncertainty measures
    probs = F.softmax(local_out, dim=1)
    top2 = torch.topk(probs, 2, dim=1).values
    
    # Margin: confidence in top prediction (top1_prob - top2_prob)
    margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
    
    # Entropy: uncertainty of the local model (normalized)
    entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_classes)
    
    logits_plus = torch.cat([local_out, margin, entropy], dim=1)  # (B, 12)
    
    # ★ SIGNAL 2: predicted_bk = difference between predicted cloud and local outputs
    # The CloudLogitPredictor acts as a "cloud oracle" that hasn't seen the ground truth
    with torch.no_grad():
        if hasattr(cloud_predictor, 'logits_only') and cloud_predictor.logits_only:
            # FC-only mode: only use local logits
            predicted_cloud_logits = cloud_predictor(local_out)
        else:
            # CNN+Logits mode: use both features and logits (better accuracy)
            predicted_cloud_logits = cloud_predictor(local_feats, local_out)
        
        predicted_cloud_probs = F.softmax(predicted_cloud_logits, dim=1)
        local_probs = F.softmax(local_out, dim=1)
        predicted_bk = predicted_cloud_probs - local_probs  # (B, 10)
    
    # Concatenate both signals for OffloadMechanism input
    combined_input = torch.cat([logits_plus, predicted_bk], dim=1)  # (B, 22)
    return combined_input


def compute_logits_with_real_bk(local_out, cloud_out, num_classes):
    """
    Helper function to compute combined input for logits_with_real_bk mode (TESTING ONLY).
    Uses REAL cloud logits to compute actual bk.
    
    Returns:
        combined_input: shape (batch, 22) = logits_plus (12) + real_bk (10)
    """
    # Compute logits_plus
    probs = F.softmax(local_out, dim=1)
    top2 = torch.topk(probs, 2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
    entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_classes)
    logits_plus = torch.cat([local_out, margin, entropy], dim=1)  # (batch, 12)
    
    # Compute REAL bk (using actual cloud logits)
    real_cloud_probs = F.softmax(cloud_out, dim=1)
    local_probs = F.softmax(local_out, dim=1)
    real_bk = real_cloud_probs - local_probs  # (batch, 10)
    
    # Concatenate
    combined_input = torch.cat([logits_plus, real_bk], dim=1)  # (batch, 22)
    return combined_input


def compute_logits_predicted_regression(local_out, local_feats, cloud_predictor):
    """
    Compute input for logits_predicted_regression mode.
    
    This is a simplified regression approach where:
    - Input: local_logits (10) + predicted_cloud_logits (10) = 20-dim
    - Output: OffloadMechanism predicts continuous bk value
    - Decision: predicted_bk ≥ b_star → offload
    
    The key insight: we concatenate raw logits from both models,
    and let the OffloadMechanism learn to predict bk directly.
    
    Args:
        local_out: (B, num_classes) local logits
        local_feats: (B, 32, 16, 16) local feature maps
        cloud_predictor: CloudLogitPredictor model
    
    Returns:
        combined_input: (B, 20) concatenation of local_logits + predicted_cloud_logits
    """
    with torch.no_grad():
        if hasattr(cloud_predictor, 'logits_only') and cloud_predictor.logits_only:
            predicted_cloud_logits = cloud_predictor(local_out)  # FC-only mode
        else:
            predicted_cloud_logits = cloud_predictor(local_feats, local_out)  # CNN+logits mode
    
    # Concatenate local and predicted cloud logits
    combined_input = torch.cat([local_out, predicted_cloud_logits], dim=1)  # (B, 20)
    return combined_input


# ============================================================================
# OFFLOAD MECHANISM EVALUATION
# ============================================================================

def evaluate_offload_decision_accuracy_CNN_test(
    offload_mechanism,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,  
    test_loader,
    b_star,
    threshold=0.5,
    *, input_mode='feat', device='cuda',
    cloud_predictor=None,  # ★ For logits_with_bk_pred mode
    test_with_real_cloud_logits=False  # ★ NEW: Use real cloud logits in inference (for testing)
):
    """
    Evaluates the offload decisions on test data by comparing:
      1) "Real" label from the oracle labeling (uses correctness of local/cloud and bk threshold for ties)
      2) "Predicted" label from the offload mechanism (if offload_prob > threshold => 1, else 0)

    We compute bk = local_cost - cloud_cost for each test sample, then generate the "real" label
    based on whether bk >= b_star. Meanwhile, the offload mechanism predicts 1/0 via the logistic
    output. If they match, it's a correct decision.

    Args:
        offload_mechanism (nn.Module): The trained offload MLP.
        local_feature_extractor (nn.Module): Local feature extractor.
        local_classifier (nn.Module): Local classifier (for cost).
        cloud_cnn (nn.Module): Cloud classifier (for cost).
        test_loader (DataLoader): Test dataset loader (images, labels).
        b_star (float): The threshold for manually labeling each sample (bk >= b_star => label=1).
        threshold (float): Offload mechanism classification threshold (default=0.5).
        device (str): 'cuda' or 'cpu'.

    Returns:
        offload_accuracy (float): The percentage of correctly matched decisions between
                                  oracle labeling and offload mechanism output.
    """
    
    # Auto-detect regression mode from input_mode
    regression_mode = (input_mode == 'logits_predicted_regression')
    
    # Put models in eval mode
    offload_mechanism.eval()
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    correct_decisions = 0
    total_samples = 0

    # We won't backprop, so use no_grad
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total_samples += batch_size

            # 1) Extract local features
            local_feats = local_feature_extractor(images)

            # 2) Compute local and cloud logits
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            # 3) Oracle labeling (0=local, 1=cloud)
            oracle_labels = my_oracle_decision_function(local_out, cloud_out, labels, b_star=b_star).float()

            # 4) Offload mechanism prediction
            if input_mode == 'logits':
                logits = offload_mechanism(local_out)
            elif input_mode == 'img':
                logits = offload_mechanism(images)         # shape (batch, 1)
            elif input_mode in ('feat', 'shallow_feat'): 
                logits = offload_mechanism(local_feats)
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                top2  = torch.topk(probs, 2, dim=1).values
                margin  = (top2[:,0] - top2[:,1]).unsqueeze(1)        # (B,1)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1,keepdim=True) \
                        / math.log(probs.size(1))
                dom_in = torch.cat([local_out, margin, entropy], dim=1)  # (B,12)
                logits = offload_mechanism(dom_in)
            elif input_mode == 'hybrid':
                # ★ HYBRID mode: logits_plus + compressed features
                probs = F.softmax(local_out, dim=1)
                top2  = torch.topk(probs, 2, dim=1).values
                margin  = (top2[:,0] - top2[:,1]).unsqueeze(1)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1,keepdim=True) \
                        / math.log(probs.size(1))
                dom_in = torch.cat([local_out, margin, entropy], dim=1)  # (B,12)
                logits = offload_mechanism(dom_in, feat=local_feats)
            elif input_mode == 'logits_with_bk_pred':
                # ★ NEW: logits_plus + predicted_bk
                num_cls = local_out.size(1)
                combined_input = compute_logits_with_bk_pred(local_out, local_feats, cloud_predictor, num_cls)
                logits = offload_mechanism(combined_input)
            elif input_mode == 'logits_with_real_bk':
                # ★ TESTING: logits_plus + REAL bk
                num_cls = local_out.size(1)
                combined_input = compute_logits_with_real_bk(local_out, cloud_out, num_cls)
                logits = offload_mechanism(combined_input)
            elif input_mode == 'logits_predicted_regression':
                # ★ Regression approach - predicts bk directly
                # INFERENCE: Use real or predicted cloud logits based on flag
                if test_with_real_cloud_logits:
                    # TESTING: Use real cloud logits (should achieve high accuracy)
                    combined_input = torch.cat([local_out, cloud_out], dim=1)  # (B, 20)
                else:
                    # PRODUCTION: Use predicted cloud logits
                    combined_input = compute_logits_predicted_regression(local_out, local_feats, cloud_predictor)
                logits = offload_mechanism(combined_input)
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")
            
            # ★ NEW: Regression mode - compare predicted bk with b_star
            if regression_mode:
                predicted_bk = logits.squeeze(1)  # Raw bk prediction
                predicted_label = (predicted_bk >= b_star).float()
            else:
                probs = torch.sigmoid(logits).squeeze(1)
                predicted_label = (probs > threshold).float()

            # 5) Compare oracle_labels vs predicted_label
            correct_decisions += (oracle_labels == predicted_label).sum().item()

    offload_accuracy = (correct_decisions / total_samples) * 100.0
    # print(f"[evaluate_offload_decision_accuracy_CNN_test] Accuracy: {offload_accuracy:.2f}% "
        #   f"(Threshold={threshold}, b_star={b_star:.3f})")
    return offload_accuracy


# ============================================================================
# DDNN TESTING FUNCTIONS
# ============================================================================

def test_DDNN_with_oracle(
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    test_loader,
    b_star,
    L0: float = None,  # ⬅️ ΝΕΟΣ ΠΑΡΑΜΕΤΡΟΣ: target local percentage
    oracle_decision_function=my_oracle_decision_function,
    device='cuda'
):
    """
    Evaluate DDNN accuracy using oracle decisions.
    
    If L0 is provided, the oracle selects the L0% samples with the **lowest score**
    to be processed locally, and the rest are offloaded to cloud.
    
    Parameters
    ----------
    L0 : float or None
        • None (default) → use binary decisions from oracle_decision_function(b_star)
        • float (0-1) → select L0*100% samples with lowest score for local processing
    
    Returns
    -------
    oracle_acc : float
        Overall classification accuracy (%)
    oracle_pct : float
        Actual percentage of samples processed locally
    """
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    # Collect logits and labels
    all_loc_logits = []
    all_cld_logits = []
    all_labels     = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            feats = local_feature_extractor(images)
            loc_logits = local_classifier(feats)
            cld_logits = cloud_cnn(feats)
            all_loc_logits.append(loc_logits)
            all_cld_logits.append(cld_logits)
            all_labels.append(labels)

    # Concatenate batches
    all_loc = torch.cat(all_loc_logits)
    all_cld = torch.cat(all_cld_logits)
    all_lbl = torch.cat(all_labels)
    N = all_lbl.size(0)

    if L0 is not None:
        # Get continuous scores
        scores = oracle_decision_function(
            all_loc, all_cld, all_lbl,
            b_star=b_star,
            not_binary_decision=True  # ⬅️ Get scores, not binary
        )
        
        # Sort samples by score (ascending)
        sorted_indices = torch.argsort(scores)
        
        # Select L0% with lowest scores for local
        num_local = int(N * L0)
        local_indices = sorted_indices[:num_local]
        
        # Create binary decision mask
        decisions = torch.ones(N, dtype=torch.long, device=device)  # default: cloud (1)
        decisions[local_indices] = 0  # mark selected as local (0)
        
    else:
        #  Binary decision based on b_star
        decisions = oracle_decision_function(
            all_loc, all_cld, all_lbl,
            b_star=b_star,
            not_binary_decision=False
        )

    # Apply decisions
    mask = decisions.bool().to(device)
    pred_loc = torch.argmax(all_loc, dim=1)
    pred_cld = torch.argmax(all_cld, dim=1)
    preds = torch.where(mask, pred_cld, pred_loc)

    # Metrics
    correct = (preds == all_lbl).sum().item()
    acc = 100.0 * correct / N
    pct_loc = 100.0 * (~mask).float().mean().item()
    
    print(f"[Oracle] L0={L0 if L0 else 'N/A'}, Acc={acc:.2f}%, Local%={pct_loc:.2f}%")
    
    return acc, pct_loc



def evaluate_offload_decision_accuracy_CNN_train(loader, local_feature_extractor, local_classifier, cloud_cnn, deep_offload_model, b_star,threshold=0.5, *, input_mode='feat', device='cuda',num_classes=10, cloud_predictor=None, test_with_real_cloud_logits=False):
    """
    Evaluate the offload decision accuracy of the deep offload mechanism on the given dataset.
    This function:
      1. Computes a ground truth (gt) based on the DDNN predictions (using local_classifier and cloud_cnn)
         via the ORACLE labeling (not just bk threshold), and compares it with the provided labels from the offload dataset.
      2. Passes the unflattened local features through the deep offload mechanism to get its predictions
         and compares those with the computed gt.
    Returns:
      The offload decision accuracy (percentage) based on the deep offload mechanism.
    """
    # Auto-detect regression mode from input_mode
    regression_mode = (input_mode == 'logits_predicted_regression')
    
    correct_offload = 0
    total_samples = 0
    correct_gt_vs_loader = 0  # Accuracy between computed ground truth and loader labels
    
    deep_offload_model.eval()
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    with torch.no_grad():
        for x_tensor, offload_label, real_label, feature_tensor in loader:
            x_tensor   = x_tensor.to(device)
            feature_tensor = feature_tensor.to(device)
            offload_label = offload_label.to(device)
            real_label    = real_label.to(device)
            bs = real_label.size(0)

            # === 1) Υπολογίζεις local_probs, cloud_probs βάσει του ΠΡΑΓΜΑΤΙΚΟΥ label ===
            # if we are in logits mode we already have them no need to compute them again
            if input_mode == 'logits':
                local_logits = x_tensor
            elif input_mode == 'logits_plus':
                # we already have them bu we need to cut the first 10 dim because now its 12
                local_logits= x_tensor[:,:num_classes]
            else:
                local_logits = local_classifier(feature_tensor)
            
            
            
            cloud_logits = cloud_cnn(feature_tensor)

            # === 1b) Oracle labeling ===
            computed_gt = my_oracle_decision_function(local_logits, cloud_logits, real_label, b_star=b_star).float()

            # === 2) Έλεγξε αν η stored offload_label ταιριάζει με computed_gt ===
            # (ιδανικά ~100%)
            # ψιλο unsused αλλα οκευ
            correct_gt_vs_loader += (computed_gt == offload_label).sum().item()
            # if not torch.equal(computed_gt,offload_label): print('Εχουμε θεμα  στην evaluate_offload_decision_train')
            
            # === 3) Δώσε local_feats στο deep_offload_model => offload απόφαση
            if input_mode == 'hybrid':
                logits_offload = deep_offload_model(x_tensor, feat=feature_tensor)
            elif input_mode == 'logits_with_bk_pred':
                combined_input = compute_logits_with_bk_pred(local_logits, feature_tensor, cloud_predictor, num_classes)
                logits_offload = deep_offload_model(combined_input)
            elif input_mode == 'logits_with_real_bk':
                combined_input = compute_logits_with_real_bk(local_logits, cloud_logits, num_classes)
                logits_offload = deep_offload_model(combined_input)
            elif input_mode == 'logits_predicted_regression':
                # Regression approach - predicts bk directly
                if test_with_real_cloud_logits:
                    # TESTING: Use real cloud logits
                    combined_input = torch.cat([local_logits, cloud_logits], dim=1)  # (B, 20)
                else:
                    # PRODUCTION: Use predicted cloud logits
                    combined_input = compute_logits_predicted_regression(local_logits, feature_tensor, cloud_predictor)
                logits_offload = deep_offload_model(combined_input)
            else:
                logits_offload = deep_offload_model(x_tensor)
            
            # ★ NEW: Regression mode - compare predicted bk with b_star
            if regression_mode:
                pred_offload = (logits_offload.squeeze(1) >= b_star).float()
            else:
                pred_offload = (torch.sigmoid(logits_offload).squeeze(1) > threshold).float()

            # Σύγκρινε pred_offload με computed_gt
            correct_offload += (pred_offload == computed_gt).sum().item()
            total_samples   += bs

    gt_vs_loader_acc = 100.0 * correct_gt_vs_loader / total_samples
    offload_acc      = 100.0 * correct_offload / total_samples
    # print(f"Ground Truth vs OffloadDataset label accuracy: {gt_vs_loader_acc:.2f}%")
    # print(f"Deep Offload CNN decision accuracy vs ground truth: {offload_acc:.2f}%")
    return offload_acc


def analyze_border_noisy_misclassification(
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    train_loader,
    val_loader,
    test_loader,
    *,
    L0: float = 0.54,
    tau: float = 0.01,
    input_mode: str = 'logits_plus',
    offload_epochs: int = 50,
    batch_size: int = 256,
    device: str = 'cuda',
    dataset_name: str = 'cifar10',
    plot: bool = True,
    num_classes: int = 10,
    cloud_predictor=None,
    test_with_real_cloud_logits: bool = False
) -> Dict:
    """
    Two-phase analysis:
      Phase A (TRAINING data): Oracle vs bk-rule agreement (labeling quality)
      Phase B (TEST data): Misclassification rates by sample type
    """
    
    print(f"\n{'='*80}")
    print(f"BORDER/NOISY SAMPLES ANALYSIS")
    print(f"{'='*80}")
    print(f"Settings: L0={L0:.2f}, τ_border={tau}, input_mode={input_mode}")
    
    # ================================================================
    # STEP 1: Train offload mechanism
    # ================================================================
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    with torch.no_grad():
        all_features, all_bks, all_labels, all_logits, all_images = \
            compute_bks_input_for_deep_offload(
                local_feature_extractor, local_classifier, cloud_cnn,
                train_loader, method=0, device=device
            )
    
    b_star = calculate_b_star(all_bks, L0)
    
    combined_data = create_3d_data_deep(
        all_bks, all_features, all_logits, all_images, all_labels,
        input_mode=input_mode
    )
    
    offload_dataset = OffloadDatasetCNN(
        combined_data, b_star,
        input_mode=input_mode,
        include_bk=False,
        use_oracle_labels=False,
        local_clf=local_classifier,
        cloud_clf=cloud_cnn,
        device=device
    )
    
    offload_loader = DataLoader(offload_dataset, batch_size=batch_size)
    
    offload_model = OffloadMechanism(
        input_mode=input_mode,
        num_classes=num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(offload_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
    
    print(f"\nTraining offload mechanism ({input_mode} mode)...")
    train_deep_offload_mechanism(
        offload_model, val_loader, optimizer, offload_loader,
        local_feature_extractor, local_classifier, cloud_cnn,
        b_star, scheduler,
        input_mode=input_mode, device=device,
        epochs=offload_epochs, lr=1e-3, stop_threshold=0.9,
        num_classes=num_classes,
        dataset_name=dataset_name
    )
    
    # ================================================================
    # PHASE A: TRAINING DATA - Labeling Quality Analysis
    # ================================================================
    offload_model.eval()
    
    train_agree_correct = train_agree_wrong = 0
    train_disagree_correct = train_disagree_wrong = 0
    train_total = 0
    
    print(f"\n{'='*80}")
    print(f"PHASE A: Analyzing TRAINING data labeling quality")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            bs = labels.size(0)
            train_total += bs
            
            # Forward pass
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            # Oracle decision
            oracle_decisions = my_oracle_decision_function(
                local_out, cloud_out, labels, b_star=b_star
            ).float()
            
            # bk-rule decision (what we use for training)
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)
            local_prob_correct = local_probs[range(bs), labels]
            cloud_prob_correct = cloud_probs[range(bs), labels]
            bk = (1.0 - local_prob_correct) - (1.0 - cloud_prob_correct)
            bk_decisions = (bk >= b_star).float()
            
            # Agreement mask
            agreement_mask = (oracle_decisions == bk_decisions)
            
            # "Correct" here means: Oracle agrees with bk-rule
            # (not about DDNN classification, just labeling agreement)
            train_agree_correct += agreement_mask.sum().item()
            train_disagree_correct += (~agreement_mask).sum().item()
    
    train_agree_total = train_agree_correct
    train_disagree_total = train_disagree_correct
    
    train_agree_pct = 100 * train_agree_total / train_total
    train_disagree_pct = 100 * train_disagree_total / train_total
    
    print(f"\nTraining Labeling Quality:")
    print(f"  Agreement (Oracle == bk-rule): {train_agree_total}/{train_total} ({train_agree_pct:.2f}%)")
    print(f"  Disagreement (Oracle ≠ bk-rule): {train_disagree_total}/{train_total} ({train_disagree_pct:.2f}%)")
    
    # ================================================================
    # PHASE B: TEST DATA - Misclassification Analysis
    # ================================================================
    print(f"\n{'='*80}")
    print(f"PHASE B: Analyzing TEST data misclassification")
    print(f"{'='*80}")
    
    noisy_correct = noisy_wrong = 0
    border_correct = border_wrong = 0
    normal_correct = normal_wrong = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            bs = labels.size(0)
            
            # Forward pass
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            # Compute bk
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)
            local_prob_correct = local_probs[range(bs), labels]
            cloud_prob_correct = cloud_probs[range(bs), labels]
            bk = (1.0 - local_prob_correct) - (1.0 - cloud_prob_correct)
            
            # Oracle decision
            oracle_decisions = my_oracle_decision_function(
                local_out, cloud_out, labels, b_star=b_star
            ).float()
            
            # Optimized Rule decision
            if input_mode == 'logits':
                dom_in = local_out
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                num_classes = probs.size(1)
                k = min(2, num_classes)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_classes)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
            elif input_mode == 'hybrid':
                probs = F.softmax(local_out, dim=1)
                num_classes = probs.size(1)
                k = min(2, num_classes)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_classes)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_model(dom_in, feat=local_feats)
            elif input_mode == 'logits_with_bk_pred':
                # Use CloudLogitPredictor to predict cloud logits
                dom_in = compute_logits_with_bk_pred(local_out, local_feats, cloud_predictor, local_out.size(1))
                offload_logits = offload_model(dom_in)
            elif input_mode == 'logits_with_real_bk':
                # Use actual cloud logits for upper bound testing
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                logits_plus = torch.cat([local_out, margin, entropy], dim=1)
                local_probs_bk = F.softmax(local_out, dim=1)
                cloud_probs_bk = F.softmax(cloud_out, dim=1)
                real_bk = cloud_probs_bk - local_probs_bk
                dom_in = torch.cat([logits_plus, real_bk], dim=1)
                offload_logits = offload_model(dom_in)
            elif input_mode == 'logits_predicted_regression':
                # Regression approach - predicts bk directly
                if test_with_real_cloud_logits:
                    # TESTING: Use real cloud logits
                    combined_input = torch.cat([local_out, cloud_out], dim=1)  # (B, 20)
                else:
                    # PRODUCTION: Use predicted cloud logits
                    combined_input = compute_logits_predicted_regression(local_out, local_feats, cloud_predictor)
                offload_logits = offload_model(combined_input)
            elif input_mode in ('feat', 'shallow_feat'):
                dom_in = local_feats
                offload_logits = offload_model(dom_in)
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")
            
            offload_decisions = (torch.sigmoid(offload_logits).squeeze(1) > 0.5).float()
            
            # DDNN prediction
            local_preds = local_out.argmax(dim=1)
            cloud_preds = cloud_out.argmax(dim=1)
            final_preds = torch.where(offload_decisions == 0, local_preds, cloud_preds)
            
            # Classification correctness
            correct_mask = (final_preds == labels)
            
            # Categorize
            noisy_mask = (oracle_decisions != offload_decisions)
            border_mask = (bk.abs() < tau)
            normal_mask = ~(noisy_mask | border_mask)
            
            noisy_correct += correct_mask[noisy_mask].sum().item()
            noisy_wrong += (~correct_mask[noisy_mask]).sum().item()
            
            border_correct += correct_mask[border_mask].sum().item()
            border_wrong += (~correct_mask[border_mask]).sum().item()
            
            normal_correct += correct_mask[normal_mask].sum().item()
            normal_wrong += (~correct_mask[normal_mask]).sum().item()
    
    # Compute metrics
    noisy_total = noisy_correct + noisy_wrong
    border_total = border_correct + border_wrong
    normal_total = normal_correct + normal_wrong
    total = noisy_total + border_total + normal_total
    
    noisy_misclass_rate = 100 * noisy_wrong / noisy_total if noisy_total > 0 else 0.0
    border_misclass_rate = 100 * border_wrong / border_total if border_total > 0 else 0.0
    normal_misclass_rate = 100 * normal_wrong / normal_total if normal_total > 0 else 0.0
    
    print(f"\nTest Misclassification Results:")
    print(f"  Noisy: {noisy_wrong}/{noisy_total} ({noisy_misclass_rate:.2f}%)")
    print(f"  Border: {border_wrong}/{border_total} ({border_misclass_rate:.2f}%)")
    print(f"  Normal: {normal_wrong}/{normal_total} ({normal_misclass_rate:.2f}%)")
    
    # ================================================================
    # PLOTTING
    # ================================================================
    if plot:
        # PLOT 1: Test Misclassification
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        categories = ['Noisy\nSamples', 'Border\nSamples', 'Normal\nSamples']
        misclass_rates = [noisy_misclass_rate, border_misclass_rate, normal_misclass_rate]
        colors = ['#FF6B6B', '#FFA500', '#4CAF50']
        
        bars1 = ax1.bar(categories, misclass_rates, color=colors, alpha=0.8, width=0.6)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Misclassification Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Test Misclassification by Sample Type', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        for bar, rate in zip(bars1, misclass_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, rate + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        counts = [noisy_total, border_total, normal_total]
        bars2 = ax2.bar(categories, counts, color=colors, alpha=0.8, width=0.6)
        ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax2.set_title('Test Sample Distribution', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        for bar, count in zip(bars2, counts):
            pct = 100 * count / total
            ax2.text(bar.get_x() + bar.get_width()/2, count + total*0.01,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        plt.suptitle(f'Test Data Analysis (τ={tau}, L0={L0:.2f}, {dataset_name.upper()})', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        output_path1 = f'test_misclassification_{dataset_name}_L0{int(L0*100)}.png'
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved Plot 1: {output_path1}")
        plt.close(fig1)
        
        # PLOT 2: TRAINING Labeling Agreement
        fig2, ax = plt.subplots(figsize=(8, 5))
        
        categories = ['Agreement\n(Oracle == bk-rule)', 'Disagreement\n(Oracle ≠ bk-rule)']
        counts = [train_agree_total, train_disagree_total]
        percentages = [train_agree_pct, train_disagree_pct]
        colors = ['#4CAF50', '#FF6B6B']
        
        bars = ax.bar(categories, percentages, color=colors, alpha=0.8, width=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage of Training Samples (%)', fontsize=12, fontweight='bold')
        ax.set_title('Training Data Labeling Quality\n(Oracle vs bk-threshold rule)', 
                    fontsize=13, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        for bar, count, pct in zip(bars, counts, percentages):
            ax.text(bar.get_x() + bar.get_width()/2, pct + 2,
                   f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        summary_text = (
            f"Total training samples: {train_total:,}\n"
            f"Agreement rate: {train_agree_pct:.2f}% | Noise rate: {train_disagree_pct:.2f}%"
        )
        fig2.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        
        output_path2 = f'train_labeling_quality_{dataset_name}_L0{int(L0*100)}.png'
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Plot 2: {output_path2}")
        plt.close(fig2)
    
    return {
        # Training labeling quality
        'train_agree_count': train_agree_total,
        'train_disagree_count': train_disagree_total,
        'train_noise_rate': train_disagree_pct,
        # Test misclassification
        'noisy_misclass_rate': noisy_misclass_rate,
        'border_misclass_rate': border_misclass_rate,
        'normal_misclass_rate': normal_misclass_rate,
        'noisy_count': noisy_total,
        'border_count': border_total,
        'normal_count': normal_total,
        'total_count': total
    }


def analyze_oracle_optimized_gap(
    offload_mechanism,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    test_loader,
    b_star,
    *,
    L0: float = 0.54,
    tau_border: float = 0.01,
    input_mode: str = 'logits',
    device: str = 'cuda',
    dataset_name: str = 'cifar10',
    plot: bool = True,
    num_classes: int = 10,
    cloud_predictor = None,  # ★ For logits_with_bk_pred mode
    test_with_real_cloud_logits: bool = False  # ★ NEW: Use real cloud logits in inference
) -> Dict:
    """
    Comprehensive analysis to explain the Oracle vs Optimized Rule performance gap.
    
    Categorizes test samples into:
      1. **Disagreement samples** (Oracle ≠ bk-rule) - "noisy labels"
      2. **Borderline samples** (|bk| < τ_border) - ambiguous cases
      3. **Normal samples** - clear cases
      4. **Rational failures** (coin toss) - disagreement where optimized rule's choice 
         was statistically reasonable but wrong due to not knowing classification result
    
    The key insight: Oracle knows the actual classification outcome (ground truth),
    so it can make "lucky" decisions that the optimized rule cannot replicate.
    This is like predicting coin tosses - before tossing, any choice is equally good,
    but after tossing, the oracle knows which were heads/tails.
    
    Parameters
    ----------
    tau_border : float
        Threshold for borderline samples (|bk| < tau_border)
    
    Returns
    -------
    dict
        Detailed metrics for each sample category including misclassification rates
    """
    
    # Auto-detect regression mode from input_mode
    regression_mode = (input_mode == 'logits_predicted_regression')
    
    print(f"\n{'='*80}")
    print(f"ORACLE vs OPTIMIZED RULE GAP ANALYSIS")
    print(f"{'='*80}")
    print(f"Settings: L0={L0:.2f}, τ_border={tau_border}, input_mode={input_mode}")
    
    offload_mechanism.eval()
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    # Counters for each category
    disagreement_correct = disagreement_wrong = 0
    borderline_correct = borderline_wrong = 0
    normal_correct = normal_wrong = 0
    rational_failure_correct = rational_failure_wrong = 0
    
    total_samples = 0
    oracle_correct_total = optimized_correct_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            bs = labels.size(0)
            total_samples += bs
            
            # Forward pass
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            # Compute bk
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)
            local_prob_correct = local_probs[range(bs), labels]
            cloud_prob_correct = cloud_probs[range(bs), labels]
            bk = (1.0 - local_prob_correct) - (1.0 - cloud_prob_correct)
            
            # Oracle decision (knows ground truth)
            oracle_decisions = my_oracle_decision_function(
                local_out, cloud_out, labels, b_star=b_star
            ).float()
            
            # bk-rule decision
            bk_decisions = (bk >= b_star).float()
            
            # Optimized Rule decision
            if input_mode == 'logits':
                dom_in = local_out
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'hybrid':
                # ★ HYBRID mode: logits_plus + compressed features
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_mechanism(dom_in, feat=local_feats)
            elif input_mode == 'logits_with_bk_pred':
                # ★ NEW: logits_plus + predicted bk from CloudLogitPredictor
                num_cls = local_out.size(1)
                dom_in = compute_logits_with_bk_pred(local_out, local_feats, cloud_predictor, num_cls)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_with_real_bk':
                # ★ TESTING: logits_plus + REAL bk (using actual cloud logits)
                num_cls = local_out.size(1)
                dom_in = compute_logits_with_real_bk(local_out, cloud_out, num_cls)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_predicted_regression':
                # Regression approach - predicts bk directly
                if test_with_real_cloud_logits:
                    # TESTING: Use real cloud logits
                    combined_input = torch.cat([local_out, cloud_out], dim=1)  # (B, 20)
                else:
                    # PRODUCTION: Use predicted cloud logits
                    combined_input = compute_logits_predicted_regression(local_out, local_feats, cloud_predictor)
                offload_logits = offload_mechanism(combined_input)
            elif input_mode in ('feat', 'shallow_feat'):
                dom_in = local_feats
                offload_logits = offload_mechanism(dom_in)
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")
            
            # ★ Regression mode - compare predicted bk with b_star
            if regression_mode:
                optimized_decisions = (offload_logits.squeeze(1) >= b_star).float()
            else:
                optimized_decisions = (torch.sigmoid(offload_logits).squeeze(1) > 0.5).float()
            
            # Predictions
            local_preds = local_out.argmax(dim=1)
            cloud_preds = cloud_out.argmax(dim=1)
            
            # Oracle DDNN result
            oracle_final_preds = torch.where(oracle_decisions == 0, local_preds, cloud_preds)
            oracle_correct_mask = (oracle_final_preds == labels)
            oracle_correct_total += oracle_correct_mask.sum().item()
            
            # Optimized Rule DDNN result
            optimized_final_preds = torch.where(optimized_decisions == 0, local_preds, cloud_preds)
            optimized_correct_mask = (optimized_final_preds == labels)
            optimized_correct_total += optimized_correct_mask.sum().item()
            
            # ============================================================
            # CATEGORIZATION (WITH OVERLAP - this is OK!)
            # ============================================================
            # Categories can overlap - e.g., rational failures are also disagreements
            # This is intentional and useful for analysis
            
            # 1. Disagreement samples (Oracle ≠ bk-rule)
            disagreement_mask = (oracle_decisions != bk_decisions)
            
            # 2. Borderline samples (|bk| < τ)
            borderline_mask = (bk.abs() < tau_border)
            
            # 3. Rational failures (coin toss cases)
            # Definition: Oracle and Optimized Rule disagree, BUT the optimized rule's
            # decision was statistically reasonable (it followed the bk-rule)
            # The failure is "rational" because without knowing the ground truth,
            # the optimized rule made a sensible choice that happened to be wrong
            rational_failure_mask = (
                (oracle_decisions != optimized_decisions) &  # They disagree
                (optimized_decisions == bk_decisions)         # Optimized followed bk-rule
            )
            
            # 4. Normal samples (clear, non-borderline, non-disagreement cases)
            normal_mask = ~(disagreement_mask | borderline_mask)
            
            # Count correct/wrong for each category (based on Optimized Rule performance)
            disagreement_correct += optimized_correct_mask[disagreement_mask].sum().item()
            disagreement_wrong += (~optimized_correct_mask[disagreement_mask]).sum().item()
            
            borderline_correct += optimized_correct_mask[borderline_mask].sum().item()
            borderline_wrong += (~optimized_correct_mask[borderline_mask]).sum().item()
            
            rational_failure_correct += optimized_correct_mask[rational_failure_mask].sum().item()
            rational_failure_wrong += (~optimized_correct_mask[rational_failure_mask]).sum().item()
            
            normal_correct += optimized_correct_mask[normal_mask].sum().item()
            normal_wrong += (~optimized_correct_mask[normal_mask]).sum().item()
    
    # Compute totals and rates
    disagreement_total = disagreement_correct + disagreement_wrong
    borderline_total = borderline_correct + borderline_wrong
    rational_failure_total = rational_failure_correct + rational_failure_wrong
    normal_total = normal_correct + normal_wrong
    
    # Track overlaps (for informational purposes)
    # Count how many samples are in multiple categories
    with torch.no_grad():
        all_disagreement_mask = []
        all_borderline_mask = []
        all_rational_mask = []
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            bs = labels.size(0)
            
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)
            local_prob_correct = local_probs[range(bs), labels]
            cloud_prob_correct = cloud_probs[range(bs), labels]
            bk = (1.0 - local_prob_correct) - (1.0 - cloud_prob_correct)
            
            oracle_decisions = my_oracle_decision_function(
                local_out, cloud_out, labels, b_star=b_star
            ).float()
            bk_decisions = (bk >= b_star).float()
            
            if input_mode == 'logits':
                dom_in = local_out
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'hybrid':
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_mechanism(dom_in, feat=local_feats)
            elif input_mode == 'logits_with_bk_pred':
                # Use CloudLogitPredictor to predict cloud logits
                dom_in = compute_logits_with_bk_pred(local_out, local_feats, cloud_predictor, num_classes)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_with_real_bk':
                # Use actual cloud logits for upper bound testing
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                logits_plus = torch.cat([local_out, margin, entropy], dim=1)
                local_probs_bk = F.softmax(local_out, dim=1)
                cloud_probs_bk = F.softmax(cloud_out, dim=1)
                real_bk = cloud_probs_bk - local_probs_bk
                dom_in = torch.cat([logits_plus, real_bk], dim=1)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_predicted_regression':
                # Regression approach - predicts bk directly
                if test_with_real_cloud_logits:
                    # TESTING: Use real cloud logits
                    combined_input = torch.cat([local_out, cloud_out], dim=1)  # (B, 20)
                else:
                    # PRODUCTION: Use predicted cloud logits
                    combined_input = compute_logits_predicted_regression(local_out, local_feats, cloud_predictor)
                offload_logits = offload_mechanism(combined_input)
            elif input_mode in ('feat', 'shallow_feat'):
                dom_in = local_feats
                offload_logits = offload_mechanism(dom_in)
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")
            
            optimized_decisions = (torch.sigmoid(offload_logits).squeeze(1) > 0.5).float()
            
            disagreement = (oracle_decisions != bk_decisions)
            borderline = (bk.abs() < tau_border)
            rational = (
                (oracle_decisions != optimized_decisions) &
                (optimized_decisions == bk_decisions)
            )
            
            all_disagreement_mask.append(disagreement.cpu())
            all_borderline_mask.append(borderline.cpu())
            all_rational_mask.append(rational.cpu())
        
        all_disagreement = torch.cat(all_disagreement_mask)
        all_borderline = torch.cat(all_borderline_mask)
        all_rational = torch.cat(all_rational_mask)
        
        # Count overlaps
        borderline_AND_disagreement = (all_borderline & all_disagreement).sum().item()
        borderline_AND_rational = (all_borderline & all_rational).sum().item()
        rational_AND_disagreement = (all_rational & all_disagreement).sum().item()
        all_three = (all_borderline & all_rational & all_disagreement).sum().item()
    
    disagreement_misclass = 100 * disagreement_wrong / disagreement_total if disagreement_total > 0 else 0.0
    borderline_misclass = 100 * borderline_wrong / borderline_total if borderline_total > 0 else 0.0
    rational_failure_misclass = 100 * rational_failure_wrong / rational_failure_total if rational_failure_total > 0 else 0.0
    normal_misclass = 100 * normal_wrong / normal_total if normal_total > 0 else 0.0
    
    oracle_acc = 100 * oracle_correct_total / total_samples
    optimized_acc = 100 * optimized_correct_total / total_samples
    gap = oracle_acc - optimized_acc
    
    # ============================================================
    # GAP DECOMPOSITION ANALYSIS
    # ============================================================
    # Calculate how much of the gap is explained by rational failures and borderline samples
    # CRITICAL: Remove overlaps to avoid double-counting in adjustments
    
    # Current optimized misclassifications
    optimized_wrong_total = total_samples - optimized_correct_total
    optimized_error_rate = 100 * optimized_wrong_total / total_samples
    
    # Count UNIQUE misclassifications (without overlap)
    # We need to go back through the data to count correctly
    rational_only_wrong = 0
    borderline_only_wrong = 0
    both_rational_and_borderline_wrong = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            bs = labels.size(0)
            
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)
            local_prob_correct = local_probs[range(bs), labels]
            cloud_prob_correct = cloud_probs[range(bs), labels]
            bk = (1.0 - local_prob_correct) - (1.0 - cloud_prob_correct)
            
            oracle_decisions = my_oracle_decision_function(
                local_out, cloud_out, labels, b_star=b_star
            ).float()
            bk_decisions = (bk >= b_star).float()
            
            if input_mode == 'logits':
                dom_in = local_out
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'hybrid':
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_mechanism(dom_in, feat=local_feats)
            elif input_mode == 'logits_with_bk_pred':
                # Use CloudLogitPredictor to predict cloud logits
                dom_in = compute_logits_with_bk_pred(local_out, local_feats, cloud_predictor, num_classes)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_with_real_bk':
                # Use actual cloud logits for upper bound testing
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                logits_plus = torch.cat([local_out, margin, entropy], dim=1)
                local_probs_bk = F.softmax(local_out, dim=1)
                cloud_probs_bk = F.softmax(cloud_out, dim=1)
                real_bk = cloud_probs_bk - local_probs_bk
                dom_in = torch.cat([logits_plus, real_bk], dim=1)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_predicted_regression':
                # Regression approach - predicts bk directly
                if test_with_real_cloud_logits:
                    # TESTING: Use real cloud logits
                    combined_input = torch.cat([local_out, cloud_out], dim=1)  # (B, 20)
                else:
                    # PRODUCTION: Use predicted cloud logits
                    combined_input = compute_logits_predicted_regression(local_out, local_feats, cloud_predictor)
                offload_logits = offload_mechanism(combined_input)
            elif input_mode in ('feat', 'shallow_feat'):
                dom_in = local_feats
                offload_logits = offload_mechanism(dom_in)
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")
            
            optimized_decisions = (torch.sigmoid(offload_logits).squeeze(1) > 0.5).float()
            
            local_preds = local_out.argmax(dim=1)
            cloud_preds = cloud_out.argmax(dim=1)
            optimized_final_preds = torch.where(optimized_decisions == 0, local_preds, cloud_preds)
            is_wrong = (optimized_final_preds != labels)
            
            rational = (
                (oracle_decisions != optimized_decisions) &
                (optimized_decisions == bk_decisions)
            )
            borderline = (bk.abs() < tau_border)
            
            # Count UNIQUE wrong samples (no overlap)
            rational_only_wrong += (is_wrong & rational & ~borderline).sum().item()
            borderline_only_wrong += (is_wrong & borderline & ~rational).sum().item()
            both_rational_and_borderline_wrong += (is_wrong & rational & borderline).sum().item()
    
    # If we remove ONLY rational failures from misclassifications
    # (these are "justified" errors - coin toss scenarios)
    adjusted_wrong_without_rational = optimized_wrong_total - rational_only_wrong - both_rational_and_borderline_wrong
    adjusted_acc_without_rational = 100 * (total_samples - adjusted_wrong_without_rational) / total_samples
    gap_explained_by_rational = adjusted_acc_without_rational - optimized_acc
    
    # If we also remove borderline samples from misclassifications
    # (these are inherently ambiguous cases)
    # This removes: rational_only + borderline_only + both
    adjusted_wrong_without_both = optimized_wrong_total - rational_only_wrong - borderline_only_wrong - both_rational_and_borderline_wrong
    adjusted_acc_without_both = 100 * (total_samples - adjusted_wrong_without_both) / total_samples
    gap_explained_by_both = adjusted_acc_without_both - optimized_acc
    
    # Remaining unexplained gap
    remaining_gap_after_rational = oracle_acc - adjusted_acc_without_rational
    remaining_gap_after_both = oracle_acc - adjusted_acc_without_both
    
    # Percentage of gap explained
    pct_gap_explained_by_rational = (gap_explained_by_rational / gap * 100) if gap > 0 else 0.0
    pct_gap_explained_by_both = (gap_explained_by_both / gap * 100) if gap > 0 else 0.0
    
    # Print results
    print(f"\n{'='*80}")
    print(f"OVERALL PERFORMANCE")
    print(f"{'='*80}")
    print(f"  Oracle Accuracy:      {oracle_acc:.2f}%")
    print(f"  Optimized Rule Acc:   {optimized_acc:.2f}%")
    print(f"  Performance Gap:      {gap:.2f}% ⬅️ TO BE EXPLAINED")
    
    print(f"\n{'='*80}")
    print(f"SAMPLE CATEGORIZATION & MISCLASSIFICATION RATES")
    print(f"{'='*80}")
    print(f"  Disagreement samples:  {disagreement_total:5d} ({100*disagreement_total/total_samples:5.1f}%) | Misclass: {disagreement_misclass:5.1f}%")
    print(f"  Borderline samples:    {borderline_total:5d} ({100*borderline_total/total_samples:5.1f}%) | Misclass: {borderline_misclass:5.1f}%")
    print(f"    ├─ Also disagreement: {borderline_AND_disagreement} ({100*borderline_AND_disagreement/borderline_total:.1f}% of borderline)")
    print(f"    └─ Also rational:     {borderline_AND_rational} ({100*borderline_AND_rational/borderline_total:.1f}% of borderline)")
    print(f"  Rational failures:     {rational_failure_total:5d} ({100*rational_failure_total/total_samples:5.1f}%) | Misclass: {rational_failure_misclass:5.1f}%")
    print(f"    ├─ Also disagreement: {rational_AND_disagreement} ({100*rational_AND_disagreement/rational_failure_total:.1f}% of rational)")
    print(f"    └─ Also borderline:   {borderline_AND_rational} ({100*borderline_AND_rational/rational_failure_total:.1f}% of rational)")
    print(f"  Normal samples:        {normal_total:5d} ({100*normal_total/total_samples:5.1f}%) | Misclass: {normal_misclass:5.1f}%")
    print(f"\n  Note: Categories may overlap (e.g., rational failures are always disagreements)")
    print(f"        All three overlap: {all_three} samples")
    
    # DETAILED DISAGREEMENT ANALYSIS
    print(f"\n{'='*80}")
    print(f"DETAILED DISAGREEMENT ANALYSIS (Oracle ≠ bk-rule)")
    print(f"{'='*80}")
    
    disagree_oracle_local = disagree_oracle_cloud = 0
    disagree_opt_local = disagree_opt_cloud = 0
    disagree_bk_positive = disagree_bk_negative = 0
    disagree_bk_values = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            bs = labels.size(0)
            
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)
            local_prob_correct = local_probs[range(bs), labels]
            cloud_prob_correct = cloud_probs[range(bs), labels]
            bk = (1.0 - local_prob_correct) - (1.0 - cloud_prob_correct)
            
            oracle_decisions = my_oracle_decision_function(
                local_out, cloud_out, labels, b_star=b_star
            ).float()
            bk_decisions = (bk >= b_star).float()
            
            if input_mode == 'logits':
                dom_in = local_out
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'hybrid':
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_mechanism(dom_in, feat=local_feats)
            elif input_mode == 'logits_with_bk_pred':
                # Use CloudLogitPredictor to predict cloud logits
                dom_in = compute_logits_with_bk_pred(local_out, local_feats, cloud_predictor, num_classes)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_with_real_bk':
                # Use actual cloud logits for upper bound testing
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                logits_plus = torch.cat([local_out, margin, entropy], dim=1)
                local_probs_bk = F.softmax(local_out, dim=1)
                cloud_probs_bk = F.softmax(cloud_out, dim=1)
                real_bk = cloud_probs_bk - local_probs_bk
                dom_in = torch.cat([logits_plus, real_bk], dim=1)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_predicted_regression':
                # Regression approach - predicts bk directly
                if test_with_real_cloud_logits:
                    # TESTING: Use real cloud logits
                    combined_input = torch.cat([local_out, cloud_out], dim=1)  # (B, 20)
                else:
                    # PRODUCTION: Use predicted cloud logits
                    combined_input = compute_logits_predicted_regression(local_out, local_feats, cloud_predictor)
                offload_logits = offload_mechanism(combined_input)
            elif input_mode in ('feat', 'shallow_feat'):
                dom_in = local_feats
                offload_logits = offload_mechanism(dom_in)
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")
            
            optimized_decisions = (torch.sigmoid(offload_logits).squeeze(1) > 0.5).float()
            
            # Analyze disagreements
            disagree_mask = (oracle_decisions != bk_decisions)
            
            if disagree_mask.any():
                disagree_oracle_local += (oracle_decisions[disagree_mask] == 0).sum().item()
                disagree_oracle_cloud += (oracle_decisions[disagree_mask] == 1).sum().item()
                disagree_opt_local += (optimized_decisions[disagree_mask] == 0).sum().item()
                disagree_opt_cloud += (optimized_decisions[disagree_mask] == 1).sum().item()
                disagree_bk_positive += (bk[disagree_mask] >= 0).sum().item()
                disagree_bk_negative += (bk[disagree_mask] < 0).sum().item()
                disagree_bk_values.extend(bk[disagree_mask].cpu().tolist())
    
    import numpy as np
    bk_array = np.array(disagree_bk_values)
    
    print(f"  Total disagreements: {disagreement_total}")
    print(f"\n  Oracle choices in disagreements:")
    print(f"    Local:  {disagree_oracle_local} ({100*disagree_oracle_local/disagreement_total:.1f}%)")
    print(f"    Cloud:  {disagree_oracle_cloud} ({100*disagree_oracle_cloud/disagreement_total:.1f}%)")
    print(f"\n  Optimized choices in disagreements:")
    print(f"    Local:  {disagree_opt_local} ({100*disagree_opt_local/disagreement_total:.1f}%)")
    print(f"    Cloud:  {disagree_opt_cloud} ({100*disagree_opt_cloud/disagreement_total:.1f}%)")
    print(f"\n  bk values in disagreements:")
    print(f"    Positive (bk≥0): {disagree_bk_positive} ({100*disagree_bk_positive/disagreement_total:.1f}%)")
    print(f"    Negative (bk<0): {disagree_bk_negative} ({100*disagree_bk_negative/disagreement_total:.1f}%)")
    print(f"    Mean: {bk_array.mean():.4f}, Std: {bk_array.std():.4f}")
    print(f"    Min: {bk_array.min():.4f}, Max: {bk_array.max():.4f}")
    print(f"    Median: {np.median(bk_array):.4f}")
    print(f"\n  💡 Current b_star threshold: {b_star:.4f}")
    print(f"  💡 If most disagreements have |bk| small, they are 'coin toss' scenarios!")
    
    # ALTERNATIVE DEFINITION ANALYSIS
    print(f"\n{'='*80}")
    print(f"ALTERNATIVE RATIONAL FAILURE DEFINITION")
    print(f"{'='*80}")
    print(f"  Current definition: (Oracle ≠ Optimized) AND (Optimized = bk-rule)")
    print(f"  → Only {rational_failure_total} samples")
    print(f"\n  PROPOSAL: Use disagreements where |bk| is small (ambiguous cases)")
    print(f"  → Rational failures = (Oracle ≠ bk-rule) AND (|bk| < threshold)")
    
    # Try different thresholds
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        alt_rational = (np.abs(bk_array) < threshold).sum()
        print(f"     With |bk| < {threshold:.2f}: {alt_rational} samples ({100*alt_rational/disagreement_total:.1f}% of disagreements)")
    
    print(f"\n  💡 RECOMMENDATION: Use |bk| < 0.10 or 0.15 as 'rational failure' threshold")
    print(f"     This captures cases where the choice between local/cloud is genuinely ambiguous")
    
    # ============================================================
    # BK-RULE QUALITY ANALYSIS
    # ============================================================
    print(f"\n{'='*80}")
    print(f"BK-RULE QUALITY ANALYSIS")
    print(f"{'='*80}")
    print(f"  The bk-rule uses: bk = (1 - P_local_correct) - (1 - P_cloud_correct)")
    print(f"                      = P_cloud_correct - P_local_correct")
    print(f"  Decision: bk ≥ b_star → cloud, else → local")
    
    bk_correct_local = bk_correct_cloud = 0
    bk_wrong_local = bk_wrong_cloud = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            bs = labels.size(0)
            
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)
            local_prob_correct = local_probs[range(bs), labels]
            cloud_prob_correct = cloud_probs[range(bs), labels]
            bk = (1.0 - local_prob_correct) - (1.0 - cloud_prob_correct)
            
            bk_decisions = (bk >= b_star).float()
            
            # Get actual predictions
            local_preds = local_out.argmax(dim=1)
            cloud_preds = cloud_out.argmax(dim=1)
            
            # Apply bk-rule decisions
            bk_final_preds = torch.where(bk_decisions == 0, local_preds, cloud_preds)
            bk_correct = (bk_final_preds == labels)
            
            # Breakdown by decision
            bk_chose_local = (bk_decisions == 0)
            bk_chose_cloud = (bk_decisions == 1)
            
            bk_correct_local += (bk_correct & bk_chose_local).sum().item()
            bk_wrong_local += (~bk_correct & bk_chose_local).sum().item()
            bk_correct_cloud += (bk_correct & bk_chose_cloud).sum().item()
            bk_wrong_cloud += (~bk_correct & bk_chose_cloud).sum().item()
    
    bk_total_local = bk_correct_local + bk_wrong_local
    bk_total_cloud = bk_correct_cloud + bk_wrong_cloud
    bk_overall_acc = 100 * (bk_correct_local + bk_correct_cloud) / total_samples
    
    print(f"\n  bk-rule performance:")
    print(f"    Overall accuracy: {bk_overall_acc:.2f}%")
    print(f"    Local decisions: {bk_total_local} ({100*bk_total_local/total_samples:.1f}%)")
    print(f"      ├─ Correct: {bk_correct_local} ({100*bk_correct_local/bk_total_local:.1f}%)")
    print(f"      └─ Wrong:   {bk_wrong_local} ({100*bk_wrong_local/bk_total_local:.1f}%)")
    print(f"    Cloud decisions: {bk_total_cloud} ({100*bk_total_cloud/total_samples:.1f}%)")
    print(f"      ├─ Correct: {bk_correct_cloud} ({100*bk_correct_cloud/bk_total_cloud:.1f}%)")
    print(f"      └─ Wrong:   {bk_wrong_cloud} ({100*bk_wrong_cloud/bk_total_cloud:.1f}%)")
    
    print(f"\n  Comparison:")
    print(f"    Oracle:    {oracle_acc:.2f}% (knows ground truth)")
    print(f"    Optimized: {optimized_acc:.2f}% (learned from bk-rule)")
    print(f"    bk-rule:   {bk_overall_acc:.2f}% (probability-based)")
    
    if optimized_acc > bk_overall_acc:
        print(f"\n  ✅ GOOD NEWS: Optimized outperforms bk-rule by {optimized_acc - bk_overall_acc:.2f}%!")
        print(f"     The network learned to deviate from bk-rule in beneficial ways")
    else:
        print(f"\n  ⚠️  WARNING: bk-rule performs better than optimized!")
        print(f"     The network may be undertrained or the input features are insufficient")
    
    print(f"\n{'='*80}")
    print(f"GAP DECOMPOSITION ANALYSIS")
    print(f"{'='*80}")
    print(f"  Current Optimized Accuracy:                    {optimized_acc:.2f}%")
    print(f"  Adjusted Accuracy (w/o rational failures):     {adjusted_acc_without_rational:.2f}% (+{gap_explained_by_rational:.2f}%)")
    print(f"  Adjusted Accuracy (w/o rational + borderline): {adjusted_acc_without_both:.2f}% (+{gap_explained_by_both:.2f}%)")
    print(f"  Oracle Accuracy (theoretical upper bound):     {oracle_acc:.2f}%")
    print(f"")
    print(f"  Gap explained by rational failures:     {gap_explained_by_rational:.2f}% ({pct_gap_explained_by_rational:.1f}% of total gap)")
    print(f"  Gap explained by rational + borderline: {gap_explained_by_both:.2f}% ({pct_gap_explained_by_both:.1f}% of total gap)")
    print(f"  Remaining unexplained gap:              {remaining_gap_after_both:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"KEY INSIGHT:")
    print(f"{'='*80}")
    print(f"  Rational failures represent 'coin toss' scenarios where the optimized")
    print(f"  rule made a statistically sound decision but was wrong because it")
    print(f"  doesn't know the classification outcome (unlike the oracle).")
    print(f"  ")
    print(f"  By removing these 'justified errors' from the optimized rule's performance,")
    print(f"  we see that {pct_gap_explained_by_rational:.1f}% of the gap is due to rational failures alone,")
    print(f"  and {pct_gap_explained_by_both:.1f}% when including borderline samples.")
    print(f"  ")
    print(f"  This demonstrates that the gap is NOT due to poor model design, but rather")
    print(f"  the inherent advantage of oracle having privileged information (ground truth).")
    
    # ================================================================
    # PLOTTING
    # ================================================================
    if plot:
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        # PLOT 1: Misclassification rates
        categories = ['Disagreement\nSamples', 'Borderline\nSamples', 
                     'Rational\nFailures', 'Normal\nSamples']
        misclass_rates = [disagreement_misclass, borderline_misclass, 
                         rational_failure_misclass, normal_misclass]
        colors = ['#FF6B6B', '#FFA500', '#9B59B6', '#4CAF50']
        
        bars1 = ax1.bar(categories, misclass_rates, color=colors, alpha=0.8, width=0.6)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Misclassification Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Misclassification Rates by Sample Category', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        for bar, rate in zip(bars1, misclass_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, rate + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # PLOT 2: Sample distribution (STACKED to show overlaps)
        counts = [disagreement_total, borderline_total, rational_failure_total, normal_total]
        bars2 = ax2.bar(categories, counts, color=colors, alpha=0.8, width=0.6)
        ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax2.set_title('Sample Distribution by Category\n(Categories may overlap)', 
                     fontsize=13, fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add count labels with percentage of TOTAL (not sum, because of overlap)
        for bar, count, cat in zip(bars2, counts, categories):
            pct = 100 * count / total_samples
            ax2.text(bar.get_x() + bar.get_width()/2, count + total_samples*0.01,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        # Add note about overlap at bottom
        note_text = f"Total unique samples: {total_samples:,}\n"
        note_text += f"Note: Rational failures ⊆ Disagreements, some overlap with Borderline"
        ax2.text(0.5, -0.15, note_text, transform=ax2.transAxes,
                ha='center', va='top', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        
        # PLOT 3: Gap Decomposition (NEW!)
        gap_stages = ['Optimized\nRule', 'w/o Rational\nFailures', 'w/o Rational\n+ Borderline', 'Oracle\n(Upper Bound)']
        accuracies = [optimized_acc, adjusted_acc_without_rational, adjusted_acc_without_both, oracle_acc]
        stage_colors = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB']
        
        bars3 = ax3.bar(gap_stages, accuracies, color=stage_colors, alpha=0.8, width=0.6)
        ax3.set_ylim(min(accuracies) - 5, 100)
        ax3.set_ylabel('DDNN Accuracy (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Gap Decomposition Analysis', fontsize=13, fontweight='bold')
        ax3.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add accuracy labels on bars
        for bar, acc in zip(bars3, accuracies):
            ax3.text(bar.get_x() + bar.get_width()/2, acc + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
        
        # Add arrows showing gap reduction
        arrow_props = dict(arrowstyle='->', lw=2, color='black', alpha=0.6)
        
        # Arrow 1: Optimized → w/o Rational
        ax3.annotate('', xy=(1, adjusted_acc_without_rational - 0.5), 
                    xytext=(0, optimized_acc + 0.5),
                    arrowprops=arrow_props)
        # Show breakdown of rational failures
        rational_breakdown = f'+{gap_explained_by_rational:.2f}%\n({pct_gap_explained_by_rational:.0f}% of gap)\n'
        rational_breakdown += f'{rational_only_wrong + both_rational_and_borderline_wrong} errors removed'
        ax3.text(0.5, (optimized_acc + adjusted_acc_without_rational)/2,
                rational_breakdown,
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Arrow 2: w/o Rational → w/o Both
        ax3.annotate('', xy=(2, adjusted_acc_without_both - 0.5),
                    xytext=(1, adjusted_acc_without_rational + 0.5),
                    arrowprops=arrow_props)
        borderline_contrib = gap_explained_by_both - gap_explained_by_rational
        borderline_breakdown = f'+{borderline_contrib:.2f}%\n{borderline_only_wrong} errors removed'
        ax3.text(1.5, (adjusted_acc_without_rational + adjusted_acc_without_both)/2,
                borderline_breakdown,
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # Arrow 3: w/o Both → Oracle (remaining gap)
        if remaining_gap_after_both > 0.5:
            ax3.annotate('', xy=(3, oracle_acc - 0.5),
                        xytext=(2, adjusted_acc_without_both + 0.5),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.6))
            ax3.text(2.5, (adjusted_acc_without_both + oracle_acc)/2,
                    f'Remaining\n{remaining_gap_after_both:.2f}%',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        # Add gap info to title
        plt.suptitle(
            f'Oracle vs Optimized Rule Gap Analysis (Gap={gap:.2f}%, {pct_gap_explained_by_both:.0f}% Explained, τ={tau_border}, L0={L0:.2f}, {dataset_name.upper()})', 
            fontsize=15, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        output_path = f'oracle_optimized_gap_{dataset_name}_L0{int(L0*100)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot: {output_path}")
        plt.close(fig)
    
    return {
        'oracle_acc': oracle_acc,
        'optimized_acc': optimized_acc,
        'gap': gap,
        'disagreement_misclass': disagreement_misclass,
        'borderline_misclass': borderline_misclass,
        'rational_failure_misclass': rational_failure_misclass,
        'normal_misclass': normal_misclass,
        'disagreement_count': disagreement_total,
        'borderline_count': borderline_total,
        'rational_failure_count': rational_failure_total,
        'normal_count': normal_total,
        'total_count': total_samples,
        # Overlap statistics
        'borderline_AND_disagreement': borderline_AND_disagreement,
        'borderline_AND_rational': borderline_AND_rational,
        'rational_AND_disagreement': rational_AND_disagreement,
        'all_three_overlap': all_three,
        # Gap decomposition metrics (NO OVERLAP in adjustments)
        'adjusted_acc_without_rational': adjusted_acc_without_rational,
        'adjusted_acc_without_both': adjusted_acc_without_both,
        'gap_explained_by_rational': gap_explained_by_rational,
        'gap_explained_by_both': gap_explained_by_both,
        'pct_gap_explained_by_rational': pct_gap_explained_by_rational,
        'pct_gap_explained_by_both': pct_gap_explained_by_both,
        'remaining_gap_after_both': remaining_gap_after_both,
        # Breakdown for unique adjustments
        'rational_only_wrong': rational_only_wrong,
        'borderline_only_wrong': borderline_only_wrong,
        'both_rational_and_borderline_wrong': both_rational_and_borderline_wrong
    }


def analyze_borderline_threshold_sensitivity(
    offload_mechanism,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    test_loader,
    b_star,
    *,
    L0: float = 0.54,
    tau_values: List[float] = None,
    input_mode: str = 'logits',
    device: str = 'cuda',
    dataset_name: str = 'cifar10',
    plot: bool = True,
    num_classes: int = 10,
    cloud_predictor=None,
    test_with_real_cloud_logits: bool = False  # ★ NEW: Use real cloud logits in inference
) -> Dict:
    """
    Analyze how different borderline thresholds affect classification performance.
    
    For each τ threshold, categorize samples as borderline (|bk| < τ) and measure
    their misclassification rates to determine where "borderline" truly begins.
    
    Parameters
    ----------
    tau_values : List[float]
        List of threshold values to test (e.g., [0.01, 0.05, 0.10, 0.15, 0.20])
        If None, defaults to [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    Returns
    -------
    dict
        Results for each threshold including misclassification rates
    """
    
    # Auto-detect regression mode from input_mode
    regression_mode = (input_mode == 'logits_predicted_regression')
    
    if tau_values is None:
        tau_values = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    print(f"\n{'='*80}")
    print(f"BORDERLINE THRESHOLD SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Testing thresholds: {tau_values}")
    print(f"Settings: L0={L0:.2f}, input_mode={input_mode}")
    
    offload_mechanism.eval()
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    # Store all data first
    all_bks = []
    all_optimized_decisions = []
    all_optimized_correct = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            bs = labels.size(0)
            
            # Forward pass
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            # Compute bk
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)
            local_prob_correct = local_probs[range(bs), labels]
            cloud_prob_correct = cloud_probs[range(bs), labels]
            bk = (1.0 - local_prob_correct) - (1.0 - cloud_prob_correct)
            
            # Optimized Rule decision
            if input_mode == 'logits':
                dom_in = local_out
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'hybrid':
                # ★ HYBRID mode
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                offload_logits = offload_mechanism(dom_in, feat=local_feats)
            elif input_mode == 'logits_with_bk_pred':
                # Use CloudLogitPredictor to predict cloud logits
                dom_in = compute_logits_with_bk_pred(local_out, local_feats, cloud_predictor, local_out.size(1))
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_with_real_bk':
                # Use actual cloud logits for upper bound testing
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                logits_plus = torch.cat([local_out, margin, entropy], dim=1)
                local_probs_bk = F.softmax(local_out, dim=1)
                cloud_probs_bk = F.softmax(cloud_out, dim=1)
                real_bk = cloud_probs_bk - local_probs_bk
                dom_in = torch.cat([logits_plus, real_bk], dim=1)
                offload_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_predicted_regression':
                # Regression approach - predicts bk directly
                if test_with_real_cloud_logits:
                    # TESTING: Use real cloud logits
                    combined_input = torch.cat([local_out, cloud_out], dim=1)  # (B, 20)
                else:
                    # PRODUCTION: Use predicted cloud logits
                    combined_input = compute_logits_predicted_regression(local_out, local_feats, cloud_predictor)
                offload_logits = offload_mechanism(combined_input)
            elif input_mode in ('feat', 'shallow_feat'):
                dom_in = local_feats
                offload_logits = offload_mechanism(dom_in)
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")
            
            # ★ Regression mode - compare predicted bk with b_star
            if regression_mode:
                optimized_decisions = (offload_logits.squeeze(1) >= b_star).float()
            else:
                optimized_decisions = (torch.sigmoid(offload_logits).squeeze(1) > 0.5).float()
            
            # Predictions
            local_preds = local_out.argmax(dim=1)
            cloud_preds = cloud_out.argmax(dim=1)
            
            optimized_final_preds = torch.where(optimized_decisions == 0, local_preds, cloud_preds)
            optimized_correct_mask = (optimized_final_preds == labels)
            
            all_bks.append(bk.cpu())
            all_optimized_decisions.append(optimized_decisions.cpu())
            all_optimized_correct.append(optimized_correct_mask.cpu())
    
    # Concatenate all batches
    all_bks = torch.cat(all_bks)
    all_optimized_decisions = torch.cat(all_optimized_decisions)
    all_optimized_correct = torch.cat(all_optimized_correct)
    total_samples = len(all_bks)
    
    # Analyze for each threshold
    results = {
        'tau_values': [],
        'borderline_misclass_rates': [],
        'normal_misclass_rates': [],
        'borderline_counts': [],
        'normal_counts': []
    }
    
    print(f"\n{'='*80}")
    print(f"RESULTS BY THRESHOLD")
    print(f"{'='*80}")
    print(f"{'Tau':>6} | {'Border Count':>12} | {'Border %':>8} | {'Border Misclass':>15} | {'Normal Misclass':>15}")
    print(f"{'-'*80}")
    
    for tau in tau_values:
        borderline_mask = (all_bks.abs() < tau)
        normal_mask = ~borderline_mask
        
        borderline_total = borderline_mask.sum().item()
        normal_total = normal_mask.sum().item()
        
        borderline_wrong = (~all_optimized_correct[borderline_mask]).sum().item()
        normal_wrong = (~all_optimized_correct[normal_mask]).sum().item()
        
        borderline_misclass = 100 * borderline_wrong / borderline_total if borderline_total > 0 else 0.0
        normal_misclass = 100 * normal_wrong / normal_total if normal_total > 0 else 0.0
        
        borderline_pct = 100 * borderline_total / total_samples
        
        results['tau_values'].append(tau)
        results['borderline_misclass_rates'].append(borderline_misclass)
        results['normal_misclass_rates'].append(normal_misclass)
        results['borderline_counts'].append(borderline_total)
        results['normal_counts'].append(normal_total)
        
        print(f"{tau:6.2f} | {borderline_total:12d} | {borderline_pct:7.1f}% | {borderline_misclass:14.1f}% | {normal_misclass:14.1f}%")
    
    # ================================================================
    # PLOTTING
    # ================================================================
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PLOT 1: Misclassification rates vs threshold
        ax1.plot(results['tau_values'], results['borderline_misclass_rates'], 
                marker='o', linewidth=2.5, markersize=8, color='#FF6B6B', 
                label='Borderline Samples', linestyle='-')
        ax1.plot(results['tau_values'], results['normal_misclass_rates'], 
                marker='s', linewidth=2.5, markersize=8, color='#4CAF50', 
                label='Normal Samples', linestyle='--')
        
        ax1.set_xlabel('Threshold τ (for |bk| < τ)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Misclassification Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Impact of Borderline Threshold on Misclassification', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11, loc='best')
        
        # Add value labels
        for tau, border_misc in zip(results['tau_values'], results['borderline_misclass_rates']):
            ax1.text(tau, border_misc + 1, f'{border_misc:.1f}%', 
                    ha='center', fontsize=9, color='#FF6B6B')
        
        # PLOT 2: Sample counts vs threshold
        ax2.plot(results['tau_values'], results['borderline_counts'], 
                marker='o', linewidth=2.5, markersize=8, color='#FF6B6B', 
                label='Borderline Count', linestyle='-')
        
        ax2_pct = ax2.twinx()
        borderline_pcts = [100 * count / total_samples for count in results['borderline_counts']]
        ax2_pct.plot(results['tau_values'], borderline_pcts, 
                    marker='D', linewidth=2, markersize=6, color='#9B59B6', 
                    label='Borderline %', linestyle=':')
        
        ax2.set_xlabel('Threshold τ (for |bk| < τ)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Borderline Sample Count', fontsize=12, fontweight='bold', color='#FF6B6B')
        ax2_pct.set_ylabel('Borderline Percentage (%)', fontsize=12, fontweight='bold', color='#9B59B6')
        ax2.set_title('Borderline Sample Distribution vs Threshold', 
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='#FF6B6B')
        ax2_pct.tick_params(axis='y', labelcolor='#9B59B6')
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_pct.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='best')
        
        plt.suptitle(
            f'Borderline Threshold Sensitivity Analysis (L0={L0:.2f}, {dataset_name.upper()})', 
            fontsize=15, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        output_path = f'borderline_threshold_sensitivity_{dataset_name}_L0{int(L0*100)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot: {output_path}")
        plt.close(fig)
    
    return results


def test_DDNN_with_random(
    local_feature_extractor: nn.Module,
    local_classifier: nn.Module,
    cloud_cnn: nn.Module,
    test_loader: DataLoader,
    L0: float,  # Probability of staying local
    device: str = 'cuda',
    seed: int = None
) -> Tuple[float, float]:
    """
    Test the DDNN with random offloading decisions based on Bernoulli(L0).
    
    For each sample, flip a coin with probability L0 to decide:
      - p=L0   → process locally  (0)
      - p=1-L0 → offload to cloud (1)
    
    This serves as a baseline to show that learned mechanisms outperform
    random chance.
    
    Parameters
    ----------
    local_feature_extractor : nn.Module
        Local feature extraction network
    local_classifier : nn.Module
        Local classification head
    cloud_cnn : nn.Module
        Cloud classification network
    test_loader : DataLoader
        Test dataset loader
    L0 : float
        Probability of processing locally (0.0 to 1.0)
    device : str
        Compute device ('cuda' or 'cpu')
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    overall_acc : float
        Overall classification accuracy (%)
    local_percentage : float
        Actual percentage of samples processed locally
    """
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    total_correct = 0
    total_samples = 0
    local_count = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            
            # Extract local features (always needed)
            local_feats = local_feature_extractor(images)
            
            # Random decision for each sample: Bernoulli(p=L0)
            # torch.rand returns values in [0, 1)
            # if rand < L0 → local (True), else cloud (False)
            random_decisions = torch.rand(batch_size, device=device) < L0
            
            # Get local and cloud outputs
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            # Select predictions based on random decision
            local_preds = local_out.argmax(dim=1)
            cloud_preds = cloud_out.argmax(dim=1)
            
            # Final prediction: local if random_decision==True, else cloud
            final_preds = torch.where(random_decisions, local_preds, cloud_preds)
            
            # Count correct predictions
            total_correct += (final_preds == labels).sum().item()
            total_samples += batch_size
            local_count += random_decisions.sum().item()
    
    overall_acc = 100.0 * total_correct / total_samples
    local_pct = 100.0 * local_count / total_samples
    
    print(f"[Random L0={L0:.2f}] Overall Accuracy: {overall_acc:.2f}%, "
          f"Local%: {local_pct:.2f}% (expected {L0*100:.1f}%)")
    
    return overall_acc, local_pct

def testing_offload_mechanism(
    L0_values: List[float],
    local_feature_extractor: nn.Module,
    local_classifier: nn.Module,
    cloud_cnn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    val_loader: DataLoader,
    *,
    methods_to_test: List[str] = ['feat', 'logits', 'entropy', 'oracle'],
    device: str = 'cuda',
    offload_epochs: int = 70,
    batch_size: int = 128,
    dataset_name: str = 'cifar10',
    evaluation_target: str = 'ddnn_overall',
    num_classes: int = 10,
    cloud_predictor: nn.Module = None,
    feat_latent_dim: int = 32
):
    """
    Universal testing function for offload mechanism evaluation.
    
    Parameters
    ----------
    evaluation_target : str
        • 'ddnn_overall' → measure DDNN classification accuracy
        • 'offload_validation' → measure offload decision accuracy on validation set
    
    Returns
    -------
    dict
        results[method_name] = {
            'L0_values': [...],
            'local_percentages': [...],
            'ddnn_accuracies': [...],        # only for ddnn_overall
            'offload_train_accs': [...],     # only for offload_validation
            'offload_val_accs': [...]        # only for offload_validation
        }
    """
    
    INPUT_MODES = {'feat', 'shallow_feat', 'img', 'logits', 'logits_plus', 
                    'hybrid', 'logits_with_bk_pred', 'logits_with_real_bk', 'logits_predicted_regression'}
    BASELINES = {'entropy', 'oracle', 'local_standalone', 'cloud_standalone', 'random'}
    
    offload_modes = [m for m in methods_to_test if m in INPUT_MODES]
    baseline_modes = [m for m in methods_to_test if m in BASELINES]
    standalone_modes = [m for m in baseline_modes if 'standalone' in m]
    
    # Validation: baselines only make sense for ddnn_overall
    if evaluation_target == 'offload_validation' and baseline_modes:
        print(f"Warning: baselines {baseline_modes} are ignored in 'offload_validation' mode")
        baseline_modes = []
        standalone_modes = []
    
    # ⬇️ ΑΛΛΑΓΗ: Storage structure depends on evaluation_target
    results = {}
    for method in methods_to_test:
        if method in INPUT_MODES:
            # Offload mechanisms have full metrics
            if evaluation_target == 'ddnn_overall':
                results[method] = {
                    'L0_values': [],
                    'local_percentages': [],
                    'ddnn_accuracies': []
                }
            else:  # offload_validation
                results[method] = {
                    'L0_values': [],
                    'local_percentages': [],
                    'offload_train_accs': [],
                    'offload_val_accs': []
                }
        else:
            # Baselines & standalone only have ddnn_accuracies (no offload metrics)
            results[method] = {
                'L0_values': [],
                'local_percentages': [],
                'ddnn_accuracies': []
            }
    
    # Precompute features, bks, logits, images ONCE
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    with torch.no_grad():
        all_features, all_bks, all_labels, all_logits, all_images = \
            compute_bks_input_for_deep_offload(
                local_feature_extractor, local_classifier, cloud_cnn,
                train_loader, method=0, device=device
            )
    
    # ================================================================
    # MAIN LOOP - Iterate over L0 values
    # ================================================================
    for L0 in L0_values:
        print(f"\n{'='*80}")
        print(f"Testing L0 = {L0:.2f} ({L0*100:.0f}% target local)")
        print(f"{'='*80}")
        
        b_star = calculate_b_star(all_bks, L0)
        reference_local_pct = None
        
        # ------------------------------------------------------------
        # A) Test each OFFLOAD MECHANISM
        # ------------------------------------------------------------
        for mode in offload_modes:
            print(f"\n--- Training offload mechanism with input_mode='{mode}' ---")
            
            combined_data = create_3d_data_deep(
                all_bks, all_features, all_logits, all_images, all_labels,
                input_mode=mode
            )
            
            # Determine if regression mode for dataset
            is_regression = (mode == 'logits_predicted_regression')
            
            offload_dataset = OffloadDatasetCNN(
                combined_data, b_star,
                input_mode=mode,
                include_bk=False,
                use_oracle_labels=False,
                local_clf=local_classifier,
                cloud_clf=cloud_cnn,
                device=device,
                regression_target=is_regression
            )
            
            offload_loader = DataLoader(offload_dataset, batch_size=batch_size)
            
            offload_model = OffloadMechanism(
                input_mode=mode,
                num_classes=num_classes,
                feat_latent_dim=feat_latent_dim
            ).to(device)
            
            optimizer = torch.optim.Adam(offload_model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
            
            train_deep_offload_mechanism(
                offload_model, val_loader, optimizer, offload_loader,
                local_feature_extractor, local_classifier, cloud_cnn,
                b_star, scheduler,
                input_mode=mode, device=device,
                epochs=offload_epochs, lr=1e-3, stop_threshold=0.9,
                num_classes=num_classes,
                cloud_predictor=cloud_predictor,
                dataset_name=dataset_name
            )
            
            # ⬇️ ΑΛΛΑΓΗ: Evaluate offload accuracies ONLY in offload_validation mode
            if evaluation_target == 'offload_validation':
                off_train = evaluate_offload_decision_accuracy_CNN_train(
                    offload_loader, local_feature_extractor, local_classifier,
                    cloud_cnn, offload_model, b_star, input_mode=mode, device=device,
                    num_classes=num_classes
                )
                off_val = evaluate_offload_decision_accuracy_CNN_test(
                    offload_model, local_feature_extractor, local_classifier,
                    cloud_cnn, test_loader, b_star, input_mode=mode, device=device
                )
                
                # Placeholder for local_pct (could compute from dataset)
                local_pct = (offload_dataset.combined_data[0][1] < b_star) * 100
                
                results[mode]['L0_values'].append(L0)
                results[mode]['local_percentages'].append(local_pct)
                results[mode]['offload_train_accs'].append(off_train)
                results[mode]['offload_val_accs'].append(off_val)
                
                print(f"  {mode}: LocalPct={local_pct:.2f}%, "
                      f"OffTrain={off_train:.2f}%, OffVal={off_val:.2f}%")
            
            elif evaluation_target == 'ddnn_overall':
                local_pct, ddnn_acc = test_DDNN_with_optimized_rule(
                    offload_model, local_feature_extractor, local_classifier,
                    cloud_cnn, test_loader, b_star, input_mode=mode, device=device,
                    cloud_predictor=cloud_predictor
                )
                
                if reference_local_pct is None:
                    reference_local_pct = local_pct
                
                results[mode]['L0_values'].append(L0)
                results[mode]['local_percentages'].append(local_pct)
                results[mode]['ddnn_accuracies'].append(ddnn_acc)
                
                print(f"  {mode}: LocalPct={local_pct:.2f}%, DDNN_Acc={ddnn_acc:.2f}%")
        
        # ------------------------------------------------------------
        # B) Test BASELINES (only for ddnn_overall)
        # ------------------------------------------------------------
        if evaluation_target == 'ddnn_overall':
            if 'entropy' in baseline_modes and reference_local_pct is not None:
                print(f"\n--- Testing entropy baseline ---")
                _, _, ent_acc, _, _, ent_pct, _ = test_DDNN_with_entropy(
                    local_feature_extractor, local_classifier, cloud_cnn,
                    test_loader, target_local_percent=reference_local_pct, device=device
                )
                
                results['entropy']['L0_values'].append(L0)
                results['entropy']['local_percentages'].append(ent_pct)
                results['entropy']['ddnn_accuracies'].append(ent_acc)
                
                print(f"  entropy: LocalPct={ent_pct:.2f}%, DDNN_Acc={ent_acc:.2f}%")
            
            if 'oracle' in baseline_modes:
                print(f"\n--- Testing oracle baseline ---")
                oracle_acc, oracle_pct = test_DDNN_with_oracle(
                    local_feature_extractor, local_classifier, cloud_cnn,
                    test_loader, b_star, L0=L0, device=device
                )
                
                results['oracle']['L0_values'].append(L0)
                results['oracle']['local_percentages'].append(oracle_pct)
                results['oracle']['ddnn_accuracies'].append(oracle_acc)
                
                print(f"  oracle: LocalPct={oracle_pct:.2f}%, DDNN_Acc={oracle_acc:.2f}%")
            
            if 'random' in baseline_modes:
                print(f"\n--- Testing random baseline ---")
                acc, pct = test_DDNN_with_random(
                    local_feature_extractor, local_classifier, cloud_cnn,
                    test_loader, L0=L0, device=device, seed=42
                )
                
                results['random']['L0_values'].append(L0)
                results['random']['local_percentages'].append(pct)
                results['random']['ddnn_accuracies'].append(acc)
    
    # ================================================================
    # C) Test STANDALONE BASELINES (only for ddnn_overall, after L0 loop)
    # ================================================================
    if evaluation_target == 'ddnn_overall' and standalone_modes:
        print(f"\n{'='*80}")
        print("Testing STANDALONE baselines (100% local / 100% cloud)")
        print(f"{'='*80}")
        
        if 'local_standalone' in standalone_modes:
            print(f"\n--- Testing local_standalone baseline ---")
            acc, pct = test_DDNN_with_standalone(
                local_feature_extractor, local_classifier, cloud_cnn,
                test_loader, mode='local_standalone', device=device
            )
            
            results['local_standalone']['L0_values'].append(None)
            results['local_standalone']['local_percentages'].append(pct)
            results['local_standalone']['ddnn_accuracies'].append(acc)
        
        if 'cloud_standalone' in standalone_modes:
            print(f"\n--- Testing cloud_standalone baseline ---")
            acc, pct = test_DDNN_with_standalone(
                local_feature_extractor, local_classifier, cloud_cnn,
                test_loader, mode='cloud_standalone', device=device
            )
            
            results['cloud_standalone']['L0_values'].append(None)
            results['cloud_standalone']['local_percentages'].append(pct)
            results['cloud_standalone']['ddnn_accuracies'].append(acc)
    
    # ================================================================
    # PLOTTING
    # ================================================================
    for method in results:
        if len(results[method]['local_percentages']) > 0 and 'standalone' not in method:
            sort_idx = np.argsort(results[method]['local_percentages'])
            for key in results[method]:
                if len(results[method][key]) > 0:
                    results[method][key] = np.array(results[method][key])[sort_idx]

    MODE_LABELS = {
        'feat': 'Local Features DDNN',
        'shallow_feat': 'Shallow Features DDNN',
        'img': 'Image DDNN',
        'logits': 'Logits DDNN',
        'logits_plus': 'Logits Plus DDNN',
        'local_standalone': 'Local Joined-Trained',
        'cloud_standalone': 'Cloud Joined-Trained',
        'random': 'Random Offload'
    }

    if evaluation_target == 'ddnn_overall':
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for mode in offload_modes:
            label = MODE_LABELS.get(mode, mode)
            ax.plot(results[mode]['local_percentages'], 
                    results[mode]['ddnn_accuracies'], 
                    marker='o', linestyle='-', linewidth=2,
                    label=label)
        
        if 'entropy' in baseline_modes:
            ax.plot(results['entropy']['local_percentages'], 
                    results['entropy']['ddnn_accuracies'], 
                    marker='x', linestyle='-.', linewidth=2,
                    label='Entropy DDNN')
        
        if 'oracle' in baseline_modes:
            ax.plot(results['oracle']['local_percentages'], 
                    results['oracle']['ddnn_accuracies'], 
                    marker='d', linestyle='--', linewidth=2,
                    label='Oracle DDNN')
        
        if 'random' in baseline_modes:
            ax.plot(results['random']['local_percentages'], 
                    results['random']['ddnn_accuracies'], 
                    marker='v', linestyle=':', linewidth=2, color='gray',
                    label='Random Offload')
        
        if 'local_standalone' in standalone_modes:
            ax.plot([100.0], results['local_standalone']['ddnn_accuracies'], 
                    marker='*', markersize=15, linestyle='', color='green',
                    label='Local Only')
        
        if 'cloud_standalone' in standalone_modes:
            ax.plot([0.0], results['cloud_standalone']['ddnn_accuracies'], 
                    marker='*', markersize=15, linestyle='', color='red',
                    label='Cloud Only')
        
        ax.set_xlim(-5, 105)
        ax.set_xlabel('Local Percentage (%)', fontsize=12)
        ax.set_ylabel('DDNN Overall Accuracy (%)', fontsize=12)
        ax.set_title('DDNN Overall Accuracy vs Local Percentage', 
                    fontsize=14, fontweight='bold', pad=15)
        fig.text(0.5, 0.95, f'Dataset: {dataset_name.upper()}', 
                ha='center', va='top', fontsize=11, style='italic', color='gray')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

    elif evaluation_target == 'offload_validation':
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for mode in offload_modes:
            label_train = MODE_LABELS.get(mode, mode).replace(' DDNN', ' Train')
            label_val = MODE_LABELS.get(mode, mode).replace(' DDNN', ' Val')
            
            ax.plot(results[mode]['local_percentages'], 
                results[mode]['offload_train_accs'], 
                marker='o', linestyle='-', linewidth=2,
                label=label_train)
            
            ax.plot(results[mode]['local_percentages'], 
                results[mode]['offload_val_accs'], 
                marker='s', linestyle='--', linewidth=2,
                label=label_val)
        
        ax.set_xlabel('Local Percentage (%)', fontsize=12)
        ax.set_ylabel('Offload Decision Accuracy (%)', fontsize=12)
        ax.set_title('Train-Validation Accuracy for Offload Mechanism', 
                    fontsize=14, fontweight='bold', pad=15)
        fig.text(0.5, 0.95, f'Dataset: {dataset_name.upper()}', 
                ha='center', va='top', fontsize=11, style='italic', color='gray')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    fname = f'{evaluation_target}_{dataset_name}_{"_".join(offload_modes)}.png'
    fig.savefig(os.path.join(plots_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
        
    return results

def test_DDNN_with_standalone(
    local_feature_extractor: nn.Module,
    local_classifier: nn.Module,
    cloud_cnn: nn.Module,
    test_loader: DataLoader,
    mode: str,  # 'local_standalone' or 'cloud_standalone'
    device: str = 'cuda'
) -> Tuple[float, float]:
    """
    Test the DDNN with 100% local or 100% cloud execution (no offloading).
    
    This serves as a baseline to compare against learned offload mechanisms.
    
    Parameters
    ----------
    local_feature_extractor : nn.Module
        Local feature extraction network
    local_classifier : nn.Module
        Local classification head
    cloud_cnn : nn.Module
        Cloud classification network
    test_loader : DataLoader
        Test dataset loader
    mode : str
        'local_standalone' → all samples processed locally (L0=100%)
        'cloud_standalone' → all samples processed in cloud (L0=0%)
    device : str
        Compute device ('cuda' or 'cpu')
    
    Returns
    -------
    overall_acc : float
        Overall classification accuracy (%)
    local_percentage : float
        Percentage of samples processed locally (100.0 or 0.0)
    """
    
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            
            # Extract local features (always needed)
            local_feats = local_feature_extractor(images)
            
            if mode == 'local_standalone':
                # Use ONLY local classifier
                outputs = local_classifier(local_feats)
                local_pct = 100.0
            elif mode == 'cloud_standalone':
                # Use ONLY cloud classifier
                outputs = cloud_cnn(local_feats)
                local_pct = 0.0
            else:
                raise ValueError("mode must be 'local_standalone' or 'cloud_standalone'")
            
            # Get predictions
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size
    
    overall_acc = 100.0 * total_correct / total_samples
    
    print(f"[{mode}] Overall Accuracy: {overall_acc:.2f}%, Local%: {local_pct:.1f}%")
    
    return overall_acc, local_pct


def test_offload_overfitting(
    local_feature_extractor: nn.Module,
    local_classifier: nn.Module,
    cloud_cnn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    val_loader: DataLoader,
    *,
    L0: float = 0.54,
    methods_to_test: List[str] = ['feat', 'logits'],
    device: str = 'cuda',
    offload_epochs: int = 70,
    batch_size: int = 128,
    dataset_name: str = 'cifar10',
    num_classes: int = 10
) -> Dict:
    """
    Test overfitting behavior of offload mechanisms by plotting train vs validation
    accuracy across training epochs for a SINGLE L0 value.
    
    Parameters
    ----------
    L0 : float
        Single target local percentage (default 0.54 → 54% local)
    methods_to_test : List[str]
        Offload mechanisms to compare (e.g., ['feat', 'logits', 'logits_plus'])
    
    Returns
    -------
    dict
        results[method] = {
            'epochs': [1, 2, ..., N],
            'train_accs': [...],
            'val_accs': [...]
        }
    """
    
    INPUT_MODES = {'feat', 'shallow_feat', 'img', 'logits', 'logits_plus'}
    offload_modes = [m for m in methods_to_test if m in INPUT_MODES]
    
    if not offload_modes:
        raise ValueError("At least one offload mechanism must be tested")
    
    # Precompute features once
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    with torch.no_grad():
        all_features, all_bks, all_labels, all_logits, all_images = \
            compute_bks_input_for_deep_offload(
                local_feature_extractor, local_classifier, cloud_cnn,
                train_loader, method=0, device=device
            )
    
    b_star = calculate_b_star(all_bks, L0)
    
    print(f"\n{'='*80}")
    print(f"Testing Overfitting at L0 = {L0:.2f} ({L0*100:.0f}% target local)")
    print(f"b* = {b_star:.4f}")
    print(f"{'='*80}")
    
    results = {}
    
    # Test each offload mechanism
    for mode in offload_modes:
        print(f"\n--- Training {mode} offload mechanism ---")
        
        # Build dataset
        combined_data = create_3d_data_deep(
            all_bks, all_features, all_logits, all_images, all_labels,
            input_mode=mode
        )
        
        offload_dataset = OffloadDatasetCNN(
            combined_data, b_star,
            input_mode=mode,
            include_bk=False,
            use_oracle_labels=False,
            local_clf=local_classifier,
            cloud_clf=cloud_cnn,
            device=device
        )
        
        offload_loader = DataLoader(offload_dataset, batch_size=batch_size)
        
        # Instantiate model
        offload_model = OffloadMechanism(
            input_mode=mode,
            num_classes=num_classes
        ).to(device)
        
        optimizer = torch.optim.Adam(offload_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
        
        # Train with history tracking
        history = train_deep_offload_mechanism(
            offload_model, val_loader, optimizer, offload_loader,
            local_feature_extractor, local_classifier, cloud_cnn,
            b_star, scheduler,
            input_mode=mode, device=device,
            epochs=offload_epochs, lr=1e-3, stop_threshold=0.9,
            return_history=True,  # ⬅️ Enable history tracking
            num_classes=num_classes,
            dataset_name=dataset_name
        )
        
        # Store results
        results[mode] = {
            'epochs': list(range(1, len(history['val_accs']) + 1)),
            'train_accs': history['train_accs'],
            'val_accs': history['val_accs']
        }
        
        print(f"  {mode}: Final Val Acc = {history['val_accs'][-1]:.2f}%")
    
    # ================================================================
    # PLOTTING: Train vs Val Accuracy per Epoch
    # ================================================================
    MODE_LABELS = {
        'feat': 'Local Features',
        'shallow_feat': 'Shallow Features',
        'img': 'Raw Image',
        'logits': 'Logits',
        'logits_plus': 'Logits Plus'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for mode in offload_modes:
        label = MODE_LABELS.get(mode, mode)
        epochs = results[mode]['epochs']
        
        # Plot train accuracy (solid line)
        ax.plot(epochs, results[mode]['train_accs'], 
                linestyle='-', linewidth=2, marker='o', markersize=4,
                label=f'{label} (Train)')
        
        # Plot val accuracy (dashed line)
        ax.plot(epochs, results[mode]['val_accs'], 
                linestyle='--', linewidth=2, marker='s', markersize=4,
                label=f'{label} (Val)')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Offload Decision Accuracy (%)', fontsize=12)
    ax.set_title(f'Overfitting Analysis (L0={L0:.2f}, {dataset_name.upper()})',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, ncol=2)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    fname = f'overfitting_{dataset_name}_L0{int(L0*100)}.png'
    fig.savefig(os.path.join(plots_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n{'='*80}")
    print(f"Overfitting test completed!")
    print(f"Plot saved: {os.path.join(plots_dir, fname)}")
    print(f"{'='*80}")
    
    return results

def test_DDNN_with_optimized_rule(
    offload_mechanism,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    test_loader,
    b_star,
    threshold=0.5,
    input_mode= 'feat',
    device='cuda',
    cloud_predictor=None,
    test_with_real_cloud_logits=False
):
    """
    Evaluates the offload decisions on test data by comparing:
      1) "Real" label from oracle labeling (uses correctness of local/cloud and bk threshold for ties)
      2) "Predicted" label from the offload mechanism (if offload_prob > threshold => 1, else 0)

    Additionally, we now *use* the offload decision to classify each sample either locally (label=0)
    or via the cloud (label=1), measure the classification accuracy, and print:

      - Overall accuracy (percentage of samples correctly classified overall)
      - Local accuracy (percentage of local samples that are correct)
      - Cloud accuracy (percentage of cloud samples that are correct)
      - Percentage of samples that ended up local

    Returns:
        local_percentage, overall_acc
    """

    # Put models in eval mode
    offload_mechanism.eval()
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    # Variables for matching "real" label vs "predicted" label of offload (decision correctness)
    correct_decisions = 0
    total_samples = 0

    # Variables for measuring classification accuracy based on offload decision
    local_correct = 0
    local_total = 0
    cloud_correct = 0
    cloud_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total_samples += batch_size

            # 1) Extract local features
            local_feats = local_feature_extractor(images)

            # 2) Compute local and cloud logits
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            # 3) Oracle labeling (0=local, 1=cloud)
            oracle_labels = my_oracle_decision_function(local_out, cloud_out, labels, b_star=b_star).float()

            # 4) Offload mechanism prediction
            if input_mode == 'logits':
                dom_in = local_out
                dom_logits = offload_mechanism(dom_in)
            elif input_mode == 'img':
                dom_in = images
                dom_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                top2 = torch.topk(probs, 2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(probs.size(1))
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                dom_logits = offload_mechanism(dom_in)
            elif input_mode == 'hybrid':
                probs = F.softmax(local_out, dim=1)
                top2 = torch.topk(probs, 2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(probs.size(1))
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
                dom_logits = offload_mechanism(dom_in, feat=local_feats)
            elif input_mode == 'logits_with_bk_pred':
                # Use CloudLogitPredictor to predict cloud logits
                dom_in = compute_logits_with_bk_pred(local_out, local_feats, cloud_predictor, local_out.size(1))
                dom_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_with_real_bk':
                # Use actual cloud logits for upper bound testing
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(batch_size, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                logits_plus = torch.cat([local_out, margin, entropy], dim=1)
                local_probs_bk = F.softmax(local_out, dim=1)
                cloud_probs_bk = F.softmax(cloud_out, dim=1)
                real_bk = cloud_probs_bk - local_probs_bk
                dom_in = torch.cat([logits_plus, real_bk], dim=1)
                dom_logits = offload_mechanism(dom_in)
            elif input_mode == 'logits_predicted_regression':
                # Regression approach - predicts bk directly
                if test_with_real_cloud_logits:
                    # TESTING: Use real cloud logits
                    combined_input = torch.cat([local_out, cloud_out], dim=1)  # (B, 20)
                else:
                    # PRODUCTION: Use predicted cloud logits
                    combined_input = compute_logits_predicted_regression(local_out, local_feats, cloud_predictor)
                dom_logits = offload_mechanism(combined_input)
            elif input_mode in ('feat', 'shallow_feat'):
                dom_in = local_feats
                dom_logits = offload_mechanism(dom_in)
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")
            dom_probs = torch.sigmoid(dom_logits).squeeze(1)
            predicted_label = (dom_probs > threshold).float()

            # 5) Compare oracle_labels vs predicted_label
            correct_decisions += (oracle_labels == predicted_label).sum().item()

            # 6) Use predicted_label to classify each sample locally (0) or via cloud (1)
            local_mask = (predicted_label == 0)
            cloud_mask = (predicted_label == 1)

            # Classify the local samples
            if local_mask.any():
                local_outputs = local_classifier(local_feats[local_mask])
                local_preds = local_outputs.argmax(dim=1)
                local_correct += (local_preds == labels[local_mask]).sum().item()
                local_total += local_mask.sum().item()

            # Classify the cloud samples
            if cloud_mask.any():
                cloud_outputs = cloud_cnn(local_feats[cloud_mask])
                cloud_preds = cloud_outputs.argmax(dim=1)
                cloud_correct += (cloud_preds == labels[cloud_mask]).sum().item()
                cloud_total += cloud_mask.sum().item()

    offload_accuracy = (correct_decisions / total_samples) * 100.0
    local_acc = (local_correct / local_total * 100.0) if local_total > 0 else 0.0
    cloud_acc = (cloud_correct / cloud_total * 100.0) if cloud_total > 0 else 0.0
    overall_acc = (local_correct + cloud_correct) / total_samples * 100.0
    local_percentage = (local_total / total_samples * 100.0)

    print(f"[evaluate_offload_decision_accuracy_CNN_test] Decision Accuracy: {offload_accuracy:.2f}% "
          f"(Threshold={threshold}, b_star={b_star:.3f})")
    print(f"[Classification Results] Overall Accuracy: {overall_acc:.2f}%")
    print(f"[Classification Results] Local Accuracy: {local_acc:.2f}%, Local samples: {local_total}")
    print(f"[Classification Results] Cloud Accuracy: {cloud_acc:.2f}%, Cloud samples: {cloud_total}")
    print(f"[Classification Results] Percentage Local: {local_percentage:.2f}%")

    return local_percentage, overall_acc

def test_offload_finetuning(
    L0: float,
    local_feature_extractor: nn.Module,
    local_classifier: nn.Module,
    cloud_cnn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    val_loader: DataLoader,
    *,
    variable_to_test: str = 'dropout',
    test_values: List = None,  # user-provided grid
    device: str = 'cuda',
    offload_epochs: int = 70,
    batch_size: int = 128,
    dataset_name: str = 'cifar10',
    input_mode: str = 'logits',
    num_classes: int = 10
):
    """
    Fine-tuning test for offload mechanism (logits mode).

    Notes:
      - If variable_to_test is 'dropout' or 'layers' we rebuild+train a fresh offload
        model for every tested value.
      - If variable_to_test is 'threshold' we TRAIN ONCE with default architecture,
        then only vary the decision threshold (no retraining per threshold).
    """
    # -------------------- define defaults / user override --------------------
    if variable_to_test == 'dropout':
        default_values = [0.0, 0.1, 0.3, 0.5]
        test_values = test_values if test_values is not None else default_values
        default_layers = [256, 128, 64, 32, 1]
        default_threshold = 0.5
        x_label = 'Dropout Probability'

    elif variable_to_test == 'layers':
        default_values = ['shallow', 'default', 'deep']
        test_values = test_values if test_values is not None else default_values
        layer_configs = {
            'shallow': [64, 32, 1],           # 3 layers
            'default': [256, 128, 64, 32, 1], # 5 layers
            'deep': [1024, 512, 256, 128, 64, 32, 16, 1]  # 8 layers
        }
        default_dropout = 0.1
        default_threshold = 0.5
        x_label = 'Architecture (# FC Layers)'

    elif variable_to_test == 'threshold':
        default_values = [0.3, 0.5, 0.7, 'calibrated']
        test_values = test_values if test_values is not None else default_values
        default_dropout = 0.1
        default_layers = [256, 128, 64, 32, 1]
        x_label = 'Decision Threshold'

    else:
        raise ValueError("variable_to_test must be 'dropout', 'layers', or 'threshold'")

    # -------------------- precompute features & bks once --------------------
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    with torch.no_grad():
        all_features, all_bks, all_labels, all_logits, all_images = \
            compute_bks_input_for_deep_offload(
                local_feature_extractor, local_classifier, cloud_cnn,
                train_loader, method=0, device=device
            )

    b_star = calculate_b_star(all_bks, L0)

    # build offload dataset (logits mode by default)
    combined_data = create_3d_data_deep(
        all_bks, all_features, all_logits, all_images, all_labels,
        input_mode=input_mode
    )
    offload_dataset = OffloadDatasetCNN(
        combined_data, b_star,
        input_mode=input_mode,
        include_bk=False,
        use_oracle_labels=False,
        local_clf=local_classifier,
        cloud_clf=cloud_cnn,
        device=device
    )
    offload_loader = DataLoader(offload_dataset, batch_size=batch_size)

    # results container
    results = {
        'tested_values': [],
        'ddnn_accuracies': [],
        'local_percentages': []
    }

    # ---------------------------------------------------------------------
    # If testing thresholds => train once, then evaluate multiple thresholds
    # ---------------------------------------------------------------------
    if variable_to_test == 'threshold':
        print("Training single offload model (threshold sweep) with default architecture...")
        fc_dims = default_layers
        dropout = default_dropout

        input_dim = num_classes if input_mode == 'logits' else (all_logits[0].shape[-1] + 2 if input_mode == 'logits_plus' else tuple(offload_dataset[0][0].shape))
        offload_model = OffloadMechanism(
            input_shape=(num_classes,) if input_mode == 'logits' else None,
            input_mode=input_mode,
            fc_dims=fc_dims,
            dropout_p=dropout,
            latent_in=(1,),
            num_classes=num_classes
        ).to(device)

        optimizer = torch.optim.Adam(offload_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)

        train_deep_offload_mechanism(
            offload_model, val_loader, optimizer, offload_loader,
            local_feature_extractor, local_classifier, cloud_cnn,
            b_star, scheduler,
            input_mode=input_mode, device=device,
            epochs=offload_epochs, lr=1e-3, stop_threshold=0.9,
            num_classes=num_classes,
            dataset_name=dataset_name
        )

        # now loop thresholds (no retrain)
        for test_val in test_values:
            print(f"\n--- Evaluating threshold = {test_val} ---")
            if test_val == 'calibrated':
                tau, _ = calibrate_threshold(offload_model, offload_loader, device=device)
                tau_used = tau
                print(f"  calibrated τ = {tau_used:.4f}")
            else:
                tau_used = float(test_val)
                print(f"  fixed τ = {tau_used:.4f}")

            local_pct, ddnn_acc = test_DDNN_with_optimized_rule(
                offload_model, local_feature_extractor, local_classifier,
                cloud_cnn, test_loader, b_star, threshold=tau_used, input_mode=input_mode, device=device
            )

            results['tested_values'].append(test_val)
            results['ddnn_accuracies'].append(ddnn_acc)
            results['local_percentages'].append(local_pct)
            print(f"  -> Acc={ddnn_acc:.2f}%, Local%={local_pct:.2f}%")

        # end threshold branch
        # plotting handled below
    else:
        # -----------------------------------------------------------------
        # For 'dropout' and 'layers' we need to train for every tested value
        # -----------------------------------------------------------------
        for test_val in test_values:
            print(f"\n{'='*60}\nTesting {variable_to_test} = {test_val}\n{'='*60}")
            if variable_to_test == 'dropout':
                fc_dims = default_layers
                dropout = float(test_val)
                threshold = default_threshold
            else:  # layers
                fc_dims = layer_configs[test_val]
                dropout = default_dropout
                threshold = default_threshold

            # instantiate offload model for this config
            offload_model = OffloadMechanism(
                input_shape=(num_classes,) if input_mode == 'logits' else None,
                input_mode=input_mode,
                fc_dims=fc_dims,
                dropout_p=dropout,
                latent_in=(1,),
                num_classes=num_classes
            ).to(device)

            optimizer = torch.optim.Adam(offload_model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)

            # train for this configuration
            train_deep_offload_mechanism(
                offload_model, val_loader, optimizer, offload_loader,
                local_feature_extractor, local_classifier, cloud_cnn,
                b_star, scheduler,
                input_mode=input_mode, device=device,
                epochs=offload_epochs, lr=1e-3, stop_threshold=0.9,
                num_classes=num_classes,
                dataset_name=dataset_name
            )

            # determine threshold (default fixed)
            if threshold == 'calibrated':
                tau, _ = calibrate_threshold(offload_model, offload_loader, device=device)
                tau_used = tau
            else:
                tau_used = threshold

            local_pct, ddnn_acc = test_DDNN_with_optimized_rule(
                offload_model, local_feature_extractor, local_classifier,
                cloud_cnn, test_loader, b_star, threshold=tau_used, input_mode=input_mode, device=device
            )

            results['tested_values'].append(test_val)
            results['ddnn_accuracies'].append(ddnn_acc)
            results['local_percentages'].append(local_pct)
            print(f"  -> Acc={ddnn_acc:.2f}%, Local%={local_pct:.2f}%")

    # -------------------- plotting --------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    x_values = []
    x_labels_plot = []
    for i, val in enumerate(results['tested_values']):
        if variable_to_test == 'layers':
            # ⬇️ ΑΛΛΑΓΗ: Χρήση index για layers
            x_values.append(i)
            layer_name = f"{len(layer_configs[val])} layers"
            x_labels_plot.append(layer_name)
        elif val == 'calibrated':
            x_values.append(i)
            x_labels_plot.append('Calibrated')
        else:
            # ⬇️ Για dropout/threshold χρησιμοποιούμε την πραγματική τιμή
            x_values.append(val)
            x_labels_plot.append(str(val))

    ax.plot(x_values, results['ddnn_accuracies'],
            marker='o', linestyle='-', linewidth=2, markersize=8,
            color='#2E86AB', label='DDNN Overall Accuracy')
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels_plot, rotation=0)
    


    ax2 = ax.twiny()  # Twin x-axis (shares y-axis)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x_values)
    ax2.set_xticklabels([f"{pct:.1f}%" for pct in results['local_percentages']], 
                        rotation=0, color='#A23B72', fontsize=9)
    ax2.set_xlabel('Local Percentage (%)', fontsize=11, color='#A23B72', labelpad=8)
    ax2.tick_params(axis='x', colors='#A23B72')

    # Primary x-axis (bottom)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('DDNN Overall Accuracy (%)', fontsize=12)
    ax.set_title(f'Fine-tuning: {variable_to_test.capitalize()} (L0={L0:.2f}, {dataset_name.upper()})',
                 fontsize=14, fontweight='bold', pad=25)  # ⬅️ Extra padding για τον πάνω άξονα
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=10)

    # ⬇️ ΑΛΛΑΓΗ: Ενημερωμένο infobox με range
    min_local = min(results['local_percentages'])
    max_local = max(results['local_percentages'])
    avg_local_pct = np.mean(results['local_percentages'])
    
    infobox_text = f"Local% range: [{min_local:.1f}, {max_local:.1f}]\nAvg: {avg_local_pct:.1f}%"
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.6, edgecolor='navy')
    ax.text(0.02, 0.98, infobox_text, 
            transform=ax.transAxes,
            fontsize=10, 
            verticalalignment='top',
            horizontalalignment='left',
            bbox=props,
            fontweight='bold')
    
    plt.tight_layout()
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    fname = f'finetuning_{variable_to_test}_{dataset_name}_L0{int(L0*100)}.png'
    fig.savefig(os.path.join(plots_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nFine-tuning test completed for {variable_to_test}. Plot saved: {os.path.join(plots_dir, fname)}")
    return results

def test_inference_timing(
    L0: float,
    local_feature_extractor: nn.Module,
    local_classifier: nn.Module,
    cloud_cnn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    val_loader: DataLoader,
    *,
    methods_to_test: List[str] = ['feat', 'logits', 'entropy', 'oracle', 'random'],
    device: str = 'cuda',
    offload_epochs: int = 50,
    batch_size: int = 128,
    dataset_name: str = 'cifar10',
    input_mode: str = 'logits',
    num_runs: int = 3,  # Αριθμός επαναλήψεων για σταθερό timing
    num_classes: int = 10
) -> Dict:
    """
    Measure inference time for different offload decision methods.
    
    For each method:
      - Train offload mechanism (if applicable)
      - Measure total inference time on test_loader
      - Compute average time per sample
    
    Parameters
    ----------
    L0 : float
        Target local percentage (e.g., 0.54 → 54% local)
    methods_to_test : List[str]
        Methods to benchmark: 'feat', 'logits', 'logits_plus', 'entropy', 'oracle', 'random'
    num_runs : int
        Number of timing runs for averaging (default: 3)
    
    Returns
    -------
    dict
        results[method] = {
            'total_time': float (seconds),
            'per_sample_time': float (milliseconds),
            'accuracy': float (%)
        }
    """
    
    INPUT_MODES = {'feat', 'shallow_feat', 'img', 'logits', 'logits_plus'}
    BASELINES = {'entropy', 'oracle', 'random'}
    
    offload_modes = [m for m in methods_to_test if m in INPUT_MODES]
    baseline_modes = [m for m in methods_to_test if m in BASELINES]
    
    # Precompute features once
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    with torch.no_grad():
        all_features, all_bks, all_labels, all_logits, all_images = \
            compute_bks_input_for_deep_offload(
                local_feature_extractor, local_classifier, cloud_cnn,
                train_loader, method=0, device=device
            )
    
    b_star = calculate_b_star(all_bks, L0)
    N_test = len(test_loader.dataset)
    
    print(f"\n{'='*80}")
    print(f"Timing Benchmark at L0 = {L0:.2f} ({L0*100:.0f}% target local)")
    print(f"Dataset: {dataset_name.upper()}, Test samples: {N_test}")
    print(f"Number of timing runs: {num_runs}")
    print(f"{'='*80}")
    
    results = {}
    
    # ================================================================
    # A) Benchmark OFFLOAD MECHANISMS
    # ================================================================
    for mode in offload_modes:
        print(f"\n--- Benchmarking {mode} offload mechanism ---")
        
        # Build dataset
        combined_data = create_3d_data_deep(
            all_bks, all_features, all_logits, all_images, all_labels,
            input_mode=mode
        )
        
        offload_dataset = OffloadDatasetCNN(
            combined_data, b_star,
            input_mode=mode,
            include_bk=False,
            use_oracle_labels=False,
            local_clf=local_classifier,
            cloud_clf=cloud_cnn,
            device=device
        )
        
        offload_loader = DataLoader(offload_dataset, batch_size=batch_size)
        
        # Train offload model
        offload_model = OffloadMechanism(
            input_mode=mode,
            num_classes=num_classes
        ).to(device)
        
        optimizer = torch.optim.Adam(offload_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
        
        train_deep_offload_mechanism(
            offload_model, val_loader, optimizer, offload_loader,
            local_feature_extractor, local_classifier, cloud_cnn,
            b_star, scheduler,
            input_mode=mode, device=device,
            epochs=offload_epochs, lr=1e-3, stop_threshold=0.9,
            num_classes=num_classes,
            dataset_name=dataset_name
        )
        
        # ⬇️ TIMING: Run multiple times and average
        timings = []
        for run in range(num_runs):
            torch.cuda.synchronize()  # Ensure GPU is idle
            start = time.perf_counter()
            
            local_pct, ddnn_acc = test_DDNN_with_optimized_rule(
                offload_model, local_feature_extractor, local_classifier,
                cloud_cnn, test_loader, b_star, input_mode=mode, device=device
            )
            
            torch.cuda.synchronize()  # Wait for GPU to finish
            end = time.perf_counter()
            timings.append(end - start)
        
        avg_time = np.mean(timings)
        std_time = np.std(timings)
        per_sample_ms = (avg_time / N_test) * 1000
        
        results[mode] = {
            'total_time': avg_time,
            'per_sample_time': per_sample_ms,
            'accuracy': ddnn_acc,
            'local_pct': local_pct,
            'std_time': std_time
        }
        
        print(f"  {mode}: {avg_time:.3f}s ± {std_time:.3f}s | "
              f"{per_sample_ms:.3f}ms/sample | Acc={ddnn_acc:.2f}%")
    
    # ================================================================
    # B) Benchmark BASELINES
    # ================================================================
    if 'entropy' in baseline_modes:
        print(f"\n--- Benchmarking Entropy baseline ---")
        
        # Get reference local% from first offload method
        reference_local_pct = list(results.values())[0]['local_pct'] if results else 50.0
        
        timings = []
        for run in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _, _, ent_acc, _, _, ent_pct, _ = test_DDNN_with_entropy(
                local_feature_extractor, local_classifier, cloud_cnn,
                test_loader, target_local_percent=reference_local_pct, device=device
            )
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)
        
        avg_time = np.mean(timings)
        std_time = np.std(timings)
        per_sample_ms = (avg_time / N_test) * 1000
        
        results['entropy'] = {
            'total_time': avg_time,
            'per_sample_time': per_sample_ms,
            'accuracy': ent_acc,
            'local_pct': ent_pct,
            'std_time': std_time
        }
        
        print(f"  Entropy: {avg_time:.3f}s ± {std_time:.3f}s | "
              f"{per_sample_ms:.3f}ms/sample | Acc={ent_acc:.2f}%")
    
    if 'oracle' in baseline_modes:
        print(f"\n--- Benchmarking Oracle baseline ---")
        
        timings = []
        for run in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            oracle_acc, oracle_pct = test_DDNN_with_oracle(
                local_feature_extractor, local_classifier, cloud_cnn,
                test_loader, b_star, L0=L0, device=device
            )
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)
        
        avg_time = np.mean(timings)
        std_time = np.std(timings)
        per_sample_ms = (avg_time / N_test) * 1000
        
        results['oracle'] = {
            'total_time': avg_time,
            'per_sample_time': per_sample_ms,
            'accuracy': oracle_acc,
            'local_pct': oracle_pct,
            'std_time': std_time
        }
        
        print(f"  Oracle: {avg_time:.3f}s ± {std_time:.3f}s | "
              f"{per_sample_ms:.3f}ms/sample | Acc={oracle_acc:.2f}%")
    
    if 'random' in baseline_modes:
        print(f"\n--- Benchmarking Random baseline ---")
        
        timings = []
        for run in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            rand_acc, rand_pct = test_DDNN_with_random(
                local_feature_extractor, local_classifier, cloud_cnn,
                test_loader, L0=L0, device=device, seed=42
            )
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)
        
        avg_time = np.mean(timings)
        std_time = np.std(timings)
        per_sample_ms = (avg_time / N_test) * 1000
        
        results['random'] = {
            'total_time': avg_time,
            'per_sample_time': per_sample_ms,
            'accuracy': rand_acc,
            'local_pct': rand_pct,
            'std_time': std_time
        }
        
        print(f"  Random: {avg_time:.3f}s ± {std_time:.3f}s | "
              f"{per_sample_ms:.3f}ms/sample | Acc={rand_acc:.2f}%")
    
    # ================================================================
    # PLOTTING
    # ================================================================
    plot_timing_results(results, L0, dataset_name, N_test)
    
    return results

def test_DDNN_with_entropy(local_feature_extractor, local_classifier,
                           cloud_cnn, data_loader,
                           target_local_percent,
                           tolerance=0.5,
                           device='cuda'):
    """
    Evaluate the DDNN using an entropy‑based decision rule.
    The function computes *normalized* entropy (range 0‑1) for each local
    prediction, then selects the entropy threshold that yields
    `target_local_percent` (± tolerance) of samples processed locally.

    Args
    ----
    local_feature_extractor, local_classifier, cloud_cnn : torch.nn.Module
    data_loader            : DataLoader with evaluation set
    target_local_percent   : float, desired percentage of local samples
    tolerance              : float, allowed deviation in percentage points
    device                 : 'cuda' or 'cpu'

    Returns
    -------
    local_acc, cloud_acc, overall_acc : float
    local_classified, cloud_classified : int
    """

    # ------------------------------------------------------------------ #
    # 1) Forward pass once – gather per‑sample data
    # ------------------------------------------------------------------ #
    entropies      = []
    local_ok_mask  = []
    cloud_ok_mask  = []

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Local branch
            feats       = local_feature_extractor(images)
            logits_loc  = local_classifier(feats)
            probs_loc   = torch.softmax(logits_loc, dim=1)

            # *** Normalized entropy ***
            C = logits_loc.shape[1]                        # num classes
            entropy_raw = -torch.sum(probs_loc *
                                     torch.log(probs_loc + 1e-9), dim=1)
            entropy_norm = entropy_raw / math.log(C)       # range [0,1]

            # Cloud branch
            logits_cloud = cloud_cnn(feats)

            # Store to CPU
            entropies.append(entropy_norm.cpu())
            local_ok_mask.append(
                (logits_loc.argmax(1).cpu() == labels.cpu()).int())
            cloud_ok_mask.append(
                (logits_cloud.argmax(1).cpu() == labels.cpu()).int())

    entropies   = torch.cat(entropies).numpy()          # shape (N,)
    local_ok    = torch.cat(local_ok_mask).numpy()      # bool/int
    cloud_ok    = torch.cat(cloud_ok_mask).numpy()
    N           = len(entropies)

    # ------------------------------------------------------------------ #
    # 2) Choose threshold to hit target_local_percent
    # ------------------------------------------------------------------ #
    target_k = int(round(target_local_percent / 100 * N))
    target_k = max(0, min(target_k, N - 1))

    order       = entropies.argsort()
    sorted_ent  = entropies[order]

    k = target_k
    threshold = sorted_ent[k]

    def local_pct(th):       # helper
        return 100.0 * (entropies <= th).mean()

    actual_pct = local_pct(threshold)
    if abs(actual_pct - target_local_percent) > tolerance:
        direction = -1 if actual_pct > target_local_percent else 1
        while 0 <= k + direction < N:
            k += direction
            threshold = sorted_ent[k]
            actual_pct = local_pct(threshold)
            if abs(actual_pct - target_local_percent) <= tolerance:
                break

    # ------------------------------------------------------------------ #
    # 3) Final split & accuracy
    # ------------------------------------------------------------------ #
    is_local          = entropies <= threshold
    local_classified  = int(is_local.sum())
    cloud_classified  = N - local_classified

    local_correct     = int(local_ok[is_local].sum())
    cloud_correct     = int(cloud_ok[~is_local].sum())

    local_acc   = 100.0 * local_correct  / local_classified if local_classified else 0.0
    cloud_acc   = 100.0 * cloud_correct  / cloud_classified if cloud_classified else 0.0
    overall_acc = 100.0 * (local_correct + cloud_correct) / N

    # ------------------------------------------------------------------ #
    # 4) Console log (optional)
    # ------------------------------------------------------------------ #
    print(f"[Entropy] Target local%={target_local_percent:.2f} → "
          f"Threshold={threshold:.4f} (normalized) → Local%={actual_pct:.2f}")
    print(f"Samples Local={local_classified} | Cloud={cloud_classified}")
    print(f"Local Acc={local_acc:.2f} | Cloud Acc={cloud_acc:.2f} "
          f"| Overall Acc={overall_acc:.2f}")
    print(f"Mean normalized entropy: {entropies.mean():.4f}")

    return (local_acc, cloud_acc, overall_acc,
            local_classified, cloud_classified, actual_pct,threshold)


def plot_timing_results(results: Dict, L0: float, dataset_name: str, N_test: int):
    """Create beautiful timing visualization similar to data labeling analysis."""
    
    methods = list(results.keys())
    total_times = [results[m]['total_time'] for m in methods]
    per_sample_times = [results[m]['per_sample_time'] for m in methods]
    accuracies = [results[m]['accuracy'] for m in methods]
    std_times = [results[m].get('std_time', 0) for m in methods]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    # ================================================================
    # LEFT PLOT: Total time per method
    # ================================================================
    bars1 = ax1.barh(methods, total_times, color=colors, alpha=0.8, xerr=std_times)
    ax1.set_xlabel('Total Inference Time (seconds)', fontsize=12)
    ax1.set_title(f'Inference Time Comparison (L0={L0:.2f}, {dataset_name.upper()})',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    # Annotate bars with time
    for i, (bar, time_val, std_val) in enumerate(zip(bars1, total_times, std_times)):
        ax1.text(time_val + std_val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{time_val:.3f}s ± {std_val:.3f}s',
                va='center', fontsize=9, fontweight='bold')
    
    # ================================================================
    # RIGHT PLOT: Per-sample time (ms)
    # ================================================================
    bars2 = ax2.barh(methods, per_sample_times, color=colors, alpha=0.8)
    ax2.set_xlabel('Time per Sample (milliseconds)', fontsize=12)
    ax2.set_title(f'Per-Sample Inference Time (N={N_test} samples)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    # Annotate bars
    for i, (bar, time_val, acc) in enumerate(zip(bars2, per_sample_times, accuracies)):
        ax2.text(time_val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{time_val:.3f}ms\n(Acc: {acc:.1f}%)',
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    fname = f'timing_benchmark_{dataset_name}_L0{int(L0*100)}.png'
    fig.savefig(os.path.join(plots_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n{'='*80}")
    print(f"Timing benchmark plot saved: {os.path.join(plots_dir, fname)}")
    print(f"{'='*80}")


# ========================================================================
# COST-BASED vs ORACLE LABELING COMPARISON (Figure 5.5 & 5.6 combined)
# ========================================================================

def compare_labeling_methods(
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    train_loader,
    val_loader,
    test_loader,
    *,
    L0: float = 0.54,
    input_mode: str = 'logits_plus',
    offload_epochs: int = 50,
    batch_size: int = 256,
    device: str = 'cuda',
    dataset_name: str = 'cifar10',
    plot: bool = True,
    num_classes: int = 10
) -> Dict:
    """
    Compare Cost-Based Rule vs Oracle Labeling Rule for offload mechanism training.
    
    This function:
    1. Computes the Noise Rate: % of samples where bk-cost rule disagrees with Oracle
    2. Trains offload mechanism with BOTH labeling methods (2 separate trainings)
    3. Evaluates DDNN Overall Accuracy for each method
    4. Creates comparison plots (Figure 5.5 style)
    
    Parameters
    ----------
    L0 : float
        Target local processing ratio (used to compute b_star threshold)
    input_mode : str
        Input mode for offload mechanism ('logits', 'logits_plus', 'hybrid', etc.)
    offload_epochs : int
        Number of epochs to train each offload mechanism
    
    Returns
    -------
    dict
        Results including noise_rate, accuracies for both methods, etc.
    """
    
    print(f"\n{'='*80}")
    print(f"COST-BASED vs ORACLE LABELING COMPARISON")
    print(f"{'='*80}")
    print(f"Settings: L0={L0:.2f}, input_mode={input_mode}, epochs={offload_epochs}")
    print(f"Dataset: {dataset_name.upper()}")
    
    # ================================================================
    # STEP 1: Compute bk values and b_star from training data
    # ================================================================
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    print(f"\n[Step 1] Computing bk values from training data...")
    
    with torch.no_grad():
        all_features, all_bks, all_labels, all_logits, all_images = \
            compute_bks_input_for_deep_offload(
                local_feature_extractor, local_classifier, cloud_cnn,
                train_loader, method=0, device=device
            )
    
    b_star = calculate_b_star(all_bks, L0)
    print(f"  Computed b_star: {b_star:.4f}")
    
    # ================================================================
    # STEP 2: Compute Noise Rate (disagreement between methods)
    # ================================================================
    print(f"\n[Step 2] Computing Noise Rate (Cost-Based vs Oracle)...")
    
    N_total = 0
    N_agree = 0
    N_disagree = 0
    
    # Also track label distributions
    bk_local_count = 0
    bk_cloud_count = 0
    oracle_local_count = 0
    oracle_cloud_count = 0
    
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            bs = labels.size(0)
            
            # Forward pass
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            # Compute bk values
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)
            local_prob_correct = local_probs[range(bs), labels]
            cloud_prob_correct = cloud_probs[range(bs), labels]
            bk = (1.0 - local_prob_correct) - (1.0 - cloud_prob_correct)
            
            # Cost-Based Rule: bk >= b_star → cloud (1), else local (0)
            bk_decisions = (bk >= b_star).float()
            
            # Oracle Rule: knows ground truth
            oracle_decisions = my_oracle_decision_function(
                local_out, cloud_out, labels, b_star=b_star
            ).float()
            
            # Agreement/Disagreement
            agreement_mask = (bk_decisions == oracle_decisions)
            N_agree += agreement_mask.sum().item()
            N_disagree += (~agreement_mask).sum().item()
            N_total += bs
            
            # Label distributions
            bk_local_count += (bk_decisions == 0).sum().item()
            bk_cloud_count += (bk_decisions == 1).sum().item()
            oracle_local_count += (oracle_decisions == 0).sum().item()
            oracle_cloud_count += (oracle_decisions == 1).sum().item()
    
    noise_rate = 100 * N_disagree / N_total
    agreement_rate = 100 * N_agree / N_total
    
    bk_local_pct = 100 * bk_local_count / N_total
    bk_cloud_pct = 100 * bk_cloud_count / N_total
    oracle_local_pct = 100 * oracle_local_count / N_total
    oracle_cloud_pct = 100 * oracle_cloud_count / N_total
    
    print(f"\n  📊 LABELING COMPARISON:")
    print(f"  {'─'*50}")
    print(f"  Total training samples: {N_total:,}")
    print(f"  Agreement (Cost-Based == Oracle): {N_agree:,} ({agreement_rate:.1f}%)")
    print(f"  Disagreement (Noise Rate):        {N_disagree:,} ({noise_rate:.1f}%)")
    print(f"  {'─'*50}")
    print(f"  Cost-Based Rule distribution: Local={bk_local_pct:.1f}%, Cloud={bk_cloud_pct:.1f}%")
    print(f"  Oracle Rule distribution:     Local={oracle_local_pct:.1f}%, Cloud={oracle_cloud_pct:.1f}%")
    
    # ================================================================
    # STEP 3: Create meta-datasets with both labeling methods
    # ================================================================
    print(f"\n[Step 3] Creating meta-datasets...")
    
    combined_data = create_3d_data_deep(
        all_bks, all_features, all_logits, all_images, all_labels,
        input_mode=input_mode
    )
    
    # Meta-dataset 1: Cost-Based (bk-threshold) labeling
    offload_dataset_bk = OffloadDatasetCNN(
        combined_data, b_star,
        input_mode=input_mode,
        include_bk=False,
        use_oracle_labels=False,  # Use bk-threshold labels
        local_clf=local_classifier,
        cloud_clf=cloud_cnn,
        device=device
    )
    
    # Meta-dataset 2: Oracle labeling
    offload_dataset_oracle = OffloadDatasetCNN(
        combined_data, b_star,
        input_mode=input_mode,
        include_bk=False,
        use_oracle_labels=True,  # Use oracle labels
        local_clf=local_classifier,
        cloud_clf=cloud_cnn,
        device=device
    )
    
    offload_loader_bk = DataLoader(offload_dataset_bk, batch_size=batch_size, shuffle=True)
    offload_loader_oracle = DataLoader(offload_dataset_oracle, batch_size=batch_size, shuffle=True)
    
    print(f"  Created 2 meta-datasets:")
    print(f"    - Cost-Based labeling: {len(offload_dataset_bk)} samples")
    print(f"    - Oracle labeling:     {len(offload_dataset_oracle)} samples")
    
    # ================================================================
    # STEP 4: Train offload mechanism with Cost-Based labels
    # ================================================================
    print(f"\n[Step 4a] Training offload mechanism with COST-BASED labels...")
    
    offload_model_bk = OffloadMechanism(
        input_mode=input_mode,
        num_classes=num_classes
    ).to(device)
    
    optimizer_bk = torch.optim.Adam(offload_model_bk.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_bk = ReduceLROnPlateau(optimizer_bk, mode='max', factor=0.7, patience=5)
    
    train_deep_offload_mechanism(
        offload_model_bk, val_loader, optimizer_bk, offload_loader_bk,
        local_feature_extractor, local_classifier, cloud_cnn,
        b_star, scheduler_bk,
        input_mode=input_mode, device=device,
        epochs=offload_epochs, lr=1e-3, stop_threshold=0.95,
        num_classes=num_classes,
        dataset_name=dataset_name
    )
    
    # ================================================================
    # STEP 5: Train offload mechanism with Oracle labels
    # ================================================================
    print(f"\n[Step 4b] Training offload mechanism with ORACLE labels...")
    
    offload_model_oracle = OffloadMechanism(
        input_mode=input_mode,
        num_classes=num_classes
    ).to(device)
    
    optimizer_oracle = torch.optim.Adam(offload_model_oracle.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_oracle = ReduceLROnPlateau(optimizer_oracle, mode='max', factor=0.7, patience=5)
    
    train_deep_offload_mechanism(
        offload_model_oracle, val_loader, optimizer_oracle, offload_loader_oracle,
        local_feature_extractor, local_classifier, cloud_cnn,
        b_star, scheduler_oracle,
        input_mode=input_mode, device=device,
        epochs=offload_epochs, lr=1e-3, stop_threshold=0.95,
        num_classes=num_classes,
        dataset_name=dataset_name
    )
    
    # ================================================================
    # STEP 6: Evaluate DDNN Overall Accuracy on TEST set
    # ================================================================
    print(f"\n[Step 5] Evaluating DDNN Overall Accuracy on TEST set...")
    
    offload_model_bk.eval()
    offload_model_oracle.eval()
    
    bk_correct = 0
    oracle_correct = 0
    test_total = 0
    
    bk_local_test = 0
    oracle_local_test = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            bs = labels.size(0)
            
            # Forward pass
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)
            
            # Prepare input for offload mechanism
            if input_mode == 'logits':
                dom_in = local_out
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
            elif input_mode == 'hybrid':
                probs = F.softmax(local_out, dim=1)
                num_cls = probs.size(1)
                k = min(2, num_cls)
                top_k = torch.topk(probs, k, dim=1).values
                margin = (top_k[:, 0] - top_k[:, 1]).unsqueeze(1) if k == 2 else torch.zeros(bs, 1, device=device)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(num_cls)
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
            elif input_mode in ('feat', 'shallow_feat'):
                dom_in = local_feats
            else:
                dom_in = local_out  # fallback
            
            # Get decisions from both models
            if input_mode == 'hybrid':
                bk_logits = offload_model_bk(dom_in, feat=local_feats)
                oracle_logits = offload_model_oracle(dom_in, feat=local_feats)
            else:
                bk_logits = offload_model_bk(dom_in)
                oracle_logits = offload_model_oracle(dom_in)
            
            bk_decisions = (torch.sigmoid(bk_logits).squeeze(1) > 0.5).float()
            oracle_decisions_pred = (torch.sigmoid(oracle_logits).squeeze(1) > 0.5).float()
            
            # DDNN predictions
            local_preds = local_out.argmax(dim=1)
            cloud_preds = cloud_out.argmax(dim=1)
            
            # Final predictions
            bk_final = torch.where(bk_decisions == 0, local_preds, cloud_preds)
            oracle_final = torch.where(oracle_decisions_pred == 0, local_preds, cloud_preds)
            
            # Accuracy
            bk_correct += (bk_final == labels).sum().item()
            oracle_correct += (oracle_final == labels).sum().item()
            test_total += bs
            
            # Local percentages
            bk_local_test += (bk_decisions == 0).sum().item()
            oracle_local_test += (oracle_decisions_pred == 0).sum().item()
    
    bk_accuracy = 100 * bk_correct / test_total
    oracle_accuracy = 100 * oracle_correct / test_total
    accuracy_diff = oracle_accuracy - bk_accuracy
    
    bk_local_test_pct = 100 * bk_local_test / test_total
    oracle_local_test_pct = 100 * oracle_local_test / test_total
    
    print(f"\n  📊 DDNN OVERALL ACCURACY (TEST SET):")
    print(f"  {'─'*50}")
    print(f"  Cost-Based Rule training: {bk_accuracy:.2f}% (Local%: {bk_local_test_pct:.1f}%)")
    print(f"  Oracle Labeling training: {oracle_accuracy:.2f}% (Local%: {oracle_local_test_pct:.1f}%)")
    print(f"  {'─'*50}")
    print(f"  Difference (Oracle - Cost): {accuracy_diff:+.2f}%")
    
    # ================================================================
    # PLOTTING
    # ================================================================
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ========== LEFT PLOT: Noise Rate ==========
        labels_left = ['Noise Rate\n(Mislabeled Samples)']
        values_left = [noise_rate]
        colors_left = ['#FF6B6B']
        
        bars1 = ax1.bar(labels_left, values_left, color=colors_left, alpha=0.8, width=0.4)
        ax1.set_ylim(0, max(50, noise_rate + 10))
        ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Label Quality Metrics', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value label
        for bar, val in zip(bars1, values_left):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add explanation (moved lower to avoid overlap)
        ax1.text(0, -15, 
                f'Samples where Cost-Based Rule\ndisagrees with Oracle Rule', 
                ha='center', fontsize=10, style='italic', color='#FF6B6B')
        
        # Add distribution info as text box
        dist_text = (
            f"Label Distribution:\n"
            f"─────────────────────────\n"
            f"Cost-Based: Local {bk_local_pct:.1f}%, Cloud {bk_cloud_pct:.1f}%\n"
            f"Oracle:     Local {oracle_local_pct:.1f}%, Cloud {oracle_cloud_pct:.1f}%"
        )
        ax1.text(0.98, 0.98, dist_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                family='monospace')
        
        # ========== RIGHT PLOT: DDNN Accuracy Comparison ==========
        methods = ['Cost-Based\nRule', 'Oracle\nLabeling']
        accuracies = [bk_accuracy, oracle_accuracy]
        colors_right = ['#3498DB', '#27AE60']
        
        bars2 = ax2.bar(methods, accuracies, color=colors_right, alpha=0.8, width=0.5)
        ax2.set_ylim(min(accuracies) - 5, 100)
        ax2.set_ylabel('DDNN Overall Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Comparison', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars2, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add difference annotation
        if accuracy_diff > 0:
            ax2.annotate('', xy=(1, oracle_accuracy), xytext=(0, bk_accuracy),
                        arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.6))
            ax2.text(0.5, (bk_accuracy + oracle_accuracy)/2,
                    f'+{accuracy_diff:.2f}%', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # Overall title
        plt.suptitle(
            f'Cost-Based vs Oracle Labeling Comparison\n'
            f'(L0={L0:.2f}, {dataset_name.upper()}, Noise Rate={noise_rate:.1f}%)', 
            fontsize=14, fontweight='bold'
        )
        
        # Summary box at bottom
        summary_text = (
            f"Total: {N_total:,} training samples | "
            f"Noise: {noise_rate:.1f}% | "
            f"Test Accuracy: Cost-Based={bk_accuracy:.1f}%, Oracle={oracle_accuracy:.1f}%"
        )
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        
        # Save plot
        plots_dir = 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        output_path = os.path.join(plots_dir, f'labeling_comparison_{dataset_name}_L0{int(L0*100)}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✓ Saved comparison plot: {output_path}")
    
    # ================================================================
    # RETURN RESULTS
    # ================================================================
    return {
        # Noise analysis
        'noise_rate': noise_rate,
        'agreement_rate': agreement_rate,
        'total_samples': N_total,
        # Label distributions
        'bk_local_pct': bk_local_pct,
        'bk_cloud_pct': bk_cloud_pct,
        'oracle_local_pct': oracle_local_pct,
        'oracle_cloud_pct': oracle_cloud_pct,
        # DDNN Accuracy
        'bk_ddnn_accuracy': bk_accuracy,
        'oracle_ddnn_accuracy': oracle_accuracy,
        'accuracy_difference': accuracy_diff,
        # Test local percentages
        'bk_local_test_pct': bk_local_test_pct,
        'oracle_local_test_pct': oracle_local_test_pct,
        # Threshold
        'b_star': b_star
    }




