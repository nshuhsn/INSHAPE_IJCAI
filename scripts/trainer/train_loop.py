import torch
import numpy as np
import math
import copy
import wandb
from utils.util import predictor_loss, union_mask_scatter
import sys
from time import perf_counter

def _parse_batch(batch):
    """Normalize batch to (data, target, sample_ids, padding_mask)."""
    sample_ids = None
    padding_mask = None

    if isinstance(batch, (list, tuple)):
        if len(batch) == 4:
            data, target, sample_ids, padding_mask = batch
        elif len(batch) == 3:
            data, target, third = batch
            if torch.is_tensor(third):
                if third.dim() == 1:
                    sample_ids = third
                else:
                    padding_mask = third
            else:
                sample_ids = third
        else:
            data, target = batch
    else:
        data, target = batch

    return data, target, sample_ids, padding_mask


def train(model, optimizers, schedulers, train_loader, valid_loader, args, use_warmup = False,use_Lc = False, is_uea=False):
    total_loss_list, total_acc_list, prob_list = [], [], []
    best_point = -999
    initial_tau = 1.3

    patience = args.patience if hasattr(args, "patience") else 10
    epochs_no_improve = 0
    epochs_no_improve_total = 0
    if hasattr(model, "reset_seg_cache"):
        model.reset_seg_cache()

    profile_times = [] 
    for epoch in range(args.num_epochs):
        t0 = perf_counter()
        total_pred_loss = 0.0
        avg_m_cnt = 0.0
        tau = initial_tau
    
        for batch_idx, batch in enumerate(train_loader):
            data, target, sample_ids, padding_mask = _parse_batch(batch)

            if is_uea:
                # [B, seq_len, C] -> [B, C, seq_len]
                data = data.permute(0, 2, 1)

            data = data.to(model.device)
            target = target.to(model.device)

            pred_logits, m, z_tilde, z, logit, x, probs = model(
                data,
                padding_mask,
                training=True,
                tau=tau,
                sample_ids=sample_ids
            )

            pred_loss_batch = predictor_loss(target, pred_logits)
            # print(pred_loss_batch)
            avg_m_cnt += m.sum(dim=(1,2)).mean()

            optimizers['sel'].zero_grad()
            optimizers['pred'].zero_grad()

            selected = (z_tilde.abs() > 1e-8).float().sum().item()
            total = z_tilde.numel()


            selection_rate = selected / total

            def compute_penalty(selection_rate, T):
                return torch.where(
                    selection_rate <= T,
                    (selection_rate / T) - 1,
                    (selection_rate - T) / (1 - T)
                )

            selection_loss = compute_penalty(torch.tensor(selection_rate), 0.5)

            lambda_1 = wandb.config.lambda_1 if hasattr(wandb.config, 'lambda_1') else 0.01
            lambda_1 = lambda_1 * selection_loss

            if use_warmup:
                if epochs_no_improve_total <= patience//2 :
                    lambda_1 = 0
        
            if use_Lc:
                lambda_c = args.lambda_c if hasattr(args, "lambda_c") else 0.001
                Lc_soft = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).sum(dim=(2,3)).mean()

            mask = (z_tilde.abs() > 1e-8).float()
            B, M, seq_len = z_tilde.shape
            diff = torch.abs(mask[:, :, 1:] - mask[:, :, :-1])
            Lc = diff.sum(dim=(2)).mean().cpu().item()
            m_sparsity = lambda_1 * m.mean()
            if not use_Lc:
                pred_loss = pred_loss_batch + m_sparsity
            else:
                pred_loss = pred_loss_batch + m_sparsity + lambda_c * Lc_soft

            pred_loss.backward()

            optimizers['pred'].step()
            optimizers['sel'].step()
            if schedulers['pred'] is not None:
                schedulers['pred'].step()
            if schedulers['sel'] is not None:
                schedulers['sel'].step()

            total_pred_loss += pred_loss.item()
        
        train_elapsed = perf_counter() - t0

        avg_pred = total_pred_loss / len(train_loader)
        avg_m_cnt = avg_m_cnt / len(train_loader)

        # model.selector.step_epoch()

        if epoch % 1 == 0 or epoch == args.num_epochs - 1:
            v0 = perf_counter()
            val_acc, selection_rate, Lc = valid(model, valid_loader, is_uea = is_uea , tau=tau)
            valid_elapsed = perf_counter() - v0
            # print('Warning best model is selected only considering accruracy')
            point_selection_rate = True

            if point_selection_rate:   
                point = val_acc - 0.2 * selection_rate
            else:
                point = val_acc

            wandb.log({
                "epoch": epoch,
                "lambda_1": lambda_1,
                "pred_loss": avg_pred,
                "avg_m_cnt": avg_m_cnt,
                "valid_accuracy": val_acc,
                "selection_rate": selection_rate,
                "connectivity Loss" : Lc,
                "point": point,
            })

            if point >= best_point:
                best_point = point
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
                
                if selection_rate < 0.8 and lambda_1 != 0.0:
                    print("reduce patience")
                    patience = max(10, patience // 1.5)

            else:
                if selection_rate > 0.99:
                    epochs_no_improve = 0
                else: 
                    epochs_no_improve += 1
                
                epochs_no_improve_total += 1
                print(epochs_no_improve_total)

            print("-" * 100)
            print(f"Epoch {epoch}: pred_loss={avg_pred:.4f}, valid_acc={val_acc:.4f}, selection_rate={selection_rate:.4f} , Lc={Lc:.4f}, point={point:.4f}")
            print("-" * 100)

            total_loss_list.append(avg_pred)
            total_acc_list.append(val_acc)

        if epochs_no_improve >= patience and epoch > args.num_epochs // 5:
            print(f"⏹️ Early stopping at epoch {epoch} - best point: {best_point:.4f}")
            break

    return best_model, total_loss_list, total_acc_list, prob_list

def test(model, test_loader, is_uea = False, tau = 1.3):
    model.eval()
    total_correct = 0
    total_samples = 0 
    total_transitions = 0
    total_masks       = 0
    total_segment_count = 0
    total_mask_units = 0

    total_selected = 0
    total_steps = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data, target, sample_ids, padding_mask = _parse_batch(batch)

            if is_uea:
                data = data.permute(0, 2, 1)

            data = data.to(model.device)

            pred_logits, m, z_tilde, z, logit, x, probs = model(
                data, padding_mask, training=False, tau=tau, sample_ids=sample_ids
            )
            
            mask = (z_tilde.abs() > 1e-8).float()

            mask_bin = (mask > 0.5).int()
            padded = torch.nn.functional.pad(mask_bin, (1, 0))
            starts = (padded[:, :, 1:] - padded[:, :, :-1]) == 1
            total_segment_count += starts.sum().item()
            total_mask_units += starts.size(0) * starts.size(1)

            diff = torch.abs(mask[:, :, 1:] - mask[:, :, :-1])
            total_transitions += diff.sum().item()
            total_masks       += diff.size(0) * diff.size(1)

            y_pred = torch.argmax(pred_logits, dim=1).cpu()
            y_true = target.view(-1).cpu()
            print("y_pred.shape", y_pred.shape)
            print("y_true.shape", y_true.shape)
            total_correct += (y_pred == y_true).sum().item()
            print("total_correct", total_correct)
            total_samples += y_true.size(0)
            selected = (z_tilde.abs() > 1e-8).float().sum().item()
            total = z_tilde.numel()

            total_selected += selected
            total_steps += total

    avg_segment_count = total_segment_count / total_mask_units
    acc = total_correct / total_samples
    Lc = total_transitions / total_masks   
    selection_rate = total_selected / total_steps
    model.train()

    return acc, selection_rate, Lc, avg_segment_count

def valid(model, valid_loader, is_uea = False, tau= 1.3):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_transitions = 0
    total_masks       = 0
    total_selected = 0
    total_steps = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            data, target, sample_ids, padding_mask = _parse_batch(batch)

            if is_uea:
                data = data.permute(0, 2, 1)

            data = data.to(model.device)

            pred_logits, m, z_tilde, z, logit, x, probs = model(
                data, padding_mask, training=False, tau=tau, sample_ids=sample_ids
            )

            mask = (z_tilde.abs() > 1e-8).float()

            diff = torch.abs(mask[:, :, 1:] - mask[:, :, :-1])  # [B, M, seq_len - 1]
            total_transitions += diff.sum().item()
            total_masks       += diff.size(0) * diff.size(1)    # B · M

            y_pred = torch.argmax(pred_logits, dim=1).cpu()
            y_true = target.view(-1).cpu()

            total_correct += (y_pred == y_true).sum().item()
            total_samples += y_true.size(0)

            selected = (z_tilde.abs() > 1e-8).float().sum().item()
            total = z_tilde.numel()

            total_selected += selected
            total_steps += total

    acc = total_correct / total_samples
    Lc = total_transitions / total_masks   
    selection_rate = total_selected / total_steps
    model.train()

    return acc, selection_rate, Lc