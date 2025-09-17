import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
import json

# Import shared utilities
from utils import (
    # Model components
    TimeSeriesTransformer, SolarPowerDataset_Transformer,
    # Loss functions
    masked_mse_loss,
    # Data processing
    scale_dataframe_by_campus, create_model, get_target_feature_index,
    # Constants
    ENCODER_FEATURES, DECODER_INPUT_FEATURES, LOOKBACK_STEPS, LOOKFORWARD_STEPS,
    BATCH_SIZE, MAX_GRAD_NORM, PATIENCE, SENTINEL_VALUE,
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT_RATE
)

# =============================================================================
#  Phase 2A specific configurations
# =============================================================================
# Data paths
ALL_TRAIN_DATA_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\train_selected_sites.csv"
ALL_VAL_DATA_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\validation_enhanced.csv"
STATS_FILE_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\scaler\campus_scale.txt"

# Model paths
PHASE1_MODEL_PATH = r"C:\Python\Trained_models\Elite_Model_Transformer_Phase1_Complete\transformer_elite_model_phase1_V3.pth"
PHASE2A_SAVE_DIR = r"C:\Python\Trained_models\Elite_Model_Transformer_Phase2A_Complete"
os.makedirs(PHASE2A_SAVE_DIR, exist_ok=True)
PHASE2A_MODEL_SAVE_PATH = os.path.join(PHASE2A_SAVE_DIR, 'transformer_elite_model_phase2a.pth')

# Phase 2A specific hyperparameters
LEARNING_RATE_PHASE2A = 0.5e-6
EPOCHS_PHASE2A = 5
PATIENCE_PHASE2A = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get target feature index for phase 2A training
TARGET_FEATURE_INDEX = get_target_feature_index()


# Model components, dataset, loss functions, and data processing utilities 
# are imported from utils.py

# =============================================================================
#  3. 核心训练逻辑: Phase 2A (逐步监督)
# =============================================================================
def train_epoch_phase2a(model, data_loader, optimizer, criterion, device):
    """
    执行阶段2A的一个训练Epoch，采用“逐步监督”策略。
    """
    model.train()
    total_loss_for_logging = 0
    num_batches_processed = 0

    progress_bar = tqdm(data_loader, desc=f"Phase 2A Training (Step-wise Supervision)", leave=True, dynamic_ncols=True)

    for batch in progress_bar:
        # 数据准备 (与第一阶段相同)
        src, tgt_input, tgt_output, tgt_padding_mask, src_padding_mask = \
            batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                batch[3].to(device), batch[4].to(device)

        # 梯度清零，准备累积一个Batch中所有修复轮次的梯度
        optimizer.zero_grad()

        # ----- 核心逻辑：带“逐步监督”的多轮并行填充 -----
        nan_mask = (tgt_output[:, :, 0] == SENTINEL_VALUE)  # 目标是单列，所以索引为0
        max_nans_in_batch = torch.max(torch.sum(nan_mask, dim=1)).item()

        # 如果批次中没有NaN，则进行一次标准的有监督训练
        if max_nans_in_batch == 0:
            outputs = model(src, tgt_input, tgt_padding_mask, src_padding_mask)
            loss = criterion(outputs, tgt_output)
            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                total_loss_for_logging += loss.item()
        else:
            # 初始化"工作台"
            working_tgt_input = tgt_input.clone()
            nan_cumsum = torch.cumsum(nan_mask.int(), dim=1)

            # 开始多轮并行修复与逐步监督
            for k in range(int(max_nans_in_batch)):
                # a. 进行一次并行的前向传播
                current_model_output = model(src, working_tgt_input, tgt_padding_mask, src_padding_mask)

                # b. 核心改进: 立即计算当前轮次的损失
                loss_k = criterion(current_model_output, tgt_output)

                # c. 核心改进: 如果损失有效，立即进行反向传播以累积梯度
                if not torch.isnan(loss_k) and loss_k.item() > 0:
                    # 我们对每一轮的损失进行加权，越靠后的轮次权重越大，鼓励模型在信息更全时做出更好的预测
                    # 这是一个可选的trick，可以从简单的 loss_k.backward() 开始
                    weight = (k + 1) / max_nans_in_batch
                    (loss_k * weight).backward()

                    # 日志记录只记录原始loss
                    if k == max_nans_in_batch - 1:  # 只记录最后一轮的loss作为代表
                        total_loss_for_logging += loss_k.item()

                # d. 准备下一轮的输入 (如果是最后一轮则无需准备)
                if k == max_nans_in_batch - 1:
                    break

                # 定位第k+1个NaN的位置
                is_kth_nan = (nan_cumsum == k + 1) & nan_mask

                # 为了避免对需要梯度的叶子变量进行原地操作，我们.detach()预测值
                predictions_for_kth_nan = current_model_output.detach()[:, :, 0][is_kth_nan]

                # 更新工作台的对应位置
                working_tgt_input[:, :, TARGET_FEATURE_INDEX][is_kth_nan] = predictions_for_kth_nan

        # ----- 所有修复轮次结束 -----
        # d. 执行一次优化器步骤，使用累积的所有梯度来更新模型权重
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        num_batches_processed += 1
        if num_batches_processed > 0:
            progress_bar.set_postfix(avg_loss=total_loss_for_logging / num_batches_processed)

    return total_loss_for_logging / num_batches_processed if num_batches_processed > 0 else 0


def validate_epoch_phase2a(model, data_loader, criterion, device):
    """
    在验证集上评估Phase 2A模型。评估逻辑与训练逻辑相同，以保证一致性。
    """
    model.eval()
    total_val_loss = 0

    progress_bar = tqdm(data_loader, desc=f"Phase 2A Validation", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch in progress_bar:
            src, tgt_input, tgt_output, tgt_padding_mask, src_padding_mask = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                    batch[3].to(device), batch[4].to(device)

            nan_mask = (tgt_output[:, :, 0] == SENTINEL_VALUE)
            max_nans_in_batch = torch.max(torch.sum(nan_mask, dim=1)).item()

            final_output = None
            if max_nans_in_batch == 0:
                final_output = model(src, tgt_input, tgt_padding_mask, src_padding_mask)
            else:
                working_tgt_input = tgt_input.clone()
                nan_cumsum = torch.cumsum(nan_mask.int(), dim=1)
                for k in range(int(max_nans_in_batch)):
                    current_model_output = model(src, working_tgt_input, tgt_padding_mask, src_padding_mask)
                    if k == max_nans_in_batch - 1:
                        final_output = current_model_output
                        break

                    is_kth_nan = (nan_cumsum == k + 1) & nan_mask
                    predictions_for_kth_nan = current_model_output[:, :, 0][is_kth_nan]
                    working_tgt_input[:, :, TARGET_FEATURE_INDEX][is_kth_nan] = predictions_for_kth_nan

            loss = criterion(final_output, tgt_output)
            if not torch.isnan(loss):
                total_val_loss += loss.item()

    return total_val_loss / len(data_loader)


# =============================================================================
#  4. 主执行逻辑
# =============================================================================
if __name__ == "__main__":
    print(f"--- Transformer精英模型微调 (第二阶段 A - 逐步监督) ---")
    print(f"使用设备: {DEVICE}")

    # 1. 加载和准备数据 (与第一阶段相同)
    print("正在加载数据...")
    df_train_all = pd.read_csv(ALL_TRAIN_DATA_PATH)
    df_val_all = pd.read_csv(ALL_VAL_DATA_PATH)
    with open(STATS_FILE_PATH, 'r') as f:
        scaling_stats = json.load(f)

    print("正在标准化数据...")
    df_train_scaled = scale_dataframe_by_campus(df_train_all, scaling_stats)
    df_val_scaled = scale_dataframe_by_campus(df_val_all, scaling_stats)

    print("正在创建Dataset和DataLoader...")
    train_dataset = SolarPowerDataset_Transformer(df_train_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
    val_dataset = SolarPowerDataset_Transformer(df_val_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    # 2. 初始化模型 (使用从utils导入的函数)
    model = create_model(device=DEVICE)

    # 3. !! 关键步骤: 加载在第一阶段训练好的模型权重 !!
    if os.path.exists(PHASE1_MODEL_PATH):
        print(f"成功加载第一阶段预训练模型: {PHASE1_MODEL_PATH}")
        model.load_state_dict(torch.load(PHASE1_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"警告: 未找到第一阶段模型 {PHASE1_MODEL_PATH}。将从头开始训练。")

    # 4. 初始化优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_PHASE2A)
    criterion = masked_mse_loss

    # 5. 开始训练
    print("开始微调 (Phase 2A)...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS_PHASE2A):
        print(f"\nEpoch {epoch + 1}/{EPOCHS_PHASE2A}")

        train_loss = train_epoch_phase2a(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate_epoch_phase2a(model, val_loader, criterion, DEVICE)

        print(f"    Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), PHASE2A_MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"      -> New best validation loss. Model saved to {PHASE2A_MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_PHASE2A:
                print(f"  > 早停触发！在 Epoch {epoch + 1}。")
                break

    print(f"\n精英模型第二阶段A微调完成。最佳泛化验证损失: {best_val_loss:.6f}")
    print(f"模型已保存在: {PHASE2A_MODEL_SAVE_PATH}")
