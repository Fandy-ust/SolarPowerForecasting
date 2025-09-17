import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import shared utilities
from utils import (
    # Model components
    TimeSeriesTransformer, SolarPowerDataset_Transformer,
    # Loss functions
    masked_mse_loss,
    # Data processing
    scale_dataframe_by_campus, create_model,
    # Constants
    ENCODER_FEATURES, DECODER_INPUT_FEATURES, LOOKBACK_STEPS, LOOKFORWARD_STEPS,
    BATCH_SIZE, MAX_GRAD_NORM, PATIENCE, SENTINEL_VALUE,
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT_RATE
)

# Phase 1 specific configurations
ALL_TRAIN_DATA_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\train_selected_sites.csv"
ALL_VAL_DATA_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\validation_enhanced.csv"
STATS_FILE_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\scaler\campus_scale.txt"
MODEL_SAVE_DIR = r"C:\Python\Trained_models\Elite_Model_Transformer_Phase1_Complete"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Phase 1 specific hyperparameters
EPOCHS = 100
LEARNING_RATE = 0.0001


# Model components are imported from utils.py


# Dataset, loss functions, and data processing utilities are imported from utils.py


# =================================================================================
# 5. 主执行逻辑
# =================================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 训练Transformer精英模型 (第一阶段 - 完整版) ---")
    print(f"--- Encoder 和 Decoder 均启用 Padding Mask ---")
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("正在加载数据...")
    df_train_all = pd.read_csv(ALL_TRAIN_DATA_PATH)
    df_val_all = pd.read_csv(ALL_VAL_DATA_PATH)
    with open(STATS_FILE_PATH, 'r') as f:
        scaling_stats = json.load(f)

    # 2. 准备数据
    print("准备数据管道...")
    df_val_global = df_val_all.copy()
    print("正在标准化训练数据...")
    df_train_scaled = scale_dataframe_by_campus(df_train_all, scaling_stats)
    print("正在标准化验证数据...")
    df_val_scaled = scale_dataframe_by_campus(df_val_global, scaling_stats)

    if df_train_scaled.empty or df_val_scaled.empty:
        print("致命错误: 标准化后训练或验证数据为空。")
        exit(1)

    # 3. 创建Dataset和DataLoader
    print("正在创建Dataset和DataLoader...")
    train_dataset = SolarPowerDataset_Transformer(df_train_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
    val_dataset = SolarPowerDataset_Transformer(df_val_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    # 4. 初始化模型 (使用从utils导入的函数)
    model = create_model(device=device)
    criterion = masked_mse_loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. <-- 核心改动: 训练循环，处理五个张量 -->
    print("开始训练Transformer精英模型...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    model_save_path = os.path.join(MODEL_SAVE_DIR, 'transformer_elite_model_phase1_V3.pth')

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        # <-- 解包五个张量 -->
        for src, tgt_input, tgt_output, tgt_padding_mask, src_padding_mask in tqdm(train_loader,
                                                                                   desc=f"Epoch {epoch + 1}/{EPOCHS} Train",
                                                                                   leave=False):
            # <-- 将所有五个张量移动到设备 -->
            src, tgt_input, tgt_output, tgt_padding_mask, src_padding_mask = \
                src.to(device), tgt_input.to(device), tgt_output.to(device), \
                    tgt_padding_mask.to(device), src_padding_mask.to(device)

            optimizer.zero_grad()
            # <-- 调用模型时传入 src_padding_mask -->
            outputs = model(src, tgt_input, tgt_padding_mask, src_padding_mask)

            loss = criterion(outputs, tgt_output)
            if torch.isnan(loss):
                print("警告: 训练时出现NaN Loss，跳过此批次。")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_train_loss += loss.item()

        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            # <-- 验证循环同样解包五个张量 -->
            for src, tgt_input, tgt_output, tgt_padding_mask, src_padding_mask in val_loader:
                src, tgt_input, tgt_output, tgt_padding_mask, src_padding_mask = \
                    src.to(device), tgt_input.to(device), tgt_output.to(device), \
                        tgt_padding_mask.to(device), src_padding_mask.to(device)

                # <-- 调用模型时同样传入 src_padding_mask -->
                outputs = model(src, tgt_input, tgt_padding_mask, src_padding_mask)
                loss = criterion(outputs, tgt_output)

                if torch.isnan(loss):
                    print("警告: 验证时出现NaN Loss。")
                else:
                    total_val_loss += loss.item()

        epoch_val_loss = total_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"    Epoch {epoch + 1}/{EPOCHS}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            print(f"      -> New best validation loss. Model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  > 早停触发！在 Epoch {epoch + 1}。")
                break

    print(f"\n精英模型训练完成。最佳泛化验证损失: {best_val_loss:.6f}")

    # 6. 可视化训练结果
    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, label='Training Loss (on 5 Champions)')
    plt.plot(val_losses, label='Validation Loss (on all other sites)')
    plt.title('Complete Transformer Elite Model Training History (Phase 1)')
    plt.xlabel('Epoch')
    plt.ylabel('Masked MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_SAVE_DIR, 'transformer_elite_model_training_history_complete.png'))
    plt.show()

    print(f"\n{'=' * 30} Transformer精英模型第一阶段 (完整版) 训练完毕！ {'=' * 30}")
    print(f"模型和训练历史图已保存在: {MODEL_SAVE_DIR}")
