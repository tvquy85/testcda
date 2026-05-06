import os
import pickle
import random
import sys

import numpy as np
import torch as torch

from evaluator import evaluate
from model import get_loss, StockMixer
from preprocess import market_state_from_closes

np.random.seed(123456789)
torch.random.manual_seed(12345678)
# Debug: Kiểm tra CUDA availability và chọn device
print("=" * 60)
print("🔍 DEBUG: KIỂM TRA DEVICE SELECTION")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print("✅ CUDA có sẵn!")
    print(f"🎮 Số lượng GPU: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {gpu_name}")
    device = torch.device("cuda")
    print(f"🎯 Device được chọn: {device}")
else:
    print("❌ CUDA không có sẵn")
    device = torch.device("cpu")
    print(f"🎯 Device được chọn: {device}")

print("=" * 60)
print()

data_path = '../dataset'
market_name = 'NASDAQ'
relation_name = 'wikidata'
stock_num = 1026
lookback_length = 16
epochs = 100
valid_index = 756
test_index = 1008
fea_num = 5
market_num = 20
steps = 1
learning_rate = 0.001
alpha = 0.1
scale_factor = 3
activation = 'HardSwish'
depth=3

# Test

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataset_path = os.path.join(BASE_DIR, "dataset", market_name)

#dataset_path = "../dataset/" + market_name
if market_name == "SP500":
    data = np.load("../dataset/SP500/SP500.npy")
    data = data[:, 915:, :]
    price_data = data[:, :, -1]
    mask_data = np.ones((data.shape[0], data.shape[1]))
    eod_data = data
    gt_data = np.zeros((data.shape[0], data.shape[1]))
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (
                                           data[ticket][row][-1] - data[ticket][row - steps][-1]
                                   ) / data[ticket][row - steps][-1]
else:
    with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f)
    with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
        mask_data = pickle.load(f)
    with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
        gt_data = pickle.load(f)
    with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
        price_data = pickle.load(f)
market_ctx = market_state_from_closes(eod_data)
# eod_data = append_technical_indicators(eod_data)
# print(eod_data)
fea_num = eod_data.shape[2]
trade_dates = mask_data.shape[1]
model = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    market=market_num,
    depth=depth,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)


# Test
def validate(start_index, end_index):
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.0
        reg_loss = 0.0
        rank_loss = 0.0
        for cur_offset in range(
                start_index - lookback_length - steps + 1,
                end_index - lookback_length - steps + 1,
        ):
            data_batch, mask_batch, price_batch, gt_batch, ctx = map(
                lambda x: torch.Tensor(x).to(device), get_batch(cur_offset)
            )
            prediction = model(data_batch, ctx)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(
                prediction, gt_batch, price_batch, mask_batch, stock_num, alpha
            )
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[
            :, cur_offset - (start_index - lookback_length - steps + 1)
            ] = cur_rr[:, 0].cpu()
            cur_valid_gt[
            :, cur_offset - (start_index - lookback_length - steps + 1)
            ] = gt_batch[:, 0].cpu()
            cur_valid_mask[
            :, cur_offset - (start_index - lookback_length - steps + 1)
            ] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, cur_valid_perf


def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset: offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1),
        market_ctx[offset - 1]
    )


print(market_name, " : ", end="", flush=True)
for epoch in range(epochs):
    print("epoch{}##########################################################".format(epoch + 1))
    np.random.shuffle(batch_offsets)
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    for j in range(valid_index - lookback_length - steps + 1):
        if batch_offsets[j] == 0:
            continue
        data_batch, mask_batch, price_batch, gt_batch, ctx = map(
            lambda x: torch.Tensor(x).to(device), get_batch(batch_offsets[j])
        )
        optimizer.zero_grad()
        prediction = model(data_batch, ctx)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(
            prediction, gt_batch, price_batch, mask_batch, stock_num, alpha
        )
        cur_loss = cur_loss
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)
    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)

    # Print results
    print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))

    print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))



    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf

    print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
    best_test_perf['mse'],
    best_test_perf['IC'],
    best_test_perf['RIC'],
    best_test_perf['prec_10'],
    best_test_perf['sharpe5']))
    print("Best val performance:")
    print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
    best_valid_perf['mse'],
    best_valid_perf['IC'],
    best_valid_perf['RIC'],
    best_valid_perf['prec_10'],
    best_valid_perf['sharpe5']))
    
print(
    "IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}".format(
        best_test_perf["IC"],
        best_test_perf["RIC"],
        best_test_perf["prec_10"],
        best_test_perf["sharpe5"],
    ),
    "\n\n",
)