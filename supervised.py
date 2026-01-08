"""
监督学习流程。划分数据集，创建网络和优化器，训练过程中每一个epoch结束会在验证集上测试。

用法:
    python supervised.py                           # 从头训练
    python supervised.py --resume path/to/ckpt.pt  # 从checkpoint恢复
"""

from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import CNNModel
import torch.nn.functional as F
import torch
import os
import argparse
from datetime import datetime
from tqdm import tqdm

if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser(description='Supervised learning for Mahjong')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=17, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    args = parser.parse_args()

    # 用时间戳标识本次训练
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = f'supervised_checkpoint/{timestamp}/'
    log_dir = f'supervised_runs/{timestamp}/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"Training run: {timestamp}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"TensorBoard dir: {log_dir}")

    # Load dataset
    splitRatio = 0.9
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=True)
    vloader = DataLoader(dataset=validateDataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = CNNModel().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            model.load_state_dict(torch.load(args.resume))
            # 尝试从文件名提取epoch数
            try:
                filename = os.path.basename(args.resume)
                start_epoch = int(filename.split('_')[1].split('.')[0])
                print(f"Resuming from epoch {start_epoch}")
            except:
                print("Could not parse epoch from filename, starting from epoch 0")
        else:
            print(f"Checkpoint not found: {args.resume}, starting from scratch")

    # Train and validate
    for e in range(start_epoch, args.epochs):
        print(f'\nEpoch {e + 1}/{args.epochs}')
        model.train()
        epoch_loss = 0.0

        # Training loop with tqdm
        pbar = tqdm(loader, desc=f'Training Epoch {e + 1}')
        for i, d in enumerate(pbar):
            input_dict = {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            logits, _ = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.set_postfix({'loss': f'{epoch_loss / (i + 1):.4f}'})

            # TensorBoard logs
            writer.add_scalar('Loss/train', epoch_loss / (i + 1), global_step=e * len(loader) + i)

        # Validation loop with tqdm
        print('Running validation...')
        model.eval()
        correct = 0
        vbar = tqdm(vloader, desc='Validating')
        for i, d in enumerate(vbar):
            input_dict = {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            with torch.no_grad():
                logits, _ = model(input_dict)
                pred = logits.argmax(dim=1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
            vbar.set_postfix({'acc': f'{correct / ((i + 1) * args.batch_size):.4f}'})

        acc = correct / len(validateDataset)
        print(f'Epoch {e + 1} Validate acc: {acc:.4f}')

        # TensorBoard logs
        writer.add_scalar('Accuracy/validation', acc, global_step=e)

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{e + 1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')

    # Save final model for RL training
    final_path = os.path.join(checkpoint_dir, 'final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete!")
    print(f"Final model: {final_path}")
    print(f"Use this for RL training: python train.py --pretrain {final_path}")

    writer.close()       