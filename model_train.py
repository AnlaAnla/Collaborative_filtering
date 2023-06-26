import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time
from recommend_model import MatrixFactorization

cudnn.benchmark = True
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class MovieLensDataset(Dataset):
    def __init__(self, ratings_file):
        self.user_ids = []
        self.item_ids = []
        self.ratings = []
        with open(ratings_file, 'r') as f:
            for line in f:
                user_id, item_id, rating = line.strip().split('::')
                self.user_ids.append(int(user_id))
                self.item_ids.append(int(item_id))
                self.ratings.append(float(rating))

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]



def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    # 计算完成百分比
    percent_complete = f"{(100 * (iteration / float(total))):.{decimals}f}"
    # 计算进度条填充长度
    filled_length = int(length * iteration // total)
    # 创建进度条字符串
    bar = fill * filled_length + '-' * (length - filled_length)
    # 打印进度条
    print(f'\r{prefix} |{bar}| {percent_complete}% {suffix}', end=print_end)
    # 完成时打印新行
    if iteration == total:
        print()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()


    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)



        # 初始化进度条
        bar_len = len(train_loader)
        print_progress_bar(0, bar_len, prefix='进度:', suffix='完成', length=50)

        # train
        model.train()
        running_loss = 0.0

        for i, (user_ids, item_ids, ratings) in enumerate(train_loader):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings.float())

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * len(user_ids)


            print_progress_bar(i + 1, bar_len, prefix='进度:', suffix='完成', length=50)

        scheduler.step()

        rmse = torch.sqrt(torch.tensor(running_loss / len(test_dataset)))
        print(f'train Epoch {epoch + 1}, RMSE: {rmse.item()}')

        # val
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)

                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                total_loss += loss.item() * len(user_ids)
            rmse = torch.sqrt(torch.tensor(total_loss / len(test_dataset)))
        print(f'val Epoch {epoch + 1}, RMSE: {rmse.item()}')

    return model



train_dataset = MovieLensDataset('card_user_item/rating_buy_train.data')
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

test_dataset = MovieLensDataset('card_user_item/rating_buy_test.data')
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


if __name__ == '__main__':

    model = MatrixFactorization(num_users=8391 + 1, num_items=2016 + 1, embedding_dim=1024)
    model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    model = train_model(model, criterion, optimizer,
                           exp_lr_scheduler, num_epochs=130)



    torch.save(model.state_dict(), 'recommend_card01.pth')