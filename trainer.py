import torch
import torch.nn as nn
import torch.optim as optim

import cfg


def _train_epoch(model, epoch, dataloader, criterion, optimizer):
    model.train()  # set model to training mode

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(cfg.device)
        labels = labels.long().to(cfg.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f'epoch [{epoch}/{cfg.max_epoch}]', end=' | ')
        print(f'train iter [{batch_idx}/{len(dataloader)}]', end=' | ')
        print(f'loss={loss.item():.5f}')


def _val_epoch(model, epoch, dataloader, criterion, best_loss):
    model.eval()  # set model to evaluation mode

    total_val_loss = 0.
    best_val_loss = best_loss
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            images = images.to(cfg.device)
            labels = labels.long().to(cfg.device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            print(f'epoch [{epoch}/{cfg.max_epoch}]', end=' | ')
            print(f'val iter [{batch_idx}/{len(dataloader)}]', end=' | ')
            print(f'loss={loss.item():.5f}')

        mean_val_loss = total_val_loss / len(dataloader)

        if mean_val_loss < best_loss:
            torch.save(model.state_dict(), cfg.best_model_path)

            best_val_loss = mean_val_loss

        print(f'epoch [{epoch}/{cfg.max_epoch}]', end=' | ')
        print(f'val', end=' | ')
        print(f'mean loss={mean_val_loss:.5f}', end=' | ')
        print(f'best loss={best_val_loss:.5f}')

        return best_val_loss


def train(model, train_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), cfg.learing_rate)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)

    best_loss = 1.0
    for epoch in range(cfg.start_epoch, cfg.max_epoch + 1):
        _train_epoch(model, epoch, train_dataloader, criterion, optimizer)
        best_loss = _val_epoch(model, epoch, test_dataloader, criterion, best_loss)
        schedular.step()
