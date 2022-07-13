import torch

def train(trainloader, model, optimizer, criterion, recorder=None):
    # Training mode.
    model.train()
    torch.set_grad_enabled(True)

    for it, (labels, images) in enumerate(trainloader):
        # Initialize gradient
        optimizer.zero_grad()

        # Drop images and labels into GPU
        images = images.cuda().detach()
        labels = labels.cuda().detach()

        # Forward
        predicts = model(images)

        # Backward
        loss = criterion(predicts, labels)

        # 1. Calculate gradient
        loss.backward()
        # 2. Update parameters
        optimizer.step()

        if recorder:
            recorder.write("loss", loss ,it)