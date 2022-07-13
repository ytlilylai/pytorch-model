import torch

def test(dataloader, model):
    # Evaluate mode. Take of dropout.
    model.eval()
    # Not calculate gradient
    torch.set_grad_enabled(False)

    results = {'predict': [], 'label': []}
    for it, (labels, images) in enumerate(dataloader):

        # Drop images and labels into GPU
        images = images.cuda().detach()
        labels = labels.cuda().detach()

        # Take class with largest score as predict
        predicts = model(images)
        predicts = torch.argmax(predicts, 1)

        # Record results
        results['predict'] += predicts.tolist()
        results['label'] += labels.tolist()

    return results