import os
from os.path import join as PJ

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

import yaml

from utils import parser_args, data_transform, cal_acc, MyRecorder
from datasets import MyDataset
from models import AlexNet, VGG16, ReducedVGG, FourConv

###########
# Configs # 
###########

config_path = "./configs.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
args = parser_args(config)

save_path = PJ(save_root, exp_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"# Experiment: {exp_name}")
print("# Configs:")
for i in configs:
    print(f"  {i}: {configs[i]}")

recorder = MyRecorder(save_path=save_path, configs=config)

################
# Load dataset #
################

print("==== Loading dataset ====")

transform = data_transform(new_size=args.new_size, train=True)
trainset = MyDataset(data_root=args.data_root, data_file="train.txt", transform=transform)

transform = data_transform(new_size=args.new_size, train=False)
valset = MyDataset(data_root=args.data_root, data_file="val.txt", transform=transform)
testset = MyDataset(data_root=args.data_root, data_file="test.txt", transform=transform)

# Dataloader
trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size,
                         shuffle=True, num_workers=2)
valloader = DataLoader(dataset=valset, batch_size=args.test_size,
                       shuffle=False, num_workers=2)
testloader = DataLoader(dataset=testset, batch_size=args.test_size,
                        shuffle=False, num_workers=2)
print("==== Loaded dataset ====")

###############
# Bulid model #
###############
print("==== Buliding model ====")

if args.model == "AlexNet":
    model = AlexNet()

elif args.model == "VGG16":
    model = VGG16()

elif args.model == "ReducedVGG":
    model = ReducedVGG()

elif args.model == "FourConv":
    model = FourConv()

# Drop model into GPU
model.cuda()

# Criterion and Optimizer
def criterion(predicts, targets):
    """ predicts: (N, C), targets: (N) """
    ce = nn.CrossEntropyLoss()
    loss = ce(predicts, targets.reshape(-1))
    return loss

optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0)
# Learning rate decay scheduler
scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.decay_steps,
                                     gamma=args.gamma, last_epoch=-1)
print("==== Builded model ====")

##################
# Train and Save #
##################

if args.resume:

    print("==== Resume training ====")

    # load model
    model.load_state_dict(PJ(save_path, "model.pt"))
    optimizer.load_state_dict(PJ(save_path, "optimizer.pt"))

    print("==== Resumed training progress ====")

print("==== Start training ====")

for epoch in range(arg.last_epoch+1, arg.max_epoch+1):
    # train
    train(trainloader=trainloader, model=model, 
          optimizer=optimizer, criterion=criterion,
          recorder=recorder)
    scheduler.step()

    # save progress
    torch.save(model.state_dict(), PJ(save_path, "model.pt"))
    torch.save(optimizer.state_dict(), PJ(save_path, "optimizer.pt"))

    # validate
    results = test(dataloader=valloader, model=model)
    acc = cal_acc(results=results, weighted=args.weighted)
    print(f"[Epoch: {epoch}] Accuracy: {acc:0.4f}")
    recorder.write("val_acc", acc, epoch)

print("==== Training ended ====")

# Saving model
torch.save(model.state_dict(), PJ(save_path, "model.pt"))
print("==== Saved model ====")

##############
# Load model # (optional)
##############

#model.load_state_dict(PJ(save_path, "model.pt"))

########
# Test #
########

print("==== Test ====")

results = test(dataloader=testloader, model=model)
acc = cal_acc(results=results, weighted=args.weighted)
print(f"Accuracy: {acc:0.4f}\n")

recorder.write("test_acc", acc, 1)
recorder.close()