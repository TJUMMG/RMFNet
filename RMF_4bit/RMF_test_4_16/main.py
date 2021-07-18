import torch
import os
import utility
import data
import model
from option import args
from trainer import Trainer


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = None
    t = Trainer(args, loader, model, loss, checkpoint)
    print("Trainer prepare successed!")
    while not t.terminate():
        print("Training!!!")
        t.train()
        print("Testing!!!")
        t.test()

    checkpoint.done()

