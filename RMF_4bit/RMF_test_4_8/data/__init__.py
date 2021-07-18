from importlib import import_module
from torch.utils.data import dataloader



class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:

            module_train = import_module('data.' + args.data_train)
            print(module_train)
            trainset = getattr(module_train, args.data_train)(args)
            print(trainset)
            self.loader_train = dataloader.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []

        if args.data_test in ['Kodak', 'DIV2K']:
            module_test = import_module('data.benchmarktest')
            testset = getattr(module_test, 'BenchmarkTest')(args, train=False, name = args.data_test)
        else:
            module_test = import_module('data.' +  args.data_test)
            testset = getattr(module_test, args.data_test)(args, train=False, name = args.data_test)


 

        self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
