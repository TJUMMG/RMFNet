import utility

import torch
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss


        self.error_last = 1e8



    def test(self):
        torch.set_grad_enabled(False)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
                idx_img = 1
                for lbd, hbd in tqdm(d, ncols=80):

                    lbd, hbd = self.prepare(lbd, hbd)
                    hbdp, K1target, K2target, K3target, K1super, K2super, K3super = self.model([lbd, hbd])        ###

                    hbdp = utility.quantize(hbdp, self.args.rgb_range)

                    save_list = [hbdp]
                    self.ckp.log[-1, idx_data] += utility.calc_psnr(
                        hbdp, hbd, self.scale, self.args.rgb_range
                    )
                    if self.args.save_gt:
                        save_list.extend([hbd])
                    if self.args.save_results:
                        self.ckp.save_results(d, idx_img, save_list)
                        idx_img = idx_img +1

                self.ckp.log[-1, idx_data] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        self.scale,
                        self.ckp.log[-1, idx_data],
                        best[0][idx_data],
                        best[1][idx_data] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch > self.args.epochs


