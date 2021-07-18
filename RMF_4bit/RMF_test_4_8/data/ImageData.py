from data import common
import cv2      ###
import torch.utils.data as data

class ImageData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.testbin = args.testbin
        self._set_filesystem(args.dir_data)

        print('initial image data now!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('benchmark',self.benchmark)
        if self.benchmark:
            print('BenchmarkScaning')
            self.images_tar, self.images_input = self._scan()
            print('Scan finished!')
        elif args.ext == 'img':
            self.images_tar, self.images_input = self._scan()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError



    def __getitem__(self, idx):

        img_input, img_tar = self._load_file(idx)
        img_input, img_tar = common.set_channel([img_input, img_tar], self.args.n_colors)
        img_input, img_tar = self._get_patch(img_input, img_tar)
        input_tensor, tar_tensor = common.np2Tensor([img_input, img_tar], self.args.rgb_range)
        return input_tensor, tar_tensor

    def __len__(self):
        return len(self.images_tar)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):

        idx = self._get_index(idx)
        if self.benchmark:
            img_input = cv2.imread(self.images_input[idx], 3)
            img_tar = cv2.imread(self.images_tar[idx], 3)
        elif self.args.ext == 'img':
            img_input = cv2.imread(self.images_input[idx], 3)
            img_tar = cv2.imread(self.images_tar[idx], 3)
        return img_input, img_tar

    def _get_patch(self, img_input, img_tar):

        patch_size = self.args.patch_size
        scale = self.scale

        img_input, img_tar = common.get_patch_test(img_input, img_tar, scale)

        return img_input, img_tar



