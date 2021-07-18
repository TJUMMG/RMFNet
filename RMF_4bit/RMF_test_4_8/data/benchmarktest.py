import os
from data import ImageData


class BenchmarkTest(ImageData.ImageData):
    def __init__(self, args, name = '', train=True):
        super(BenchmarkTest, self).__init__(args, name = name, train=train, benchmark=True)

    def _scan(self):
        list_tar = []
        list_input = []

        # Kodak
        idx_begin = 0
        idx_end = 24

        # Kodak
        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>2}'.format(i)
            list_tar.append(os.path.join(self.apath_tar, 'kodim' + filename + self.ext))
            list_input.append(os.path.join(self.apath_in, 'kodim' + filename + self.ext))

        return list_tar, list_input



    def _set_filesystem(self, dir_data):
        # Kodak
        self.apath_in = os.path.join(dir_data, 'kodak', 'Kodak_4b')  ###
        self.apath_tar = os.path.join(dir_data, 'kodak', 'Kodak')  ###

        self.ext = '.png'

