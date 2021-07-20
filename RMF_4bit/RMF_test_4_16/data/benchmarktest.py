import os
from data import ImageData



class BenchmarkTest(ImageData.ImageData):
    def __init__(self, args, name = '', train=True):
        super(BenchmarkTest, self).__init__(args, name = name, train=train, benchmark=True)

    def _scan(self):
        list_tar = []
        list_input = []

        # UST-HK
        idx_begin = 1
        idx_end = 2



        # UST-HK
        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i)
            list_tar.append(os.path.join(self.apath_tar, filename + self.ext))
            list_input.append(os.path.join(self.apath_in, filename + self.ext))

        return list_tar, list_input


    def _set_filesystem(self, dir_data):
        # UST-HK
        self.apath_tar = os.path.join(dir_data, 'HK', 'Source1200')
        self.apath_in = os.path.join(dir_data, 'HK', 'Source1200_4b')
        self.ext = '.png'
