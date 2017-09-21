from unittest import TestCase
import numpy as np
import src.memorize as mem
import scipy.misc as misc


class TestMemorize(TestCase):
    def test_to_image_form(self):
        data = np.load("../res/test_data.npy").flatten()
        image = misc.toimage(misc.imread("../res/test_data.png"))
        data = misc.toimage(mem.to_image_form(data))
        self.assertEqual(data, image)
        image.close()
        data.close()

    def test_net(self):
        data, net_data = mem.train_and_get_data()
        diff = data - net_data.flatten()
        self.assertEquals(len(np.where(diff > 1e-6)[0]), 0)