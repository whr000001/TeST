import numpy as np
import random


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        c, t, h, w = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs, self.size)
        imgs = imgs[:, :, i:i + h, j:j + w]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, imgs):
        c, t, h, w = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, :, i:i + th, j:j + tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if random.random() < self.p:
            # c x t x h x w
            return np.flip(imgs, axis=3).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
