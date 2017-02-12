import argparse
import cv2
import hickle as hkl
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.cvtransforms as T
import torchnet as tnt


parser = argparse.ArgumentParser(description='Fast Neural Style Demo')
parser.add_argument('--network', default='wave.hkl', type=str)
parser.add_argument('--resolution', default=256, type=int)


def f(o, params):
    def g(f, x, name, stride=1, padding=0):
        return f(x, params['%s.weight'%name], params['%s.bias'%name], stride, padding)
    o = F.relu(g(F.conv2d, o, 'c1', 1, 4))
    o = F.relu(g(F.conv2d, o, 'c2', 2, 1))
    o = F.relu(g(F.conv2d, o, 'c3', 2, 1))
    for i in range(1, 6):
        o += g(F.conv2d, F.relu(g(F.conv2d, o, 'r%d.c1'%i, padding=1)), 'r%d.c2'%i, padding=1)
    o = F.relu(g(F.conv_transpose2d, o, 'd1', 2, 1))
    o = F.relu(g(F.conv_transpose2d, o, 'd2', 2, 1))
    return g(F.conv2d, o, 'd3', 1, 4)


def main():
    opt = parser.parse_args()
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()

    WINNAME = 'Fast Neural Style'

    cv2.namedWindow(winname=WINNAME, flags=cv2.WINDOW_AUTOSIZE)
    status, frame = cap.read()

    params = hkl.load(opt.network)
    params = {k: Variable(torch.from_numpy(v)) for k, v in params.iteritems()}

    tr = tnt.transform.compose([
        lambda x: x.transpose(2, 0, 1).astype(np.float32),
        torch.from_numpy,
        lambda x: x.contiguous().unsqueeze(0),
    ])

    def stylize(frame):
        im = f(Variable(tr(image), volatile=True), params).clamp(0, 255)
        return im[0].data.numpy().transpose(1, 2, 0).astype(np.uint8)

    while True:
        image = T.Scale(opt.resolution)(frame)
        cv2.imshow(winname=WINNAME, mat=stylize(image))
        if cv2.waitKey(delay=30) != 255:
            break
        cap.read(image=frame)

if __name__ == '__main__':
    main()
