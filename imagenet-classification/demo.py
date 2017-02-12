import cv2
import hickle as hkl
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.cvtransforms as T
import torchnet as tnt

cap = cv2.VideoCapture(0)
assert cap.isOpened()

WINNAME = 'torch-OpenCV ImageNet classification demo'

cv2.namedWindow(winname=WINNAME, flags=cv2.WINDOW_AUTOSIZE)
status, frame = cap.read()

params = hkl.load('./nin-export.hkl')
params = {k: Variable(torch.from_numpy(v)) for k, v in params.iteritems()}


def f(inputs, params):
    def conv2d(x, params, name, stride=1, padding=0):
        return F.conv2d(x,
                        params['%s.weight' % name],
                        params['%s.bias' % name], stride, padding)

    def block(x, names, stride, padding):
        x = F.relu(conv2d(x, params, names[0], stride, padding))
        x = F.relu(conv2d(x, params, names[1]))
        x = F.relu(conv2d(x, params, names[2]))
        return x

    o = block(inputs, ['conv0', 'conv1', 'conv2'], 4, 5)
    o = F.max_pool2d(o, 2)
    o = block(o, ['conv3', 'conv4', 'conv5'], 1, 2)
    o = F.max_pool2d(o, 2)
    o = block(o, ['conv6', 'conv7', 'conv8'], 1, 1)
    o = F.max_pool2d(o, 2)
    o = block(o, ['conv9', 'conv10', 'conv11'], 1, 1)
    o = F.avg_pool2d(o, 7)
    o = o.view(o.size(0), -1)
    o = F.linear(o, params['fc.weight'], params['fc.bias'])
    return o


tr = tnt.transform.compose([
    T.Scale(256),
    T.CenterCrop(224),
    lambda x: x.astype(np.float32) / 255.0,
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    lambda x: x.transpose(2, 0, 1).astype(np.float32),
    torch.from_numpy,
])

with open('synset_words.txt') as h:
    synset_words = [s[10:-1] for s in h.readlines()]


def classify(tensor):
    predictions = F.softmax(f(Variable(tensor.unsqueeze(0), volatile=True), params))
    probs, idx = predictions.data.view(-1).topk(k=5, dim=0, sorted=True)
    return ['%.2f: %s' % (p, synset_words[i]) for p, i in zip(probs, idx)]

while True:
    image = T.Scale(640)(frame)
    predictions = classify(tr(frame))
    for i, line in enumerate(predictions):
        cv2.putText(img=image, text=line, org=(20, 20 + i * 25),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.5, color=(205, 0, 0), thickness=1)

    cv2.imshow(winname=WINNAME, mat=image)
    if cv2.waitKey(delay=30) != 255:
        break
    cap.read(image=frame)
