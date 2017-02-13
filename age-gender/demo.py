import numpy as np
import cv2
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.serialization.read_lua_file import load_lua

age_gender_dir = '/Users/szagoruyko/research/rocks/torch-opencv-demos/age_gender/'
detector_params_file = '/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
M = 227
fx = 0.5
ages = ['0-2','4-6','8-13','15-20','25-32','38-43','48-53','60-']


def main():
    age_net = load_lua(os.path.join(age_gender_dir, 'age_net.t7'))
    gender_net = load_lua(os.path.join(age_gender_dir, 'gender_net.t7'))
    age_net.get(11).size = torch.Size((1,-1))
    gender_net.get(11).size = torch.Size((1,-1))
    img_mean = load_lua(os.path.join(age_gender_dir, 'age_gender_mean.t7')).numpy().transpose(1,2,0)

    cascade = cv2.CascadeClassifier(detector_params_file)

    cap = cv2.VideoCapture(0)

    ok, frame = cap.read()
    assert ok, 'couldnt read frame'

    while True:
        w,h,_ = frame.shape

        im2 = cv2.resize(frame, (0,0), fx=fx, fy=fx)
        faces = cascade.detectMultiScale(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))
        for f in faces:
            x, y, w, h = (f/fx).astype(np.int32)
            crop = cv2.getRectSubPix(frame, patchSize=(w,h), center=(x+w//2, y+h//2))
            im = cv2.resize(crop, (256,256)).astype(np.float32)
            I = cv2.resize(im - img_mean, (M,M)).transpose(2,0,1)
            inputs = torch.from_numpy(I).float().unsqueeze(0)

            gender_out = gender_net.forward(inputs).view(-1)
            gender = 'M' if gender_out[0] > gender_out[1] else 'F'

            age_out = age_net.forward(inputs).view(-1)
            age = ages[age_out.max(0)[1][0]]

            cv2.putText(frame, '%s: %s' % (gender, age), org=(x,y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1, color=(255,255,0), thickness=1)
            cv2.rectangle(frame, pt1=(x,y+3), pt2=(x+w, y+h), color=(30,255,30))
        cv2.imshow('PyTorch OpenCV age&gender demo', frame)
        ok = cap.read(image=frame)
        if cv2.waitKey(1) != 255:
            break

if __name__ == '__main__':
    main()
