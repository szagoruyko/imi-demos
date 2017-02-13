from PIL import Image
import torch
import torchvision.transforms as transforms
import hickle as hkl
from torch.autograd import Variable
import torch.nn.functional as F
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
import os
import argparse

# telegram bot that takes a document with an image
# downloads it and passes through Wide ResNet trained
# for classification on ImageNet
# sends back a message with top5 predictions

parser = argparse.ArgumentParser()
parser.add_argument('--token', required=True)
opt = parser.parse_args()


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


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

params = hkl.load('./nin-export.hkl')
params = {k: Variable(torch.from_numpy(v)) for k, v in params.iteritems()}


def classify(image_path):
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).cuda()

    with open('./synset_words.txt') as f:
        synset_words = [s[10:-1] for s in f.readlines()]

    predictions = F.softmax(f(Variable(tensor.unsqueeze(0), volatile=True), params))
    probs, idx = predictions.data.view(-1).topk(k=5, dim=0, sorted=True)
    return '\n'.join(['%.2f: %s' % (p, synset_words[i]) for p, i in zip(probs, idx)])

def start(bot, update):
    update.message.reply_text('Hi!')

def help(bot, update):
    update.message.reply_text('Help!')

def echo(bot, update):
    if update.message.document:
        doc = update.message.document
        tmpname = os.path.join('/tmp', doc.file_name)
        bot.getFile(update.message.document.file_id).download(tmpname)
        result = classify(tmpname)
        print result
        update.message.reply_text(result)
    else:
        update.message.reply_text(update.message.text)

def error(bot, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))

def main():
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(opt.token)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    dp.add_handler(MessageHandler(Filters.document, echo))

    dp.add_error_handler(error)

    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    main()
