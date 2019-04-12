from __future__ import division

import numpy as np
import torch
from torchvision import transforms

from model_mtl import MTL
from model_kgm import KGM
from dataloader_mtl import ArtDatasetMTL
from dataloader_kgm import ArtDatasetKGM
from attributes import load_att_class


def test_knowledgegraph(args_dict):

    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    if args_dict.att == 'type':
        att2i = type2idx
    elif args_dict.att == 'school':
        att2i = school2idx
    elif args_dict.att == 'time':
        att2i = time2idx
    elif args_dict.att == 'author':
        att2i = author2idx

    model = KGM(len(att2i))
    if args_dict.use_gpu:
        model.cuda()

    # Load best model
    print("=> loading checkpoint '{}'".format(args_dict.model_path))
    checkpoint = torch.load(args_dict.model_path)
    args_dict.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args_dict.model_path, checkpoint['epoch']))

    # Data transformation for test
    test_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    # Data Loaders for test
    test_loader = torch.utils.data.DataLoader(
        ArtDatasetKGM(args_dict, set='test', att2i=att2i, att_name=args_dict.att, transform=test_transforms),
        batch_size=args_dict.batch_size, shuffle=False, pin_memory=(not args_dict.no_cuda), num_workers=args_dict.workers)

    # Switch to evaluation mode & compute test samples embeddings
    model.eval()
    for i, (input, target) in enumerate(test_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        with torch.no_grad():
            output = model(input_var[0])
            outsoftmax = torch.nn.functional.softmax(output[0])
        conf, predicted = torch.max(outsoftmax, 1)

        # Store embeddings
        if i==0:
            out = predicted.data.cpu().numpy()
            label = target[0].cpu().numpy()
            scores = conf.data.cpu().numpy()
        else:
            out = np.concatenate((out,predicted.data.cpu().numpy()),axis=0)
            label = np.concatenate((label,target[0].cpu().numpy()),axis=0)
            scores = np.concatenate((scores, conf.data.cpu().numpy()),axis=0)

    # Compute Accuracy
    acc = np.sum(out == label)/len(out)
    print('Model %s\tTest Accuracy %.03f' % (args_dict.model_path, acc))


def test_multitask(args_dict):

    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
    att2i = [type2idx, school2idx, time2idx, author2idx]

    model = MTL(num_classes)
    if args_dict.use_gpu:
        model.cuda()

    # Load best model
    print("=> loading checkpoint '{}'".format(args_dict.model_path))
    checkpoint = torch.load(args_dict.model_path)
    args_dict.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args_dict.model_path, checkpoint['epoch']))

    # Data transformation for test
    test_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    # Data Loaders for test
    test_loader = torch.utils.data.DataLoader(
        ArtDatasetMTL(args_dict, set = 'test', att2i=att2i, transform = test_transforms),
        batch_size=args_dict.batch_size, shuffle=False, pin_memory=(not args_dict.no_cuda), num_workers=args_dict.workers)

    # Switch to evaluation mode & compute test
    model.eval()
    for i, (input, target) in enumerate(test_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        with torch.no_grad():
            output = model(input_var[0])
        _, pred_type = torch.max(output[0], 1)
        _, pred_school = torch.max(output[1], 1)
        _, pred_time = torch.max(output[2], 1)
        _, pred_author = torch.max(output[3], 1)

        # Store outputs
        if i==0:
            out_type = pred_type.data.cpu().numpy()
            out_school = pred_school.data.cpu().numpy()
            out_time = pred_time.data.cpu().numpy()
            out_author = pred_author.data.cpu().numpy()
            label_type = target[0].cpu().numpy()
            label_school = target[1].cpu().numpy()
            label_time = target[2].cpu().numpy()
            label_author = target[3].cpu().numpy()
        else:
            out_type = np.concatenate((out_type,pred_type.data.cpu().numpy()),axis=0)
            out_school = np.concatenate((out_school, pred_school.data.cpu().numpy()), axis=0)
            out_time = np.concatenate((out_time, pred_time.data.cpu().numpy()), axis=0)
            out_author = np.concatenate((out_author, pred_author.data.cpu().numpy()), axis=0)
            label_type = np.concatenate((label_type,target[0].cpu().numpy()),axis=0)
            label_school = np.concatenate((label_school,target[1].cpu().numpy()),axis=0)
            label_time = np.concatenate((label_time,target[2].cpu().numpy()),axis=0)
            label_author = np.concatenate((label_author,target[3].cpu().numpy()),axis=0)

    # Compute Accuracy
    acc_type = np.sum(out_type == label_type)/len(out_type)
    acc_school = np.sum(out_school == label_school) / len(out_school)
    acc_tf = np.sum(out_time == label_time) / len(out_time)
    acc_author = np.sum(out_author == label_author) / len(out_author)

    # Print test accuracy
    print('------------ Test Accuracy -------------')
    print('Type Accuracy %.03f' % acc_type)
    print('School Accuracy %.03f' % acc_school)
    print('Timeframe Accuracy %.03f' % acc_tf)
    print('Author Accuracy %.03f' % acc_author)
    print('----------------------------------------')


def run_test(args_dict):

    if args_dict.model == 'mtl':
        test_multitask(args_dict)
    elif args_dict.model == 'kgm':
        test_knowledgegraph(args_dict)
    else:
        assert False, 'Incorrect model type'

