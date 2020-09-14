from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss.CrossTriplet import CrossTriplet as TripletLoss
from reid.loss.triplet import TripletLoss as TripletLoss_s
from reid.trainers_two import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler_cycle import CamRandomIdentitySampler as RandomIdentitySampler
from reid.utils.data.sampler_cycle import CamSampler,IDRandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, save_checkpoint_s
from utlis import RandomErasing, WarmupMultiStepLR,CrossEntropyLabelSmooth

from reid.models.networks import AdaINGen, MsImageDis
from reid.models.reIDmodel import ft_net, ft_netAB, PCB



def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval, flip_prob, padding, re_prob):
    root = osp.join(data_dir, name)
    print(root)
    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    trainvallabel = dataset.trainvallabel
    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)


    train_transformer = T.Compose([
        T.Resize((height, width)),
        # T.RandomHorizontalFlip(),
        T.Pad(padding),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        RandomErasing(probability=re_prob, mean=[0.485, 0.456, 0.406])
        ])
    test_transformer = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalizer,
    ])

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=32, num_workers=workers,
        shuffle=False, pin_memory=True)

    query_loader = DataLoader(
        Preprocessor(list(set(dataset.query)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=32, num_workers=workers,
        sampler=CamSampler(list(set(dataset.query)), [2,5]),
        shuffle=False, pin_memory=True)

    query_loader_s = DataLoader(
        Preprocessor(list(set(dataset.query)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=32, num_workers=workers,
        sampler=CamSampler(list(set(dataset.query)), [0,1,3,4]),
        shuffle=False, pin_memory=True)

    gallery_loader_s = DataLoader(
        Preprocessor(list(set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=32, num_workers=workers,
        sampler=CamSampler(list(set(dataset.gallery)), [0, 1, 3, 4], 4),
        shuffle=False, pin_memory=True)


    gallery_loader = DataLoader(
        Preprocessor(list(set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=32, num_workers=workers,
        sampler=CamSampler(list(set(dataset.gallery)), [0,1,3,4], 4),
        shuffle=False, pin_memory=True)

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    return dataset, num_classes, train_loader, trainvallabel, val_loader, query_loader, gallery_loader, query_loader_s, gallery_loader_s


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir+'/log'))

    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else (256, 128)
    dataset, num_classes, train_loader, trainvallabel, val_loader, query_loader, gallery_loader, query_loader_s, gallery_loader_s = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
            args.width, args.batch_size, args.num_instances, args.workers,
            args.combine_trainval, args.flip_prob, args.padding, args.re_prob)
    print(num_classes)
    gen_a = AdaINGen(3, num_classes)
    gen_b = AdaINGen(3, num_classes)
    # id_a = ft_netAB(num_classes, stride=1, norm="no",  pool="max")
    # id_b = ft_netAB(num_classes, stride=1, norm="no",  pool="max")
    dis_a = MsImageDis(3)  # discriminator for domain a
    dis_b = MsImageDis(3)  # discriminator for domain a
    gen_a = gen_a.cuda()
    gen_b = gen_b.cuda()
    # id_a = id_a.cuda()
    # id_b = id_b.cuda()
    dis_a = dis_a.cuda()
    dis_b = dis_b.cuda()

    evaluator = Evaluator(gen_a.enc_content, gen_b.enc_content)
    metric = DistanceMetric(algorithm=args.dist_metric)

    # evaluator_s = Evaluator(model_s)
    # metric_s = DistanceMetric(algorithm=args.dist_metric)

    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model_t.load_state_dict(checkpoint['model'])
        # model_discriminator.load_state_dict(checkpoint['model_discriminator'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {}".format(start_epoch))

    if args.evaluate:
        metric.train(model_t, train_loader)
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
        exit()

    current_margin = args.margin
    criterion_z_s = CrossEntropyLabelSmooth(num_classes = num_classes, epsilon=0.5).cuda()
    criterion_att = nn.MSELoss().cuda()
    criterion_z = CrossEntropyLabelSmooth(num_classes = num_classes, epsilon=0.5).cuda()
    criterion_I = TripletLoss(margin = current_margin).cuda()
    criterion_I_s = TripletLoss_s(margin=current_margin).cuda()
    # criterion_D = nn.CrossEntropyLoss().cuda()

    # print(args)
    # # setup id opt
    # # if args.arch == 'ide':
    # #     ignored_params = list(map(id, model_t.model.fc.parameters() )) + list(map(id, model_t.classifier.parameters() ))
    # # else:
    # #     ignored_params = list(map(id, id_a.classifier1.parameters())) + list(map(id, id_a.classifier2.parameters()))
    # #     ignored_params_s = list(map(id, id_b.classifier1.parameters())) + list(map(id, id_b.classifier2.parameters()))
    # #
    # # base_params = filter(lambda p: id(p) not in ignored_params, id_a.parameters())
    # # base_params_s = filter(lambda p: id(p) not in ignored_params_s, id_b.parameters())
    # #
    # # if args.use_adam:
    # #     optimizer_ft = torch.optim.Adam([
    # #         {'params': filter(lambda p: p.requires_grad,base_params), 'lr': args.lr},
    # #         {'params': filter(lambda p: p.requires_grad, base_params_s), 'lr': args.lr},
    # #         {'params': id_a.classifier1.parameters(), 'lr': args.lr},
    # #         {'params': id_a.classifier2.parameters(), 'lr': args.lr},
    # #         {'params': id_b.classifier1.parameters(), 'lr': args.lr},
    # #         {'params': id_b.classifier2.parameters(), 'lr': args.lr},
    # #         ],
    # #         weight_decay=5e-4)
    # # else:
    # #     optimizer_ft = torch.optim.SGD([
    # #         {'params': filter(lambda p: p.requires_grad, base_params), 'lr': args.lr},
    # #         {'params': filter(lambda p: p.requires_grad, base_params_s), 'lr': args.lr},
    # #         {'params': model_s.classifier.parameters(), 'lr': args.lr},
    # #         {'params': model_s.attention_module.parameters(), 'lr': args.lr},
    # #         {'params': model_t.classifier.parameters(), 'lr': args.lr},
    # #         {'params': model_t.attention_module.parameters(), 'lr': args.lr},
    # #         ],
    # #         momentum=0.9,
    # #         weight_decay=5e-4,
    # #         nesterov=True)
    # #
    # # id_scheduler = WarmupMultiStepLR(optimizer_ft, args.mile_stone, args.gamma, args.warmup_factor,
    # #                                       args.warmup_iters, args.warmup_methods)
    # setup dis and gen
    dis_opt = torch.optim.Adam([
            {'params': dis_a.parameters(), 'lr': args.dis_lr},
            {'params': dis_b.parameters(), 'lr': args.dis_lr},
            ],
            weight_decay=5e-4)
    gen_opt = torch.optim.Adam([
            {'params': gen_a.parameters(), 'lr': args.gen_lr},
            {'params': gen_b.parameters(), 'lr': args.gen_lr},
            ],
            weight_decay=5e-4)
    dis_scheduler = WarmupMultiStepLR(dis_opt, args.mile_stone, args.gamma, args.warmup_factor,
                                     args.warmup_iters, args.warmup_methods)
    gen_scheduler = WarmupMultiStepLR(gen_opt, args.mile_stone, args.gamma, args.warmup_factor,
                                      args.warmup_iters, args.warmup_methods)
    trainer = Trainer(gen_a, gen_b, dis_a, dis_b, criterion_z, criterion_I, criterion_att, trainvallabel, 1, 1 ,0.15 , 0.05, 5)

    flag = 1
    best_top1 = -1
    best_top1_s = -1
    # Start training
    for epoch in range(start_epoch, args.epochs):
        print("Begin Train")
        # id_scheduler.step()
        gen_scheduler.step()
        dis_scheduler.step()
        trainer.train(epoch, train_loader, dis_opt, gen_opt)
    #
        save_checkpoint({
            'content_a': gen_a.enc_content.state_dict(),
            'content_b': gen_b.enc_content.state_dict(),
            'style_a': gen_a.enc_style.state_dict(),
            'style_b': gen_b.enc_style.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, False, epoch, args.logs_dir, fpath='checkpoint.pth.tar')

        # if epoch < 200:
        #     continue
        if not epoch % 1 ==0:
            continue


        top1 = evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, metric)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'content_a': gen_a.enc_content.state_dict(),
            'content_b': gen_b.enc_content.state_dict(),
            'style_a': gen_a.enc_style.state_dict(),
            'style_b': gen_b.enc_style.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, epoch, args.logs_dir, fpath='checkpoint.pth.tar')
    #
    #
    #
    #
    # print('Test with best model_t:')
    # print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
    #           format(epoch, top1, best_top1, ' *' if is_best else ''))
    #
    # print('Test with best model_s:')
    # print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
    #       format(epoch, top1_s, best_top1_s, ' *' if is_best else ''))
    #
    # checkpoint = load_checkpoint(osp.join(args.logs_dir,'model_best.pth.tar'))
    # model_t.load_state_dict(checkpoint['model'])
    # metric.train(model_t, train_loader)
    # evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, metric)
    #
    # checkpoint_s = load_checkpoint(osp.join(args.logs_dir, 's_model_best.pth.tar'))
    # model_s.load_state_dict(checkpoint_s['model'])
    # evaluator_s.evaluate(query_loader_s, gallery_loader_s, dataset.query, dataset.gallery, metric)

    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross_modality for Person Re-identification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)


    parser.add_argument('--height', type=int, default=312,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default= 156,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")

    # transformer
    parser.add_argument('--flip_prob', type=float, default=0.5)
    parser.add_argument('--re_prob', type=float, default=0.0)
    parser.add_argument('--padding', type=int, default=0)

    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4,
                         help="learning rate of all parameters")
    parser.add_argument('--dis_lr', type=float, default=1e-4,
                        help="learning rate of all parameters")
    parser.add_argument('--gen_lr', type=float, default=1e-4,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--use_adam', action='store_true',
                    help="use Adam as the optimizer, elsewise SGD ")
    parser.add_argument('--gamma', type=float, default = 0.1,
                    help="gamma for learning rate decay")

    #
    parser.add_argument('--mile_stone', type=list, default=[600,750])

    # parser.add_argument('--mile_stone', type=list, default=[600,750])

    parser.add_argument('--warmup_iters', type=int, default = 100)
    parser.add_argument('--warmup_methods', type=str, default = 'linear', choices=('linear', 'constant'))
    parser.add_argument('--warmup_factor', type=float, default = 0.01 )

    # att_mode
    parser.add_argument('--att_mode', type=int, default=1)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    main(parser.parse_args())
