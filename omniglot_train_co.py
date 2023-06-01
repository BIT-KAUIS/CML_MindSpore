# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import os
import numpy as np
from src.OmniglotIter import IterDatasetGenerator
from src.meta_co import Metaco
import mindspore.context as context
import mindspore.dataset as ds
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import load_checkpoint

def main(args):
    np.random.seed(222)
    print(args)
    config = [
        ('conv2d', [1, 64, 3, 3, 2, 1, 'pad']),
        ('bn', [64]),
        ('relu', [True]),
        ('conv2d', [64, 64, 3, 3, 2, 1, 'pad']),
        ('bn', [64]),
        ('relu', [True]),
        ('conv2d', [64, 64, 3, 3, 2, 1, 'pad']),
        ('bn', [64]),
        ('relu', [True]),
        ('conv2d', [64, 64, 3, 3, 2, 1, 'pad']),
        ('bn', [64]),
        ('relu', [True]),
        ('reduce_mean', []),
        # ('flatten', []),
        ('linear', [64, args.n_way])
    ]
    if args.ckpt != '':
        param_dict = load_checkpoint(args.ckpt)
        maml1 = Metaco(args, config, param_dict)
    else:
        maml1 = Metaco(args, config)
        maml2 = Metaco(args, config)

    context.set_context(device_id=args.device_id)

    db_train1 = IterDatasetGenerator(args.data_path,
                                    batchsz=args.task_num,
                                    n_way=args.n_way,
                                    k_shot=args.k_spt,
                                    k_query=args.k_qry,
                                    imgsz=args.imgsz,
                                    itera=1)
    db_train2 = IterDatasetGenerator(args.data_path,
                                    batchsz=args.task_num,
                                    n_way=args.n_way,
                                    k_shot=args.k_spt,
                                    k_query=args.k_qry,
                                    imgsz=args.imgsz,
                                    itera=1)
    config = CheckpointConfig(save_checkpoint_steps=1,
                              keep_checkpoint_max=2000,
                              saved_network=maml1)
    ckpoint_cb = ModelCheckpoint(prefix='maml1', directory=args.output_dir, config=config)

    inp1 = ds.GeneratorDataset(db_train1, ['x_spt', 'y_spt', 'x_qry', 'y_qry'])
    inp2 = ds.GeneratorDataset(db_train2, ['x_spt', 'y_spt', 'x_qry', 'y_qry'])

    print("data ready")

    maml1.set_grad(True)
    maml2.set_grad(True)
    model1 = Model(maml1)
    model2 = Model(maml2)

    # 创建空的数组来存放损失值
    loss_list1 = []
    loss_list2 = []
    loss1_show = []
    loss2_show = []
    epoches_show = []
    root = os.getcwd()
    path = os.path.join(root, "result")

    # 定义回调函数，将损失值添加到数组中，并打印出来
    class MyLossMonitor1(LossMonitor):
        def step_end(self, run_context):
            cb_params = run_context.original_args()
            outputs = np.asarray(cb_params.net_outputs)
            loss_list1.append(outputs)
            print("epoch1: {}, step1: {}, outputs1 are {}".format(cb_params.cur_epoch_num,cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)))
            # print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num,cb_params.cur_step_num,
            #                                                    str(cb_params.net_outputs)))

    class MyLossMonitor2(LossMonitor):
        def step_end(self, run_context):
            cb_params = run_context.original_args()
            outputs = np.asarray(cb_params.net_outputs)
            loss_list2.append(outputs)
            print("epoch2: {}, step2: {}, outputs2 are {}".format(cb_params.cur_epoch_num,cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)))
            # print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num,cb_params.cur_step_num,
            #                                                    str(cb_params.net_outputs)))


    print("="*20)
    print("start training")
    for epoch_iter in range(args.epoch):  # 外循环
        model1.train(5, inp1, callbacks=[TimeMonitor(), MyLossMonitor1(), ckpoint_cb], dataset_sink_mode=True, sink_size = 5)
        model2.train(5, inp2, callbacks=[TimeMonitor(), MyLossMonitor2(), ckpoint_cb], dataset_sink_mode=True, sink_size = 5)

        # Competition Update
        if loss_list1[-1][0] < loss_list2[-1][0]:  # 如果1的训练结果比2好，指标是训练损失 , [0]-loss, [1]-acc
            for p1, p2 in zip(model1._network.outer_params, model2._network.outer_params):  # 额外更新2
                p2.set_data(p2.data + args.lr_co * (p1.data - p2.data))
        elif loss_list2[-1][0] < loss_list1[-1][0] :  # 如果1的训练结果比2差
            for p1, p2 in zip(model1._network.outer_params, model2._network.outer_params):   # 额外更新1
                p1.set_data(p1.data + args.lr_co * (p2.data - p1.data))

        loss1_show.append(loss_list1[-1][0])
        loss2_show.append(loss_list2[-1][0])
        epoches_show.append(epoch_iter)
        print("epoch: {}, loss {}".format(epoch_iter,loss_list1[-1][0]))

    print("end")
    np.savetxt('%s/train_epoch.txt' % path, epoches_show, fmt="%.6f")  # 保存周期数据
    np.savetxt('%s/train_loss1.txt' % path, loss1_show, fmt="%.6f")  # 保存训练数据
    np.savetxt('%s/train_loss2.txt' % path, loss2_show, fmt="%.6f")  # 保存训练数据

    print("done")
    print("=" * 20)







if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device_id', type=int, help='device id', default=1)
    argparser.add_argument('--device_target', type=str, help='device target', default='GPU')
    argparser.add_argument('--mode', type=str, help='pynative or graph', default='graph')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=200)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4) # default
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--lr_scheduler_gamma', type=float, help='update steps for finetunning', default=0.5)
    argparser.add_argument('--output_dir', type=str, help='update steps for finetunning', default='./ckpt_outputs')
    argparser.add_argument('--ckpt', type=str, help='trained model', default='')
    argparser.add_argument('--data_path', type=str, help='path of data', default='/your/path/omniglot/')
    argparser.add_argument('--lr_co', type=int, help='competitive learning rate', default=0.5)
    arg = argparser.parse_args()
    if arg.mode == 'pynative':
        context.set_context(mode=context.PYNATIVE_MODE)
    elif arg.mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE)
    if os.path.exists('loss.txt'):
        os.remove('loss.txt')
    if not os.path.exists(arg.output_dir):
        os.makedirs(arg.output_dir)
    main(arg)
