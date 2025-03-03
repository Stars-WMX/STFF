import math
import yaml
import argparse
import torch
import torch.optim as optim
import os.path as op
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
import dataset
from collections import OrderedDict
from model.STFF_L import STFF_L


def receive_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, default='train.yml', help='Path to option YAML file.')
    parser.add_argument('--local_rank', type=int, default=0, help='Distributed launcher requires.')
    args = parser.parse_args()

    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank

    if opts_dict['train']['exp_name'] is None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join("exp", opts_dict['train']['exp_name'], "log.log")
    opts_dict['train']['checkpoint_save_path_pre'] = op.join("exp", opts_dict['train']['exp_name'], "ckp_")

    opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = True
    else:
        opts_dict['train']['is_dist'] = False

    return opts_dict

def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    rank = opts_dict['train']['rank']
    unit = opts_dict['train']['criterion']['unit']
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_val = int(opts_dict['train']['interval_val'])

    # ==========
    # init distributed training
    # ==========
    if opts_dict['train']['is_dist']:
        utils.init_dist(local_rank=rank, backend='nccl')
    pass

    if rank == 0:
        log_dir = op.join("exp", opts_dict['train']['exp_name'])
        print("log_dir", log_dir)
        utils.mkdir(log_dir)
        log_fp = open(opts_dict['train']['log_path'], 'w')

        # log all parameters
        msg = (
                f"{'<' * 10} Hello {'>' * 10}\n"
                f"Timestamp: [{utils.get_timestr()}]\n"
                f"\n{'<' * 10} Options {'>' * 10}\n"
                f"{utils.dict2str(opts_dict)}"
                )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # TO-DO: init tensorboard
    # ==========
    pass

    seed = opts_dict['train']['random_seed']
    utils.set_random_seed(seed + rank)

    torch.backends.cudnn.benchmark = True  # speed up
    # torch.backends.cudnn.deterministic = True  # if reproduce


    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']

    radius = opts_dict['network']['radius']
    assert train_ds_type in dataset.__all__, "Not implemented!"

    train_ds_cls = getattr(dataset, train_ds_type)

    train_ds = train_ds_cls(opts_dict=opts_dict['dataset']['train'], radius=radius)

    # create datasamplers
    train_sampler = utils.DistSampler(
                                        dataset=train_ds,
                                        num_replicas=opts_dict['train']['num_gpu'],
                                        rank=rank,
                                        ratio=opts_dict['dataset']['train']['enlarge_ratio']
                                     )

    # create train dataloaders
    train_loader = utils.create_dataloader(
                                            dataset=train_ds,
                                            opts_dict=opts_dict,
                                            sampler=train_sampler,
                                            phase='train',
                                            seed=opts_dict['train']['random_seed']
                                          )
    assert train_loader is not None

    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu'] * opts_dict['train']['num_gpu']  # divided by all GPUs
    num_iter_per_epoch = math.ceil(len(train_ds) * opts_dict['dataset']['train']['enlarge_ratio'] / batch_size)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)


    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)


    # ==========
    # create model    ,find_unused_parameters=True
    # ==========
    model = STFF_L()
    model = model.to(rank)
    if opts_dict['train']['is_dist']:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # # # # load pre-trained generator
    # ckp_path = opts_dict['train']['load_path']
    # checkpoint = torch.load(ckp_path)
    # state_dict = checkpoint['state_dict']
    #
    # if ('module.' in list(state_dict.keys())[0]) and (not opts_dict['train']['is_dist']):  # multi-gpu pre-trained -> single-gpu training
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:]  # remove module
    #         new_state_dict[name] = v
    #     model.load_state_dict(new_state_dict)
    #     print(f'loaded from1 {ckp_path}')
    # elif ('module.' not in list(state_dict.keys())[0]) and (opts_dict['train']['is_dist']):  # single-gpu pre-trained -> multi-gpu training
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = 'module.' + k  # add module
    #         new_state_dict[name] = v
    #     model.load_state_dict(new_state_dict)
    #     print(f'loaded from2 {ckp_path}')
    # else:  # the same way of training  ,strict=False
    #     model.load_state_dict(state_dict)
    #     print(f'loaded from3 {ckp_path}')

    # ==========
    # define loss func & optimizer & scheduler & scheduler & criterion 损失函数！！！！！！！
    # ==========
    assert opts_dict['train']['loss'].pop('type') == 'CVQE_Loss', "Not implemented."
    loss_func = utils.CVQE_Loss()


    # define optimizer
    assert opts_dict['train']['optim'].pop('type') == 'Adam', "Not implemented."
    optimizer = optim.Adam(model.parameters(), **opts_dict['train']['optim'])

    # define scheduler
    if opts_dict['train']['scheduler']['is_on']:
        assert opts_dict['train']['scheduler'].pop('type') == 'CosineAnnealingRestartLR', "Not implemented."
        del opts_dict['train']['scheduler']['is_on']
        scheduler = utils.CosineAnnealingRestartLR(optimizer, **opts_dict['train']['scheduler'])
        opts_dict['train']['scheduler']['is_on'] = True

    start_iter = opts_dict['train']['start_iter']  # should be restored
    start_epoch = start_iter // num_iter_per_epoch

    # display and log
    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"start from iter: [{start_iter}]\n"
            f"start from epoch: [{start_epoch}]"
        )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    if opts_dict['train']['is_dist']:
        torch.distributed.barrier()  # all processes wait for ending

    if rank == 0:
        msg = f"\n{'<' * 10} Training {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        # create timer
        total_train_timer = utils.system.Timer()  # total time of each epoch

    # Create a Timer object before training starts
    training_timer = utils.system.Timer()

    # ==========
    # start training
    # ==========
    model.train()
    num_iter_accum = start_iter

    for current_epoch in range(start_epoch, num_epoch + 1):
        if opts_dict['train']['is_dist']:
            train_sampler.set_epoch(current_epoch)

        # fetch the first batch
        tra_prefetcher.reset()
        train_data = tra_prefetcher.next()

        while train_data is not None:
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break

            # Get the current time before the start of each iteration
            training_timer.restart()

            # get data
            gt_data = train_data['gt'].to(rank)  # (B T C H W)
            lq_data = train_data['lq'].to(rank)  # (B T C H W)s
            b, t, c, _, _ = lq_data.shape
            lq_data = torch.cat([lq_data[:, :, i, ...] for i in range(c)],dim=1)  # B T*C H W

            enhanced = model(lq_data)
            loss = loss_func(enhanced, gt_data)

            optimizer.zero_grad()  # zero grad
            loss.backward()  # cal grad
            optimizer.step()  # update parameters

            # update learning rate
            if opts_dict['train']['scheduler']['is_on']:
                scheduler.step()  # should after optimizer.step()

            if (num_iter_accum % interval_print == 0) and (rank == 0):
                lr = optimizer.param_groups[0]['lr']
                if num_iter_accum == 150000:
                    optimizer.param_groups[0]["lr"] = 5e-5
                elif num_iter_accum == 300000:
                    optimizer.param_groups[0]["lr"] = 2e-5
                elif num_iter_accum == 450000:
                    optimizer.param_groups[0]["lr"] = 1e-5

                loss_item = loss.item()

                # Get the training time for the current iteration
                iteration_time = training_timer.get_interval()
                # Estimated training time for the remaining iterations
                remaining_time = (num_iter - num_iter_accum) * iteration_time

                msg = (
                    f'iterator: [{num_iter_accum}]/{num_iter}, '
                    f'epoch: [{current_epoch}]/{num_epoch - 1}, '
                    f'lr: [{lr * 1e4:.3f}]x1e-4, loss: [{loss_item:.4f}], '
                    f'eta: [{remaining_time / (24 * 3600):.2f} days, '
                    f'{remaining_time / 3600:.4f} h], '
                    f'iteration time: [{iteration_time:.4f}] s'
                )

                print(msg)
                log_fp.write(msg + '\n')

            if ((num_iter_accum % interval_val == 0) or (num_iter_accum == num_iter)) and (rank == 0):
                # save model
                checkpoint_save_path = (f"{opts_dict['train']['checkpoint_save_path_pre']}"
                                        f"{num_iter_accum}"
                                        ".pth")
                state = {'num_iter_accum': num_iter_accum,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),}
                if opts_dict['train']['scheduler']['is_on']:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, checkpoint_save_path)

                # log
                msg = "> model saved at {:s}\n".format(checkpoint_save_path)
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            if opts_dict['train']['is_dist']:
                torch.distributed.barrier()  # all processes wait for ending

            # fetch next batch
            train_data = tra_prefetcher.next()

    if rank == 0:
        total_time = total_train_timer.get_interval() / 3600
        total_day = total_train_timer.get_interval() / (24 * 3600)

        msg_hours = "TOTAL TIME: [{:.4f}] h".format(total_time)
        msg_days = "TOTAL TIME: [{:.4f}] 天".format(total_day)

        print(msg_hours)
        print(msg_days)
        log_fp.write(msg_hours + '\n')
        log_fp.write(msg_days + '\n')

        goodbye_msg = (f"\n{'<' * 10} Goodbye {'>' * 10}\n"
                       f"Timestamp: [{utils.get_timestr()}]")
        print(goodbye_msg)
        log_fp.write(goodbye_msg + '\n')

        log_fp.close()


if __name__ == '__main__':
    main()
