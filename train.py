import os
import torch
import torch.nn.functional as func
import torch.nn as nn
from model import MyModel
from config import get_config
from losses import MultiSegmentLoss
from losses import calc_bce_loss
from dataset import (load_video_data, get_video_info, get_video_anno, detection_collate)
from dataset import THUMOSDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def forward_one_epoch(net, clips, targets, device, CPDLoss, scores=None, training=True, ssl=True):
    clips = clips.to(device)
    targets = [t.to(device) for t in targets]

    if training:
        if ssl:
            output_dict = net(clips, proposals=targets, ssl=ssl)
        else:
            output_dict = net(clips, ssl=False)
    else:
        with torch.no_grad():
            output_dict = net(clips)

    if ssl:
        anchor, positive, negative = output_dict
        loss_ = []
        weights = [1, 0.1, 0.1]
        for i in range(3):
            loss_.append(nn.TripletMarginLoss()(anchor[i], positive[i], negative[i]) * weights[i])
        trip_loss = torch.stack(loss_).sum(0)
        return trip_loss
    else:
        loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct = CPDLoss(
            [output_dict['loc'], output_dict['conf'],
             output_dict['prop_loc'], output_dict['prop_conf'],
             output_dict['center'], output_dict['priors'][0]],
            targets)
        # print(output_dict['priors'][0].shape)
        loss_start, loss_end = calc_bce_loss(output_dict['start'], output_dict['end'], scores)
        scores_ = func.interpolate(scores, scale_factor=1.0 / 4)
        # print(scores_.shape)
        # print(output_dict['start_loc_prop'].shape)
        loss_start_loc_prop, loss_end_loc_prop = calc_bce_loss(output_dict['start_loc_prop'],
                                                               output_dict['end_loc_prop'],
                                                               scores_)
        loss_start_conf_prop, loss_end_conf_prop = calc_bce_loss(output_dict['start_conf_prop'],
                                                                 output_dict['end_conf_prop'],
                                                                 scores_)
        loss_start = loss_start + 0.1 * (loss_start_loc_prop + loss_start_conf_prop)
        loss_end = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)
        return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_start, loss_end


def save_model(epoch, model, checkpoint_path):
    torch.save(model.state_dict(),
               os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, config, device, CPDLoss, training=True):
    if training:
        net.train()
    else:
        net.eval()
    loss_loc_val = 0
    loss_conf_val = 0
    loss_prop_l_val = 0
    loss_prop_c_val = 0
    loss_ct_val = 0
    loss_start_val = 0
    loss_end_val = 0
    loss_trip_val = 0
    # loss_contras_val = 0
    cost_val = 0

    pbar = tqdm(data_loader, total=epoch_step_num, ncols=0)
    for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
        loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_start, loss_end = \
            forward_one_epoch(net, clips, targets, device, CPDLoss, scores, training=training, ssl=False)

        loss_l = loss_l * config['training']['lw']
        loss_c = loss_c * config['training']['cw']
        loss_prop_l = loss_prop_l * config['training']['lw']
        loss_prop_c = loss_prop_c * config['training']['cw']
        loss_ct = loss_ct * config['training']['cw']
        cost = loss_l + loss_c + loss_prop_l + loss_prop_c + loss_ct + loss_start + loss_end

        ssl_count = 0
        loss_trip = torch.zeros(1).to(device)
        for i in range(len(flags)):
            if flags[i] and config['training']['ssl'] > 0:
                loss_trip += forward_one_epoch(net, ssl_clips[i].unsqueeze(0), [ssl_targets[i]], device, CPDLoss,
                                               training=training, ssl=True) * config['training']['ssl']
                loss_trip_val += loss_trip.to('cpu').detach().item()
                ssl_count += 1
        if ssl_count:
            loss_trip_val /= ssl_count
            loss_trip /= ssl_count
        cost = cost + loss_trip
        if training:
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        loss_loc_val += loss_l.cpu().detach().item()
        loss_conf_val += loss_c.cpu().detach().item()
        loss_prop_l_val += loss_prop_l.cpu().detach().item()
        loss_prop_c_val += loss_prop_c.cpu().detach().item()
        loss_ct_val += loss_ct.cpu().detach().item()
        loss_start_val += loss_start.cpu().detach().item()
        loss_end_val += loss_end.cpu().detach().item()
        cost_val += cost.cpu().detach().item()
        pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().item())))

    loss_loc_val /= len(data_loader)
    loss_conf_val /= len(data_loader)
    loss_prop_l_val /= len(data_loader)
    loss_prop_c_val /= len(data_loader)
    loss_ct_val /= len(data_loader)
    loss_start_val /= len(data_loader)
    loss_end_val /= len(data_loader)
    loss_trip_val /= len(data_loader)
    cost_val /= len(data_loader)

    if training:
        prefix = 'Train'
        save_model(epoch, net, config['training']['checkpoint_path'])
    else:
        prefix = 'Val'

    plog = 'Epoch-{} {} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}, ' \
           'prop_loc - {:.5f}, prop_conf - {:.5f}, ' \
           'IoU - {:.5f}, start - {:.5f}, end - {:.5f}'.format(epoch, prefix, cost_val, loss_loc_val, loss_conf_val,
                                                               loss_prop_l_val, loss_prop_c_val, loss_ct_val,
                                                               loss_start_val, loss_end_val)
    plog = plog + ', Triplet - {:.5f}'.format(loss_trip_val)
    with open(config['training']['checkpoint_path'] + '.txt', 'a') as f:
        f.write(plog + '\n')
    print(plog)


def main():
    config = get_config()
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    max_epoch = config['training']['max_epoch']
    num_classes = config['dataset']['num_classes']
    checkpoint_path = config['training']['checkpoint_path']
    focal_loss = config['training']['focal_loss']
    # train_state_path = os.path.join(checkpoint_path, 'training')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    conv_channels = 512
    layer_num = 2
    depth = 2
    feat_t = 256 // 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feat_pro = [
        {'num_frames': 64, 'image_size': 6, 'patch_size': 2, 'in_channels': 832},
        {'num_frames': 32, 'image_size': 3, 'patch_size': 1, 'in_channels': 1024}
    ]
    model = MyModel(feat_pro=feat_pro, conv_channels=conv_channels, layer_num=layer_num, depth=depth, feat_t=feat_t,
                    in_channels=config['model']['in_channels'], num_classes=num_classes, frame_num=256,
                    backbone_model=config['model']['backbone_model'])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    piou = config['training']['piou']
    CPDLoss = MultiSegmentLoss(num_classes, piou, 1.0, use_focal_loss=focal_loss)
    train_video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    train_video_annos = get_video_anno(train_video_infos,
                                       config['dataset']['training']['video_anno_path'])
    train_data_dict = load_video_data(train_video_infos,
                                      config['dataset']['training']['video_data_path'])
    train_dataset = THUMOSDataset(train_data_dict,
                                  train_video_infos,
                                  train_video_annos,
                                  clip_length=config['dataset']['training']['clip_length'],
                                  crop_size=config['dataset']['training']['crop_size'],
                                  stride=config['dataset']['training']['clip_stride']
                                  )
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4,
                                   collate_fn=detection_collate, pin_memory=True, drop_last=True)
    epoch_step_num = len(train_dataset) // batch_size

    for i in range(0, max_epoch + 1):
        run_one_epoch(i, model, optimizer, train_data_loader, epoch_step_num, config, device, CPDLoss)


if __name__ == '__main__':
    main()
