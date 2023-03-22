# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
# from models.swin_transformer import *
from models.experimental import *
# from models.deeplab import DeepLab
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
# from models import build_model
try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
import torch.nn.functional as F

class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml',input_mode='RGB',ch_steam=3, ch=3, nc=None, anchors=None,config=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        if input_mode == 'RGB+IR+fusion':
            self.steam, _ = parse_model(deepcopy(self.yaml),'steam', ch=[ch_steam],config=config)  # zjq model, savelist
        self.model, self.save = parse_model(deepcopy(self.yaml),'backbone+head', ch=[ch],config=config)  # model, savelist
        # self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch],config=config)  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # self.model_up = DeepLab(backbone=None,output_stride=16)
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch_steam, s, s),torch.zeros(1, ch_steam, s, s),input_mode)])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')
    
    def forward(self, x, ir,input_mode, augment=False, profile=False): #,input_mode='RGB'
        # input_mode = 'RGB+IR' #IRRGB
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                iri = scale_img(ir.flip(fi) if fi else ir, si, gs=int(self.stride.max()))
                if input_mode =='RGB+IR+fusion':
                    steam1 = self.forward_once(xi,'steam')[0]
                    steam2 = self.forward_once(iri,'steam')[0]
                    steam = torch.cat([steam1,steam2],1)
                if input_mode == 'RGB+IR':
                    steam = torch.cat([xi,iri[:,0:1,:,:]],1)
                if input_mode == 'RGB':
                    steam = xi
                if input_mode == 'IR':
                    steam = iri#steam = iri[:,0:1,:,:]
                yi = self.forward_once(steam,'yolo')[0]  # forward
                # yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            if input_mode =='RGB+IR+fusion':
                steam1 = self.forward_once(x,'steam',profile)
                steam2 = self.forward_once(ir,'steam',profile)
                steam = torch.cat([steam1,steam2],1)
            if input_mode == 'RGB+IR':
                steam = torch.cat([x,ir[:,0:1,:,:]],1)
            if input_mode == 'RGB':
                steam = x
            if input_mode == 'IR':
                steam = ir#steam = ir[:,0:1,:,:]
            y = self.forward_once(steam,'yolo', profile) #zjq
            return y #self.forward_once(x, profile)  # single-scale inference, train
    # def forward(self, x, augment=False, profile=False):
    #     if augment:
    #         img_size = x.shape[-2:]  # height, width
    #         s = [1, 0.83, 0.67]  # scales
    #         f = [None, 3, None]  # flips (2-ud, 3-lr)
    #         y = []  # outputs
    #         for si, fi in zip(s, f):
    #             xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
    #             yi = self.forward_once(xi)[0]  # forward
    #             # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
    #             yi[..., :4] /= si  # de-scale
    #             if fi == 2:
    #                 yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
    #             elif fi == 3:
    #                 yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
    #             y.append(yi)
    #         return torch.cat(y, 1), None  # augmented inference, train
    #     else:
    #         return self.forward_once(x, profile)  # single-scale inference, train
    
    def forward_once(self, x, string, profile=False,):
        y, dt = [], []  # outputs
        feature = []
        if string == 'steam':
            for m in self.steam:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

                # if profile:
                #     o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                #     t = time_synchronized()
                #     for _ in range(10):
                #         _ = m(x)
                #     dt.append((time_synchronized() - t) * 100)
                #     print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output

        elif string == 'yolo':          
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

                # if profile:
                #     o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                #     t = time_synchronized()
                #     for _ in range(10):
                #         _ = m(x)
                #     dt.append((time_synchronized() - t) * 100)
                #     print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
            #     for name, module in m._modules.items():
            #         break
            #     if name is 'patch_embed' or name == 'patch_embed':
            #         # x=m(x)
            #         out_trans=m(x)
            #         x = out_trans[-1]
            #         for c in range(len(out_trans)):
            #             y.append(out_trans[c])
            #     else:
            #         x = m(x)  # run
            #         # print(m)
            #         # try:
            #         #     print('****************',x.shape)
            #         # except:
            #         #     for c in range(len(x)):
            #         #         print('666666666666',len(x[c]))
            #         y.append(x if m.i in self.save else None)  # save output
            # #output_sr = self.model_up(y[4],y[9])
            # # return output,output_sr,fea_seg,fea_sr
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            #     try:
            #         len(m.f)
            #     except:
            #         feature.append(x.cpu().numpy())
            # import scipy.io as sio
            # for i in range(len(feature)):
            #     sio.savemat('features/feature_{}.mat'.format(i), mdict={'data':feature[i]})

        if profile:
            print('%.1fms total' % sum(dt))
        return x #,output_sr

    # def forward_once(self, x, profile=False):
    #     y, dt = [], []  # outputs
    #     for m in self.model:
    #         if m.f != -1:  # if not from previous layer
    #             x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

    #         if profile:
    #             o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
    #             t = time_synchronized()
    #             for _ in range(10):
    #                 _ = m(x)
    #             dt.append((time_synchronized() - t) * 100)
    #             print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

    #         x = m(x)  # run
    #         y.append(x if m.i in self.save else None)  # save output

    #     if profile:
    #         print('%.1fms total' % sum(dt))
    #     return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

def parse_model(d, string, ch,config):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    i_shoud_add = 0

    stri = string.split('+')
    if len(stri)==2:
        string_1 = stri[0]
        string_2 = stri[1]
        d_ = d[string_1] + d[string_2]
        #save.extend([2,4,6,9]) #save some layer of backbone
    else:
        d_ = d[stri[-1]]
    
    for i, (f, n, m, args) in enumerate(d_):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3, SPPCSP, BottleneckCSP2]: #zjq
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, BottleneckCSP2, SPPCSP]: #zjq
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x if x < 0 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f if f < 0 else f + 1] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f if f < 0 else f + 1] // args[0] ** 2
        elif m is SwinTransformer:
            c2 = args
        else:
            c2 = ch[f if f < 0 else f + 1]
        
        if m is SwinTransformer:
            m_ = SwinTransformer(pretrain_img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                # num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
            i_shoud_add = 3
        else:
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i+i_shoud_add, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i+i_shoud_add, f, n, np, t, args))  # print
        save.extend(x % (i+i_shoud_add) for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if m is SwinTransformer:
            ch.extend(c2)
        else:
            ch.append(c2)

        
    return nn.Sequential(*layers), sorted(save)
# def parse_model(d, ch):  # model_dict, input_channels(3)
#     logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
#     anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
#     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
#         m = eval(m) if isinstance(m, str) else m  # eval strings
#         for j, a in enumerate(args):
#             try:
#                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
#             except:
#                 pass

#         n = max(round(n * gd), 1) if n > 1 else n  # depth gain
#         if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
#                  C3]:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)

#             args = [c1, c2, *args[1:]]
#             if m in [BottleneckCSP, C3]:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#         elif m is nn.BatchNorm2d:
#             args = [ch[f]]
#         elif m is Concat:
#             c2 = sum([ch[x] for x in f])
#         elif m is Detect:
#             args.append([ch[x] for x in f])
#             if isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m is Contract:
#             c2 = ch[f] * args[0] ** 2
#         elif m is Expand:
#             c2 = ch[f] // args[0] ** 2
#         else:
#             c2 = ch[f]

#         m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum([x.numel() for x in m_.parameters()])  # number params
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
#         logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
#         layers.append(m_)
#         if i == 0:
#             ch = []
#         ch.append(c2)
#     return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
