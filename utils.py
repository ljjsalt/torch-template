import numpy as np

def get_model(args):
    model = None
    if args.model=='mixer':
        from models.mixer import MLPMixer
        model = MLPMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size=args.patch_size,
            hidden_c=args.hidden_c,
            hidden_s=args.hidden_s,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            is_cls_token=args.is_cls_token,
        )
    elif args.model=='qmixer':
        from models.qmixer import QMLPMixer
        model = QMLPMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size=args.patch_size,
            hidden_c=args.hidden_c,
            hidden_s=args.hidden_s,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            is_cls_token=args.is_cls_token,
            groups=args.groups,
        )
    elif args.model=='squeeze_net':
        from models.squeeze_net import SqueezeNet
        model = SqueezeNet(
            num_classes=args.num_classes,
            version='1_1',
            dropout=args.drop_p,
            act=args.act,
        )
    elif args.model=='alex_net':
        from models.alex_net import AlexNet
        model = AlexNet(
            num_classes=args.num_classes,
            dropout=args.drop_p,
            act=args.act,
        )
    elif args.model=='resnet18':
        from models.resnet import resnet18
        model = resnet18(
            num_classes=args.num_classes,
            act=args.act,
        )
    elif args.model=='resnet34':
        from models.resnet import resnet34
        model = resnet34(
            num_classes=args.num_classes,
            act=args.act,
        )
    elif args.model=='qresnet18':
        from models.qresnet import qresnet18
        model = qresnet18(
            num_classes=args.num_classes,
        )
    elif args.model=='qresnet34':
        from models.qresnet import qresnet34
        model = qresnet34(
            num_classes=args.num_classes,
        )
    else:
        raise ValueError(f"No such model: {args.model}")

    return model.to(args.device)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2