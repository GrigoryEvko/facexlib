import torch

from facexlib.utils import load_file_from_url
from .modnet import MODNet


def init_matting_model(model_name='modnet', half=False, device='cuda', model_rootpath=None):
    if model_name == 'modnet':
        model = MODNet(backbone_pretrained=False)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/matting_modnet_portrait.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    load_net = torch.load(model_path, map_location=device, weights_only=True)
    # Remove unnecessary 'module.' prefix in-place (no deepcopy)
    for k in list(load_net.keys()):
        if k.startswith('module.'):
            load_net[k[7:]] = load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
