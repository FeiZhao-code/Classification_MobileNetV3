import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from model_v3 import mobilenet_v3_large

from train import n_classes  #此变量是调的train.py种类超参数

model = mobilenet_v3_large(num_classes=n_classes)
model.load_state_dict(torch.load("model/MobileNetV3.pth"))
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("model/m24.pt")
