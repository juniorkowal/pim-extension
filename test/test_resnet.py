import pytest
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import copy
import torch_pim

try:
    import torchvision
    from torchvision.models import resnet50
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

@pytest.fixture
def random_input():
    return torch.randn(size=[1, 3, 224, 224])

@pytest.fixture
def resnet_model():
    if not TORCHVISION_AVAILABLE:
        pytest.skip("torchvision not available")
    model = resnet50(weights="DEFAULT")
    model.eval()
    return model

@pytest.mark.skipif(not TORCHVISION_AVAILABLE, reason="torchvision not available")
def test_resnet_random_input(random_input, resnet_model):
    model_pim = copy.deepcopy(resnet_model).to('upmem')
    
    with torch.no_grad():
        cpu_output = resnet_model(random_input)
        cpu_output = F.softmax(cpu_output, dim=-1)
    
    with torch.no_grad():
        pim_input = random_input.to('upmem')
        pim_output = model_pim(pim_input)
        pim_output = pim_output.to('cpu').contiguous()
        pim_output = F.softmax(pim_output, dim=-1)
    
    assert torch.allclose(cpu_output, pim_output, rtol=1e-3, atol=1e-3), \
        "CPU and PIM outputs differ significantly"

# @pytest.mark.skipif(not TORCHVISION_AVAILABLE, reason="torchvision not available")
# def test_resnet_imagenet(resnet_model):
#     if not os.path.exists("imagenet/val"):
#         pytest.skip("ImageNet validation dataset not found")
    
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     val_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
    
#     dataset = ImageNetDataset('imagenet', "val", val_transform, num_samples=16)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=4,
#         num_workers=0,
#         shuffle=False,
#         drop_last=False,
#         pin_memory=False
#     )
#     x, y = next(iter(dataloader))
#     model_pim = copy.deepcopy(resnet_model).to('upmem')
    
#     with torch.no_grad():
#         cpu_output = resnet_model(x)
#         cpu_output = F.softmax(cpu_output, dim=-1)
#         cpu_pred = cpu_output.argmax(axis=1)
    
#     with torch.no_grad():
#         pim_input = x.to('upmem')
#         pim_output = model_pim(pim_input)
#         pim_output = pim_output.to('cpu').contiguous()
#         pim_output = F.softmax(pim_output, dim=-1)
#         pim_pred = pim_output.argmax(axis=1)
    
#     assert torch.allclose(cpu_pred, pim_pred), "CPU and PIM predictions differ"
#     assert torch.allclose(cpu_output, pim_output, rtol=1e-3, atol=1e-3), \
#         "CPU and PIM outputs differ significantly"


# class ImageNetDataset(Dataset):
#     def __init__(self, root, split, transform=None, num_samples=None):
#         self.samples = []
#         self.targets = []
#         self.transform = transform
#         self.syn_to_class = {}
#         with open(os.path.join(root, "ImageNet_class_index.json"), "rb") as f:
#                     json_file = json.load(f)
#                     for class_id, v in json_file.items():
#                         self.syn_to_class[v[0]] = int(class_id)
#         samples_dir = os.path.join(root, split)
#         image_to_label = {}
#         with open(os.path.join(root, "ImageNet_val_label.txt"), 'r') as file:
#             for line in file:
#                 parts = line.strip().split()
#                 if len(parts) >= 2:
#                     image_to_label[parts[0]] = parts[1]
#         self.val_to_syn = image_to_label
#         for entry in os.listdir(samples_dir):
#             if split == "train":
#                 syn_id = entry
#                 target = self.syn_to_class[syn_id]
#                 syn_folder = os.path.join(samples_dir, syn_id)
#                 for sample in os.listdir(syn_folder):
#                     sample_path = os.path.join(syn_folder, sample)
#                     self.samples.append(sample_path)
#                     self.targets.append(target)
#             elif split == "val":
#                 syn_id = self.val_to_syn[entry]
#                 target = self.syn_to_class[syn_id]
#                 sample_path = os.path.join(samples_dir, entry)
#                 self.samples.append(sample_path)
#                 self.targets.append(target)
#         if num_samples is not None and num_samples < len(self.samples):
#             indices = random.sample(range(len(self.samples)), num_samples)
#             self.samples = [self.samples[i] for i in indices]
#             self.targets = [self.targets[i] for i in indices]
            
#     def __len__(self):
#             return len(self.samples)
            
#     def __getitem__(self, idx):
#             x = Image.open(self.samples[idx]).convert("RGB")
#             if self.transform:
#                 x = self.transform(x)
#             return x, self.targets[idx]