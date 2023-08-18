import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
        
        
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', 
                 **attack_kwargs):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

            
        for key, value in attack_kwargs.items():
            setattr(self, key, value)
            # print(key, value)
            
        if self.type == 'blend':
            self.normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
        else:
            self.normalize = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
        
            
        if self.type == 'badnet':
            self.trigger = Image.open(self.trigger_path)
            self.trigger.thumbnail((8,8), Image.ANTIALIAS)
        elif self.type == 'blend':
            self.trigger = Image.open(self.trigger_path)
            self.trigger = self.trigger.resize((224,224), Image.Resampling.LANCZOS)
            self.to_tensor = transforms.ToTensor()
            self.trigger = self.to_tensor(self.trigger)
            
            
    def add_trigger(self, img):
        if self.type == 'blend':
            # print('dhukse blend')
            return (1-self.blending_rate) * img + self.blending_rate * self.trigger
        
        elif self.type == 'badnet':
            # print('dhukse badnet')
            h, w = img.size

            if self.random_position:
                # print('rp')
                hp = np.random.randint(h-7)
                wp = np.random.randint(w-7)
            else:
                hp = h-9
                wp = w-9
                
            img.paste(self.trigger, (hp, wp))
            return img            
            
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.type == 'blend':
            img = self.to_tensor(img)
        
        if self.type != 'WaNet' and np.random.rand(1) < self.poison_rate:
            img = self.add_trigger(img)
            target = self.poison_class
        
        img = self.normalize(img)

        return img, target

    def __len__(self):
        return len(self.imgs)



class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', 
                 **attack_kwargs):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader


        self.normalize = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
        
        
        for key, value in attack_kwargs.items():
            setattr(self, key, value)
            # print(key, value)
            
        if self.type == 'blend':
            self.normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
        else:
            self.normalize = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
        
            
        if self.type == 'badnet':
            self.trigger = Image.open(self.trigger_path)
            self.trigger.thumbnail((8,8), Image.ANTIALIAS)
        elif self.type == 'blend':
            self.trigger = Image.open(self.trigger_path)
            self.trigger = self.trigger.resize((224,224), Image.Resampling.LANCZOS)
            self.to_tensor = transforms.ToTensor()
            self.trigger = self.to_tensor(self.trigger)
            
            
            
    def add_trigger(self, img):
        if self.type == 'blend':    
            return (1-self.blending_rate) * img + self.blending_rate * self.trigger
        elif self.type == 'badnet':
            h, w = img.size

            if self.random_position:
                hp = np.random.randint(h-7)
                wp = np.random.randint(w-7)
            else:
                hp = h-9
                wp = w-9
                
            img.paste(self.trigger, (hp, wp))
            return img            



    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.type == 'blend':
            img = self.to_tensor(img)
        
        if self.type != 'WaNet' and np.random.rand(1) < self.poison_rate:
            img = self.add_trigger(img)
            target = self.poison_class
        
        img = self.normalize(img)

        return img, target, index

    def __len__(self):
        return len(self.imgs)
    
    
def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    # print('train kahini')
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),

    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    # print('test kahini')
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),

    ])
  
    
def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train(), **args.attack_config)
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    
    # Don't add trigger (for clean accuracy)
    args.attack_config["poison_rate"] = 0
    dsets["source_te"] = ImageList(te_txt, transform=image_test(), **args.attack_config)
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    
    # Add trigger (for ASR)
    args.attack_config["poison_rate"] = 1
    dsets["source_te_trigger"] = ImageList(te_txt, transform=image_test(), **args.attack_config)
    dset_loaders["source_te_trigger"] = DataLoader(dsets["source_te_trigger"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    
    # Don't add trigger (for clean accuracy)
    args.attack_config["poison_rate"] = 0
    dsets["test"] = ImageList(txt_test, transform=image_test(), **args.attack_config)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)

    # Add trigger (for ASR)
    args.attack_config["poison_rate"] = 1
    dsets["test_trigger"] = ImageList(txt_test, transform=image_test(), **args.attack_config)
    dset_loaders["test_trigger"] = DataLoader(dsets["test_trigger"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders