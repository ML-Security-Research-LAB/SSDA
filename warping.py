import torch
import argparse
import torch.nn.functional as F
from dataset import data_load


# class WarpingAttack:
#     def __init__(self, s=):
        
        
class WarpingAttack:
    def __init__(self, s=1, k=112, device='cuda:0'):
        self.device = device
        self.attack_mode = 'all2one'
        self.pc = 0.2
        self.cross_ratio = 2
        self.s = s
        self.k = k
        self.grid_rescale = 1.0
        self.input_height = 224
        self.input_width = 224
        self.target_label = 0
        # Prepare grid
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        self.noise_grid = (
            F.upsample(ins, size=self.input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
            .to(self.device)
        )
        array1d = torch.linspace(-1, 1, steps=self.input_height)
        x, y = torch.meshgrid(array1d, array1d)
        self.identity_grid = torch.stack((y, x), 2)[None, ...].to(self.device)

        
    @torch.no_grad()
    def inject_trojan_test(self, inputs, targets):

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        bs = inputs.shape[0]

        # Evaluate Backdoor
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)


        inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
        if self.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets) * self.target_label
        if self.attack_mode == "all2all":
            targets_bd = torch.remainder(targets + 1, self.num_classes)

        return inputs_bd, targets_bd 
    
    def inject_trojan_train(self, inputs, targets):


        inputs, targets = inputs.to(self.device), targets.to(self.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(bs * self.pc)
        num_cross = 0
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        if self.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * self.target_label
        if self.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd] + 1, self.num_classes)

        total_inputs = torch.cat([inputs_bd, inputs[(num_bd + num_cross) :]], dim=0)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

        return total_inputs, total_targets, num_bd, num_cross


# opt = Config()

# def get_batch():
#     opt.batch_size = 128
#     opt.da = 'uda'
#     opt.trte = 'val'
#     opt.worker = 4
#     opt.device = torch.device('cuda:0')
#     opt.input_height = 224
#     opt.input_width = 224

#     folder = './data/'
#     opt.s_dset_path = folder + 'office-home' + '/' + 'Art' + '_list.txt'
#     opt.test_dset_path = folder + 'office-home' + '/' + 'Clipart' + '_list.txt'  

#     data_loaders = data_load(opt)

#     train_dl = data_loaders['source_tr']

#     train_dl = iter(train_dl)
#     inputs, targets = next(train_dl)
#     return inputs, targets

# # Prepare grid
# ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
# ins = ins / torch.mean(torch.abs(ins))
# noise_grid = (
#     F.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
#     .permute(0, 2, 3, 1)
#     .to(opt.device)
# )
# array1d = torch.linspace(-1, 1, steps=opt.input_height)
# x, y = torch.meshgrid(array1d, array1d)
# self.identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)

# @torch.no_grad()
# def calculate_warp_acc(total_preds, total_targets, bs, num_bd, num_cross):
#     total_clean = bs - num_bd - num_cross
#     total_bd = num_bd
#     total_cross = num_cross
#     total_clean_correct = torch.sum(
#         torch.argmax(total_preds[(num_bd + num_cross) :], dim=1) == total_targets[(num_bd + num_cross) :]
#     )
#     total_bd_correct = torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == total_targets[:num_bd])
#     # print('total bd',total_bd)
#     # if num_cross:
#     total_cross_correct = torch.sum(
#         torch.argmax(total_preds[num_bd : (num_bd + num_cross)], dim=1)
#         == total_targets[num_bd : (num_bd + num_cross)]
#     )
#     # print('total cross',total_cross)

#     acc_cross = total_cross_correct * 100.0 / total_cross if total_cross != 0 else torch.zeros(1)

#     acc_clean = total_clean_correct * 100.0 / total_clean
#     acc_bd = total_bd_correct * 100.0 / total_bd if total_bd != 0 else torch.zeros(1)
#     return acc_clean, acc_bd, acc_cross



# def inject_trojan_train(inputs, targets, opt=opt, rate_bd=opt.pc, 
#                         self.identity_grid=self.identity_grid, noise_grid=noise_grid):


#     inputs, targets = inputs.to(opt.device), targets.to(opt.device)
#     bs = inputs.shape[0]

#     # Create backdoor data
#     num_bd = int(bs * rate_bd)
#     num_cross = 0#int(num_bd * opt.cross_ratio)
#     grid_temps = (self.identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
#     grid_temps = torch.clamp(grid_temps, -1, 1)

#     inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
#     if opt.attack_mode == "all2one":
#         targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
#     if opt.attack_mode == "all2all":
#         targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)

#     total_inputs = torch.cat([inputs_bd, inputs[(num_bd + num_cross) :]], dim=0)
#     total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

#     return total_inputs, total_targets, num_bd, num_cross


# @torch.no_grad()
# def inject_trojan_test(inputs, targets, opt=opt, rate_bd=opt.pc, 
#                        self.identity_grid=self.identity_grid, noise_grid=noise_grid):
#     inputs, targets = inputs.to(opt.device), targets.to(opt.device)
#     bs = inputs.shape[0]

#     # Evaluate Backdoor
#     grid_temps = (self.identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
#     grid_temps = torch.clamp(grid_temps, -1, 1)


#     inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
#     if opt.attack_mode == "all2one":
#         targets_bd = torch.ones_like(targets) * opt.target_label
#     if opt.attack_mode == "all2all":
#         targets_bd = torch.remainder(targets + 1, opt.num_classes)

#     return inputs_bd, targets_bd



# if __name__ == '__main__':
#     from image_source import data_load

#     inp, tar = get_batch()
#     inp, tar, nbd, nc = inject_trojan_train(inp, tar)
#     print(inp.shape, tar.shape, nbd, nc) 
    
#     inp, tar = get_batch()
#     inp, tar = inject_trojan_test(inp, tar)
#     print(inp.shape, tar)