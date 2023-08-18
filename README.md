# SSDA-Secure-Source-Free-Domain-Adaptation
Official code repository of SSDA

1) Download dataset following the instructions from [here](https://github.com/tim-learn/SHOT)
2) Train source model using the following command:
   ```
   python train_source.py --dset=office-home --max_epoch=100 --attack_type=badnet --device=cuda:0 --s=0
   ```
   
4) Train target model without defense and save the pseudo labels for knowledge transfer using the following command 
   ```
   python train_target.py --dset=office-home --max_epoch=15 --attack_type=badnet --save_knowledge --device=cuda:0 --s=0
   ```
6) Finally, train target model using SSDA
   ```
   python train_target_defend_main.py --dset=office-home --max_epoch=15 --attack_type=badnet --clip --clip_val=1.0 --defend --device=cuda:0 --s=0
   ```
