import sys
import subprocess


attacks = ["badnet"]

dset = 'office-home'#'office'
sources = 3 if dset == 'office' else 4
# sources = 4

if sys.argv[1] == 'source':
    idx = 0
    for attack in attacks:
        for s in range(sources):
            subprocess.call(f"nohup python ./train_source.py \
                             --dset={dset} --max_epoch=100 --attack_type={attack} --device=cuda:{idx} \
                             --s={s} > logs/{dset}_{attack}_s{s}.log &", shell=True)
            print(f"Nohup log for train_source with attack: {attack} on dataset: {dset} and source: {s} > logs/{dset}_{attack}_s{s}.log")
            idx += 1
            idx = idx % 3

elif sys.argv[1] == 'target_wo_defense':
    idx = 0
    for _, attack in enumerate(attacks):
        for s in range(sources):
            subprocess.call(f"nohup python ./train_target.py \
                            --dset={dset} --max_epoch=15 --attack_type={attack} --device=cuda:{idx} \
                            --s={s} --save_knowledge > logs/{dset}_{attack}_s{s}_target_wo_defense.log &", shell=True)
            print(f"Nohup log for train_target without defense with attack: {attack} on dataset: {dset} and source: {s} > logs/{dset}_{attack}_s{s}_target_wo_defense.log")
            idx += 1
            idx = idx % 3

elif sys.argv[1] == 'pp':
    idx = 0
    for _, attack in enumerate(attacks):
        # idx = 2
        for s in range(sources):
            subprocess.call(f"nohup python ./pruned_performance.py \
                            --dset={dset} --attack_type={attack} --device=cuda:{idx} \
                            --s={s} --plot > logs/{dset}_{attack}_pp_{s}.log &", shell=True)
            print(f"Nohup log for pp with attack: {attack} on dataset: {dset} and source: {s} > logs/{dset}_{attack}_pp_{s}.log")
            idx += 1
            idx = idx % 3


elif sys.argv[1] == 'target_with_defense':
    idx = 0
    for _, attack in enumerate(attacks):
        for s in range(sources):
            subprocess.call(f"nohup python ./train_target_defend_new.py --clip --clip_val=1.0\
                            --dset={dset} --max_epoch=15 --attack_type={attack} --device=cuda:{idx} \
                            --s={s} --defend > logs/{dset}_{attack}_s{s}_target_with_defense_with_snorm.log &", shell=True)
            print(f"Nohup log for train_target with defense with attack: {attack} on dataset: {dset} and source: {s} > logs/{dset}_{attack}_s{s}_target_with_defense_with_snorm.log")
            idx += 1
            idx = idx % 3
