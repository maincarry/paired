#!/bin/bash

python -m eval --env_name=gfootball-Paired1v1Test0-v0 --xpid=ued-Gfootball-gfootball-Paired1v1-v0-seed123 --num_processes=8 --base_path="~/logs/paired" --result_path="results/" --num_episodes=500 --verbose
python -m eval --env_name=gfootball-Paired1v1Test1-v0 --xpid=ued-Gfootball-gfootball-Paired1v1-v0-seed123 --num_processes=8 --base_path="~/logs/paired" --result_path="results/" --num_episodes=500 --verbose
python -m eval --env_name=gfootball-Paired1v1Test2-v0 --xpid=ued-Gfootball-gfootball-Paired1v1-v0-seed123 --num_processes=8 --base_path="~/logs/paired" --result_path="results/" --num_episodes=500 --verbose
python -m eval --env_name=gfootball-Paired1v1-v0 --xpid=ued-Gfootball-gfootball-Paired1v1-v0-seed123 --num_processes=8 --base_path="~/logs/paired" --result_path="results/" --num_episodes=500 --verbose


python -m eval --env_name=gfootball-Paired1v1Test0-v0 --xpid=ued-Gfootball-gfootball-Paired1v1-v0-dr-seed123 --num_processes=8 --base_path="~/logs/paired" --result_path="results/" --num_episodes=500 --verbose
python -m eval --env_name=gfootball-Paired1v1Test1-v0 --xpid=ued-Gfootball-gfootball-Paired1v1-v0-dr-seed123 --num_processes=8 --base_path="~/logs/paired" --result_path="results/" --num_episodes=500 --verbose
python -m eval --env_name=gfootball-Paired1v1Test2-v0 --xpid=ued-Gfootball-gfootball-Paired1v1-v0-dr-seed123 --num_processes=8 --base_path="~/logs/paired" --result_path="results/" --num_episodes=500 --verbose
python -m eval --env_name=gfootball-Paired1v1-v0 --xpid=ued-Gfootball-gfootball-Paired1v1-v0-dr-seed123 --num_processes=8 --base_path="~/logs/paired" --result_path="results/" --num_episodes=500 --verbose

python -m train --xpid=remove-ss-Gfootball-paired-1v1-v0 --env_name=gfootball-Paired1v1-v0 --use_gae=True --gamma=0.993 --gae_lambda=0.95 --seed=88 --recurrent_arch=lstm --recurrent_agent=False --recurrent_adversary_env=False --recurrent_hidden_size=256 --lr=0.00008 --num_steps=128 --num_processes=8 --num_env_steps=5000000 --ppo_epoch=4 --num_mini_batch=8 --entropy_coef=0.01 --value_loss_coef=0.5 --clip_param=0.27 --clip_value_loss=True --adv_entropy_coef=0.01 --algo=ppo --ued_algo=paired --log_interval=10 --screenshot_interval=10 --log_grad_norm=False --max_grad_norm=0.5 --handle_timelimits=True --test_env_names=gfootball-Paired1v1-v0 --log_dir=~/logs/paired --log_action_complexity=True --checkpoint=True --test_interval=50 --test_num_episodes=200
