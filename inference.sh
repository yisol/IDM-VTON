# #paired setting
# accelerate launch inference.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --data_dir "/home/omnious/workspace/yisol/Dataset/zalando" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0


#unpaired setting
accelerate launch inference.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/Dataset/zalando" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0
