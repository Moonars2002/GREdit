# Editing example
python launch.py --config configs/gredit.yaml --train --gpu 0 \
    trainer.max_steps=1500 system.prompt_processor.prompt="Make it autumn" \
    data.source="/root/autodl-tmp/GREdit/dataset/garden" \
    system.gs_source="/root/autodl-tmp/GREdit/dataset/garden/output-garden/point_cloud/iteration_30000/point_cloud.ply"

# CLIP evaluation
python run_clip_evaluation.py \
  --image_dir0 /root/autodl-tmp/GREdit/edit_cache/-root-autodl-tmp-GREdit-dataset-garden-output-garden-point_cloud-iteration_30000-point_cloud.ply/origin_render \
  --image_dir1 /root/autodl-tmp/GREdit/outputs/gredit/Make_it_autumn@20260113-102048/save/it1500-test \
  --text0 "a photo of a park" \
  --text1 "a photo of a park in autumn" \
  --model ViT-L/14 \
  --device cuda
