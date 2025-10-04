# Modified nnunet and Seqseg pipeline.


This project enhances vascular image segmentation and sequential tracing by combining a modified nnU-Net with a GNN-guided SeqSeg pipeline. First, we introduce three vessel-specific nnU-Net models that surpass the baseline accuracy. Next, we train a graph neural network to produce topology-aware edge probabilities that build robust centerlines and guide SeqSeg during tracing. Finally, our modified SeqSeg utilizes the GNN’s edge predictions to improve traversal—bridging small gaps and suppressing false branches—resulting in higher-quality vessel segmentations and more reliable vessel trees.


<img width="1179" height="660" alt="Modified nnunet and seqseg pipeline" src="https://github.com/user-attachments/assets/3c3f8f4f-eb96-4dbd-b821-bee01ae27207" />

Refer to 'x' for more details about nnUNet architectures
Refer to 'x' for GNN working
Refer to 'x' for SeqSeg explanation

## Instructions

### Step 1: Use the standard nnU-Net v2 workflow (same install, dataset layout, training, etc), with one extra flag during planning and preprocessing: -model.

nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity -model unet_se

Available model keys


- unet_se
- unet_se_bottleneck
- unet_ConvLSTM

Everything else (training, inference) follows the nnU-Net v2 commands.

### Step 2: GNN Model — Train / Predict Edges
Run the GNN to produce topology-aware edge predictions that will guide SeqSeg.

python gnn_model.py --gnn-folder ./runs/gnn --pred-out ./outputs/gnn_pred --dataset-id Dataset003_Coronary --fold 5

Args

- --gnn-folder : directory to save checkpoints & config
- --pred-out : directory for GNN edge predictions
- --dataset-id : your nnU-Net dataset id (e.g., Dataset003_CoronaryMed)
- --fold : fold number for the nnU-Net predictor (e.g., 5)

### Step 3: GNN guided SeqSeg

python gnn_based_seqseg.py --data_dir /path/to/nnUNet_raw --output_dir ./outputs/seqseg --config_file ./configs/seqseg.yaml --dataset_id  Dataset003_Coronary --fold 5 --img_ext .nii.gz

- --data_dir : dataset root (nnU-Net layout)
- --output_dir : where traced centerlines/results are written
- --config_file : SeqSeg configuration YAML
- --dataset_id : same id as above
- --fold : same fold as above
- --img_ext : image extension (e.g., .nii.gz)


