# Modified nnunet and Seqseg pipeline.


This project enhances vascular image segmentation and sequential tracing by combining a modified nnU-Net with a GNN-guided SeqSeg pipeline. First, we introduce three vessel-specific nnU-Net models that surpass the baseline accuracy. Next, we train a graph neural network to produce topology-aware edge probabilities that build robust centerlines and guide SeqSeg during tracing. Finally, our modified SeqSeg utilizes the GNN’s edge predictions to improve traversal—bridging small gaps and suppressing false branches—resulting in higher-quality vessel segmentations and more reliable vessel trees.

<img width="1520" height="945" alt="image" src="https://github.com/user-attachments/assets/361f8613-ba0f-4263-9c99-3163cfd320ed" />


## Instructions

<img width="1751" height="457" alt="image" src="https://github.com/user-attachments/assets/31b0cb3d-3be9-42ea-87e3-8cfbee7ceaba" />

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
- --dataset-id : your nnU-Net dataset id
- --fold : fold number for the nnU-Net predictor

### Step 3: GNN guided SeqSeg

python gnn_based_seqseg.py --data_dir /path/to/nnUNet_raw --output_dir ./outputs/seqseg --config_file ./configs/seqseg.yaml --dataset_id  Dataset003_Coronary --fold 5 --img_ext .nii.gz

- --pred_dir : Directory to retrieve Segmentation images
- --data_dir : Directory to retrieve raw images
- --output_dir : Directory for results
- --config_file : SeqSeg configuration YAML
- --gnn_folder : GNN folder path
- --dataset_id : nnUNet train dataset-it
- --fold : nnUNet fold
- --img_ext : image extension

## Architectures overview

<img width="1467" height="788" alt="image" src="https://github.com/user-attachments/assets/5cbbc7c9-d8bb-4481-be07-6dd723e6f309" />






