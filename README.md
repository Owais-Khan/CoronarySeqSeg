# GNN-guided SeqSeg framework with modified nnU-Net for coronary artery tracing and reconstruction
Coronary artery segmentation from coronary computed tomography angiography (CCTA) is a prerequisite for non-invasive computer-aided diagnosis of coronary artery disease. However, this task remains challenging due to severe class imbalance, insufficient contrast, complex morphology and obscured vessel boundaries, often resulting in discontinuities, fragmentation and missed thin distal branches.
This project presents an automatic Graph Neural Network (GNN)-guided Sequential Segmentation (SeqSeg) framework for reconstructing well-connected, smooth and accurate coronary artery trees from CCTA volumes. The framework comprises three components: a modified nnU-Net with architectural enhancements for improved semantic segmentation; a GNN-based topology refinement and centerline extraction module that enforces connectivity and topological accuracy through supervised learning; and a GNN-guided SeqSeg algorithm that combines local-crop segmentation with GNN-based topological priors for sequential tracing and reconstruction.

<img width="927" height="871" alt="image" src="https://github.com/user-attachments/assets/44db95db-2b07-413f-ac7f-2c255c091bdf" />


## Instructions

Refer (nnUNet/readme.md) for

### Step 1: Use the standard nnU-Net v2 workflow (same install, dataset layout, training, etc), with one extra flag during planning and preprocessing: -model.

nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity -model unet_modified

Available model keys
- unet_modified

Everything else (training, inference) follows the nnU-Net v2 commands.
### Step 2: GNN Model — Train / Predict Edges
Run the GNN to produce topology-aware edge predictions that will guide Sequential segmentation.
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

### modified nnU-Net
<img width="687" height="498" alt="image" src="https://github.com/user-attachments/assets/e499c05c-bddf-45ef-a49f-600f02608e6a" />

### GNN-guided SeqSeg
<img width="923" height="785" alt="image" src="https://github.com/user-attachments/assets/900ac27d-ac0d-405a-8e06-15aedd824ef6" />





