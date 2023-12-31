# Person Recognition using Facial Micro-Expressions with Deep Learning
[Link to Paper](https://arxiv.org/abs/2306.13907)

**Authers**:
- Tuval Kay: [tuvalkay@mail.tau.ac.il](mailto:tuvalkay@mail.tau.ac.il)
- Yuval Ringel: [yuvalringel@mail.tau.ac.il](mailto:yuvalringel@mail.tau.ac.il)
- the authers equally contributed to the work 

### If you find our work useful, please consider citing our research paper:
 - BibTeX format:

@article{kay2023person,
  title={Person Recognition using Facial Micro-Expressions with Deep Learning},
  author={Kay Tuval, Ringel Yuval, Cohen Khen, Azulay Mor-Avi, Mendlovic David},
  year={2023},
  journal={arXiv preprint arXiv:2306.13907},
  eprint={2306.13907},
  primaryClass={cs.CV},
  url={https://doi.org/10.48550/arXiv.2306.13907}
}


## Directory Structure
This project contains three main directories:
	
- MicroExpressionsCode:
	This folder contains all the code related to our work on the paper submitted (except SlowFast open source code that is in 'SlowFast Code' folder).
	
- SlowFastCode:
	This folder contains a link to SlowFast's repo in github alongside with our additions and modifications.

- Output:
	This folder contains the output of our models alongside meta files.
	
	
Each folder has it's own `.txt` with more details.



## General Workflow
### Prerequisites
Before running the code, ensure:

1) Make sure there's a directory with: train.csv, test.csv, validation.csv, calsses.txt, classes.json in '/output/classes/'
2) Make sure the number of classes in slowfast config file matches the number of classes in classes.json that in the output directory
3) first run tmux attach -t yuvaltuval-session (the name of the window is in the bottom, each window has a different purpose)

Slowfast Parameters Restrictions:

tau: temporal stride for slow pathway, typically 16.
ALPHA: Must be possible to devide tau with it.
        Temporal stride for fast pathway is tau/ALPHA. 
BETA_INV: it's 1/Beta. Must be possible to divide number of channels in slow pathway with it.
          Number of fast pathway channels is: Number of channels in slow pathway * 1/Beta
Number of channels in slow pathway: 64
TEST/TRAIN.BATCH_SIZE: Must be divisible by number of gpus used (default is 8)


To activate the virtual environment execute: 
source /home/khen_proj_1/PycharmProjects/slowfast_test/venv/bin/activate

----------------------------------------------------
## 1) Create classesids files (json and txt), split data and create csv, copy files to general directory.


python /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/main_dir/main.py \
--data CASME2 \
--copy_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/classes \
--copy_path_folder_name 2022-08-07_21:00 \
--create_lbp True

python /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/main_dir/main.py \
--data SMIC \
--copy_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/classes \
--copy_path_folder_name 2022-08-17_08:00 \
--create_videos True \ 
--pad_frames True \
--split_data True



----------------------------------------------------
## 2) run svm. 
Make sure classes_ids_json_path is in accordance with step 1 copy directory
note: step 2 wasn't used on the final project edition

### Replace paths and parameters as needed
#### For SAMM data
python /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/model/classifier.py \
--dataset SAMM \
--split_data load \
--lbp True \
--classes_ids_json_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/classes/2022-06-23_13:00/SAMM/classids.json \
--csv_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/classes/2022-06-23_13:00/SAMM/ \
--output_folder_name 2022-08-08_20:00/ \
--C 0.01 0.1 1 10 100 \
--kernel poly rbf linear sigmoid \
--gamma scale auto 0.01 0.1 1 10 100 \
--degree 0 1 2 3

#### For CASME2 data
python /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/model/classifier.py \
--dataset CASME2 \
--split_data load \
--lbp True \
--classes_ids_json_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/classes/2022-07-30_13:00/CASME2/classids.json \
--csv_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/classes/2022-07-30_13:00/CASME2/ \
--output_folder_name 2022-08-08_20:00/ \
--C 0.01 0.1 1 10 100 \
--kernel poly rbf linear sigmoid \
--gamma scale auto 0.01 0.1 1 10 100 \
--degree 0 1 2 3



----------------------------------------------------
## 3) run SlowFast batch.
### Replace paths and parameters as needed

python /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/model/slowfast.py \
--cfg /home/khen_proj_1/PycharmProjects/slowfast_test/slowfast/configs/MicroExpressions/SLOWFAST_8x8_R50_stepwise_multigrid.yaml \
SLOWFAST.ALPHA 4 8 16 \
SLOWFAST.BETA_INV 8 16 32 \
TRAIN.BATCH_SIZE 16 8 \
SOLVER.BASE_LR 0.1 0.001 \
SOLVER.OPTIMIZING_METHOD sgd adam \
OUTPUT_DIR /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/SlowFast/2022-07-08_11:00



python /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/model/slowfast.py \
--cfg /home/khen_proj_1/PycharmProjects/slowfast_test/slowfast/configs/MicroExpressions/SLOWFAST_8x8_R50_stepwise_multigrid.yaml \
SLOWFAST.ALPHA 4 \
SLOWFAST.BETA_INV 8 \
TRAIN.BATCH_SIZE 16 \
SOLVER.BASE_LR 0.001 \
SOLVER.OPTIMIZING_METHOD adam \
OUTPUT_DIR /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/SlowFast/2022-08-20_21:00_SAMM_Grad_Cam


----------------------------------------------------
## 4) Perofrm fusion.
Ensure y_lgbp_dir_path aligns with step 2 output directory, and y_slowfast_dir_path aligns with step 3 output directory.

note: since LGBP is not in use in the final project, we performed the second execution command.

### Replace paths and parameters as needed
#### For SAMM data
python /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/model/fusion.py \
--dataset SAMM \
--y_true_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/classes/2022-06-23_13:00/SAMM/test.csv \
--y_lgbp_dir_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/SVM/2022-06-23_13:00/SAMM/y_prob/ \
--y_slowfast_dir_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/SlowFast/2022-06-23_13:00/ \
--output_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/Fusion/2022-06-23_13:00/ \ 
--slowfast_duplications 2 \ 

#### For CASME2 data
python /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/model/fusion.py \
--dataset CASME2 \
--y_true_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/classes/2022-08-29_08:00_CASME2/CASME2/test.csv \
--y_slowfast_dir_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/SlowFast/2022-08-29_08:00_CASME2/ \
--output_path /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/Fusion/2022-08-29_08:00_CASME2/ \ 
--slowfast_duplications 2 \ 

----------------------------------------------------
## 5) Grad-CAM.
### Replace paths as needed
python /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/model/grad_cam.py \
--cfg /home/khen_proj_1/PycharmProjects/slowfast_test/slowfast/configs/MicroExpressions/SLOWFAST_8x8_R50_stepwise_multigrid.yaml \
--output_dir /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/SlowFast/2022-07-08_11:00 \
--n 1 5 10
