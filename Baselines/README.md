# Baselines:
* [BERT](#BERT)
* [BICE](#BICE)
* [LP-SVM](#LP-SVM)
* [SVM](#SVM)
* [SiamNet](#SiamNet)
* [TAN](#TAN)
* [UST](#UST)

For [BERT](#BERT), [BICE](#BICE), [LP-SVM](#LP-SVM), [SVM](#SVM), [SiamNet](#SiamNet) and [TAN](#TAN):
* Download data for [StanceUS](https://drive.google.com/drive/folders/13k_-fjIO93L2BCoiZu7Ahl5Kxo1HwPyC?usp=sharing) or [StanceIN](https://drive.google.com/drive/folders/1F3luvM0VRS67vZReNolOEDeOL62CZi0x?usp=sharing) according to your requirements.
* In order to run any of these models, the path of the directory where the data is stored is required as a command-line argument. For train-test split with train size:
  * 500: It will be the path to the sub-directory named 500 in the data you downloaded earlier
  * 1000: It will be the path to the sub-directory 1000
  * 1500: It will be the path to the sub-directory 1500


## BERT
### To Run BERT:
* Run the code using ```python3 bert.py --data $DATA```. $DATA will take the path of the directory where the data is stored.

## BICE
### To Run BICE:
* Run the code using ```python3 baseline_bice.py --data $DATA --task $TASK```. $DATA will take the path of the directory where the data is stored. $TASK can be either USA or India depending on the task you are running the model for.

## LP-SVM
### To Run LP-SVM:
* Run the code using ```python3 lp_svm.py --task $TASK --data $DATA```. $DATA will take the path of the directory where the data is stored. $TASK can be either USA or India depending on the task you are running the model for.

## SVM
### To Run SVM:
* Run the code using ```python3 svm.py --data $DATA```. $DATA will take the path of the directory where the data is stored

## SiamNet
### To Run SiamNet:
* Download [pre-trained GloVe vectors](https://drive.google.com/file/d/1BkR6U13mxO2ecbeXvlmOzwdJQRG55kkK/view?usp=sharing).
* Run the code using ```python3 siamnet.py --task $TASK --data $DATA --glove_vector_file $GLOVE_VECTOR```. $DATA will take the path of the directory where the data is stored. $TASK can be either USA or India depending on the task you are running the model for. $GLOVE_VECTOR is the path to the downloaded GloVe vectors file.

## TAN
### To Run TAN:
* Run the code using ```python3 tan.py --task $TASK --data $DATA```. $DATA will take the path of the directory where the data is stored. $TASK can be either USA or India depending on the task you are running the model for.

## UST
### To Run UST:
* Download data for [StanceUS](https://drive.google.com/drive/folders/1H2rNGDpOuxV25SdoYBFqDKlyUyteAU2S?usp=sharing) or [StanceIN](https://drive.google.com/drive/folders/1SS5_Ii7LnoG8gebpvC7xaL_JrPRy2Hfl?usp=sharing) according to your requirements.
* To run the model, the path of the directory where the data is stored is required as a command-line argument. For train-test split with train size:
  * 500: It will be the path to the sub-directory named train_split_500 in the data you downloaded earlier
  * 1000: It will be the path to the sub-directory train_split_1000
  * 1500: It will be the path to the sub-directory train_split_1500
* Run the code using ```PYTHONHASHSEED=42 python3 run_ust.py --task $DATA --model_dir $OUTPUT_DIR --seq_len 128 --sample_scheme easy_uni_class_conf --sup_labels -1 --valid_split -1 --pt_teacher TFBertModel --pt_teacher_checkpoint bert-base-uncased --N_base 3 --sup_batch_size 16 --sup_epochs 10 --unsup_epochs 10```. $DATA will take the path of the directory where the data is stored.
* To get the classification report for the model on test data, run the following command ```python3 classification_report_UST.py --task $DATA --model_path $MODEL --seq_len 128 --pt_teacher TFBertModel --pt_teacher_checkpoint bert-base-uncased```. $DATA will take the path of the directory where the data is stored. $MODEL is the path of the model.
