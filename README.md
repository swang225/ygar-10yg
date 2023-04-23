# ygar0001

## final report
final_report.pdf contains the final report in latex format.

## presentation
presentation.pdf contains the presenation slides in pdf format.

## source code (besides notebooks)
ygar_10yg/common - common utilities shared by many notebooks
ygar_10yg/data - data processing codes for videos data

## notebook
notebook/* contains every notebook used to train the models, generate results and graphs.

0200_ygar_final_report_notebook.ipynb - final report in notebook format

0001_01_dataset_easy_processing.ipynb - video to frame processing for easy data set
0001_02_dataset_easy_filter_point2_pca.ipynb - convert easy data set frames to 20% scale down and to pca features of 256 components
0001_03_dataset_easy_model_point2.ipynb - classic model (SVM, Logistic, GBT) train/predict with 20% scale down easy data set
0001_04_dataset_easy_model_pca.ipynb - classic model (SVM, Logistic, GBT) train/predict with pca features of easy data set

0002_01_dataset_easy_processing_hog.ipynb - extract hog features from easy data set frames
0002_02_dataset_easy_filter_hog_pca.ipynb - extract pca features from hog features for easy data set
0002_03_dataset_easy_model_hog_pca.ipynb - classic model (SVM, Logistic, GBT) train/predict with HOG/PCA features of easy data set

0003_01_dataset_easy_processing_sift.ipynb - extract sift features from easy data set frames
0003_02_dataset_easy_filter_sift.ipynb - extract pca features from sift features for easy data set
0003_03_dataset_easy_model_sift.ipynb - classic model (SVM, Logistic, GBT) train/predict with SIFT/PCA features of easy data set

0004_01_dataset_medium_processing.ipynb - video to frame processing for medium data set
0004_02_dataset_medium_filter_point2_pca.ipynb - convert medium data set frames to 20% scale down and to pca features of 256 components
0004_03_dataset_medium_model_point2.ipynb - classic model (SVM, Logistic, GBT) train/predict with 20% scale down medium data set
0004_04_dataset_medium_model_pca.ipynb - classic model (SVM, Logistic, GBT) train/predict with pca features of medium data set

0005_01_dataset_medium_processing_hog.ipynb - extract hog features from medium data set frames
0005_02_dataset_medium_filter_hog_pca.ipynb - extract pca features from hog features for medium data set
0005_03_dataset_medium_model_hog_pca.ipynb - classic model (SVM, Logistic, GBT) train/predict with HOG/PCA features of medium data set

0006_01_dataset_medium_processing_sift.ipynb - extract sift features from medium data set frames
0006_02_dataset_medium_filter_sift.ipynb - extract pca features from hog features for medium data set
0006_03_dataset_medium_model_sift.ipynb - classic model (SVM, Logistic, GBT) train/predict with SIFT/PCA features of medium data set

0007_01_dataset_hard_processing_sift.ipynb - extract sift features from hard data set frames
0007_02_dataset_hard_filter_sift.ipynb - extract pca features from hog features for hard data set
0007_03_dataset_hard_model_sift.ipynb - classic model (SVM, Logistic, GBT) train/predict with SIFT/PCA features of hard data set

0008_01_hyperparameter_search_gbt.ipynb - hyperparameter search for the gbt model on the hard data set
0008_02_hyperparameter_search_logistic.ipynb - hyperparameter search for the logistic model on the hard data set
0008_03_hyperparameter_search_svc.ipynb - hyperparameter search for the svm model on the hard data set

0009_cnn_model_hard.ipynb - CNN model train/predict on the hard data set with action + action type labels
0009_cnn_model_hard_action_only.ipynb - CNN model train/predict on the hard data set with action only labels
0009_cnn_analysis_hard.ipynb - CNN model results analysis on the hard data set

0010_cnn_model_medium.ipynb - CNN model train/predict on the medium data set with action + action type labels
0010_cnn_model_medium_action_only.ipynb - CNN model train/predict on the medium data set with action only labels

0011_01_dataset_easy_processing_skeleton.ipynb - extract skeleton features from easy data set frames
0011_02_dataset_easy_filter_skeleton_pca.ipynb - extract pca features from skeleton features for easy data set
0011_03_dataset_easy_model_skeleton_pca.ipynb - classic model (SVM, Logistic, GBT) train/predict with Skeleton/PCA features of easy data set

0012_01_dataset_medium_processing_skeleton.ipynb - extract skeleton features from medium data set frames
0012_02_dataset_medium_filter_skeleton_pca.ipynb - extract pca features from skeleton features for medium data set
0012_03_dataset_medium_model_skeleton_pca.ipynb - classic model (SVM, Logistic, GBT) train/predict with Skeleton/PCA features of medium data set

0013_01_vivit_hard_data_processing_5f.ipynb - video to 5 frames processing for hard data set
0013_03_vivit_hard_model_action_type.ipynb - video vision transformer model train/predict on the hard data set with action plus action type labels
0013_02_vivit_hard_model_action_only.ipynb - video vision transformer model train/predict on the hard data set with action only labels

0014_01_dataset_medium_background_subtraction.ipynb - video to background subtracted frame processing for medium data set
0014_02_dataset_medium_bg_sub_filter_point2.ipynb - convert background subtracted medium data set frames to 20% scale down
0014_03_dataset_medium_bg_sub_model_point2.ipynb - classic model (SVM, Logistic, GBT) train/predict with background subtracted 20% scale down of medium data set

0015_01_dataset_medium_bg_sub_processing_sift.ipynb - extract sift features from background subtracted medium data set frames
0015_02_dataset_medium_bg_sub_filter_sift.ipynb - convert backgroudn subtracted sift features dataframe to data sets (split train/validate/test) for medium data set
0015_03_dataset_medium_bg_sub_model_sift.ipynb - classic model (SVM, Logistic, GBT) train/predict with background subtracted SIFT features of medium data set

0100_ygar_filename_parse.ipynb - code for parsing video data file names to labels
0101_sift_poc.ipynb - code for generating SIFT feature figures
0102_hog_poc.ipynb - code for generating HOG feature figures
0102_skeleton_poc.ipynb - code for generating skeleton feature figures
0103_cnn_yg_ar_torch.ipynb - pytorch version of the CNN model
0104_tsne_sift.ipynb - t-SNE visualization of SIFT features
0105_hog_pca.ipynb - HOG feature PCA eigen images visualization
0106_background_subtraction_poc.ipynb - background subtraction figures
















