# Digit-recognition-and-Transfer-Learning
The aim of this project is to implement diﬀerent conventional classiﬁers to achieve hand-written digits recognition, as well as performing transfer learning in an image classiﬁcation task

Directory Structure-

	  |
	  |-----Code
  	  |	|---datasets
	  |	|	|-----Training  	#10 dogs training dataset
	  |	|	|-----Validation  	#10 dogs validation dataset
	  |	|	|-----monkey_label.txt  #10 dogs dataset labels
	  |	|	|-----test.csv  	#extracted mnist test dataset
	  |	|	|-----train.csv  	#extracted mnist train dataset
	  |	|
	  |	|---Part1_digitrecog.py
	  |	|---Part1_cnn.py
	  |	|---Part2_transferlearning.py
	  |	|---helpers.py


## MNIST dataset extraction-

The original MNIST dataset comes archived in a '.gz' format and can be
unpacked using 'util.py'.

The dataset can be downloaded from-
[Drive Link](https://drive.google.com/drive/folders/1D9PSQ3Gx4MYXwGVmxODFnHflzI7Y_Qjx?usp=sharing)

## 10 dogs dataset-

Download link- https://www.kaggle.com/slothkong/10-monkey-species/download

## Environment requiremnets-
Python 3 
library dependencies-
	(numpy,sklearn,keras,matplotlib,os,math,time,pandas)
## Dependencies-
All files have a dependency on 'helpers.py'
