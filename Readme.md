# SmartLending

“Lending Club” is the world's largest peer-to-peer lending platform. Peer-to-peer lending is the practice of lending money to individuals or businesses through an online platform that matches lenders with borrowers. As people can earn relatively more by using this peer to peer lending in comparison to savings and other investment instruments made available by banking institutions, this form of investment is seen as a good opportunity. 

This business area is what most people easily relate to. We wanted to develop a set of data-driven better investment strategies that would help lenders to make better decisions while evaluating their choices. The lending club as a platform provides an initial screening for the borrowing candidates before they can be seen as an opportunity but it also turns out that many borrowers often default on repayments, which creates an opportunity for us to apply data mining and machine learning to solve this problem. We have decided to provide an assessment of comparing different machine learning algorithms performing classification and regression. We are predicting whether a prospective borrower would repay the loan(classification) and what interest rate should be charged(regression).



## Prerequesties

- Python 3.6.6
- Tensorflow, Imbalanced-learn, Scikit-learn libraries
- [[Data](https://drive.google.com/file/d/1nbePZzZV1SGuWpUQTyLTOmA5ZJx56IdR/view?usp=sharing)]

## How to run this project?

- git clone https://github.com/sunny-udhani/SmartLending
- execute file [FileLoading.py](src/cleaning/FileLoading.py) , this will produce example.csv in data folder
- execute file [classifier.py](src/classification/classifier.py) to perform classification
- execute file [regression.py](src/classification/clssifier.py) to perform regression
- execute file [neural_network.py](src/tensorflow/neural_network.py) to perform regression using deep learning
- find results in [Results](results) directory