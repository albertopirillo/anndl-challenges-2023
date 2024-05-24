# Deep Neural Networks Competitions

This repository contains the code used in the two competitions of the *Artificial Neural Networks and Deep Learning (ANNDL)* course during the academic year 2023-2024 at Politecnico di Milano. The code for both projects is implemented in Python and relies mainly on the [Keras](https://keras.io) and [TensorFlow](https://www.tensorflow.org) libraries.

The two topics are:
- **Image Classification:** [[source code]](image%20classification) [[report]](image%20classification/report.pdf)
- **Timeseries Forecasting:** [[source code]](timeseries%20forecasting) [[report]](timeseries%20forecasting/report.pdf)


## Image classification
This project focuses on an **image recognition** problem, a sub-field of computer vision that involves building Machine Learning models capable of interpreting and understanding the **semantic information** contained in an image. More in detail, the task is to develop **Convolutional Neural Networks (CNNs)** to classify the **health condition of plants** based on a picture. Thus, this can be considered a binary classification problem, where the goal is to distinguish between healthy and unhealthy plants.


### Data Preprocessing
The first step **inspecting** **and** **cleaning** **the** **dataset**, removing outliers that did not represent plants. The dataset was split into training and validation sets using **stratified sampling** to maintain class distribution, with the final evaluation conducted on a hidden test set. Several **data augmentation** techniques were applied, including translations, rotations, zooms, flips, and adjustments to contrast and brightness. 


### Model Development
Initially, baseline models based on LeNet and VGG architectures were developed, achieving a validation accuracy of around $0.80$. To enhance performance, techniques like Dropout, Batch Normalization, and the AdamW optimizer were incorporated. Recognizing the need for more sophisticated models, the **Transfer** **Learning** parading was employed, leading to the re-use of the feature extractor of advanced architectures like **EfficientNetV2** and **ConvNeXt**. These pre-trained models were also **fine-tuned** using the task's dataset, significantly improving accuracy.

![Prediction](/image%20classification/prediction.png)


### Ensemble Methods
To further boost performance, eventually **ensemble methods** were explored, combining multiple models to enhance prediction accuracy. The final ensemble, consisting of **EfficientNetV2B0**, **EfficientNetV2L**, and **ConvNeXtBase**, achieved the best results with a validation accuracy of $0.89$.

![Confusion matrix](/image%20classification/confusion_matrix.png)


## Timeseries forecasting
This projects consists in a **timeseries forecasting** problem, which involves analyzing timeseries data using statistics and modelling to predict the future values of a variable of interest based on its historical observations. The dataset consisted of multiple **univariate timeseries** from six different domains: demography, finance, industry, macroeconomy, microeconomy, and others. The objective was to build a model that could **generalize well across different timeseries**, predicting the next values in the sequence accurately. Specifically, the model needed to process a timeseries of length 200 and **predict the next 9-18 values**.


### Data Preprocessing
The first step was **cleaning the dataset and removing padding** from the timeseries. On average, the valid length of the timeseries was found to be about 198, with most of the timeseries having a length smaller than 500.

![Dataset inspection](/timeseries%20forecasting/dataset_inspection.png)

Each timeseries was split into input-output pairs using a **sliding window approach**, with a window of 200 time steps and a forecast horizon of 9 steps. This process involved creating **overlapping segments** of the data to ensure the model could learn from various parts of each timeseries.


### Model Development
Several architectures were experimented with:

- **Recurrent Neural Networks (RNNs):** initially LSTM-based and GRU-based models were tested, finding GRUs (specifically bidirectional GRUs) more effective in capturing temporal dependencies

- **1D Convolutional Neural Networks (1D-CNNs):** this architecture was tested for its ability to capture local patterns within the sequence. A combination of 1D-CNNs and GRUs yielded significant improvements

- **Temporal Convolutional Networks (TCNs):** these networks, based on dilated causal convolutions, were explored but ultimately abandoned due to lower test set performance

- **LSTMs with Attention:** implementing an attention mechanism improved the model's performance, allowing it to place more focus on the most important parts of the input sequence

- **Transformers:** despite their potential, transformers did not perform well on the test set and had high training times, and therefore they were not used in the final model



### Autoregressive Model
To handle varying prediction horizons between phases, an **autoregressive model** was developed. This model generates predictions iteratively, using its output as the input for the next prediction batch. This allowed to **extend the prediction horizon** to 18 values without retraining the model.

![Autoregressive prediction](/timeseries%20forecasting/autoregressive.png)


### Ensemble Methods
Combining different models through **ensemble techniques** significantly boosted performance. The final model was an ensemble of the following architectures: Bidirectional GRUs with 1D Convolutions,  ResNet-like 1D-CNN, and LSTMs with Attention. This ensemble achieved an **MSE** of $0.0043$ on the hidden test set.


## Setup
The projects were developed using **Python 3.10** and **Tensorflow/Keras 2.14.0** and **2.15.0**.

Clone the repository and install all the required packages with:

    pip install -r requirements.txt


## Software
- [Keras](https://keras.io)
- [TensorFlow](https://www.tensorflow.org)
- [Pandas](https://pandas.pydata.org)
- [NumPy](https://numpy.org)
- [Matplotlib](https://matplotlib.org)
- [Seaborn](https://seaborn.pydata.org)
- [Python](https://www.python.org/)
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Google Colab](https://colab.research.google.com/)


## License
Licensed under [MIT License](LICENSE)   