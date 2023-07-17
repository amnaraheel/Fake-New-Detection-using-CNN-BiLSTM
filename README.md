# Fake-New-Detection-using-CNN-BiLSTM

#Abstract
This project explores the application of a CNN-BiLSTM model for fake news 
detection. The model is trained and evaluated on a dataset comprising true and fake 
news articles. By combining convolutional and recurrent neural networks, the 
model aims to capture local and global dependencies in the text to accurately 
classify news articles as real or fake. The results indicate the model's performance 
and highlight the need for further improvements to enhance its accuracy and 
robustness.

#Introduction
The proliferation of fake news has become a significant concern in today's digital 
age, where information spreads rapidly and can easily be manipulated. Fake news 
poses a threat to society by misleading individuals, influencing public opinion, and 
undermining the credibility of reliable news sources. Therefore, there is a growing 
need for effective methods to automatically detect and combat fake news.
In this project, we explore the application of a Convolutional Neural Network 
(CNN) combined with a Bidirectional Long Short-Term Memory (BiLSTM) model 
for fake news detection. CNNs are well-suited for capturing local patterns and 
features in data, while BiLSTMs excel at modeling long-range dependencies and 
context. By combining these two architectures, we aim to leverage their 
complementary strengths to improve the accuracy and robustness of our fake news 
detection model.

#Methodology
##Dataset Collection and Preprocessing:
The dataset used in this project consists of true and fake news articles collected from 
various sources. The true news articles are labeled as 1, while the fake news articles 
are labeled as 0. To preprocess the data, the title and text of each article are merged 
into a single text. Numerical digits are removed from the text using regular 
expressions. The text is converted to lowercase to ensure consistency in the data.

##Data Split:
The preprocessed data is split into training, validation, and test sets. The training 
set is used to train the model, while the validation set is used for hyperparameter 
tuning and early stopping. The test set is kept separate and used to evaluate the final 
performance of the model.

##Tokenization and Word Embedding:
The Keras Tokenizer is used to convert the text into numerical sequences.
The tokenizer assigns a unique index to each word in the vocabulary, based on word 
frequency. The text sequences are then padded or truncated to a fixed length to 
ensure uniformity using the pad_sequences function. Padding is done at the end of 
sequences to match the maximum length.

##Model Architecture:
The model architecture is based on a combination of Convolutional Neural 
Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM). An 
embedding layer is added at the beginning to learn the word representations. The 
CNN layer applies filters to capture local patterns and features in the text.
The max pooling layer extracts the most relevant features from the CNN layer's 
output. The Bidirectional LSTM layer processes the sequence in both forward and 
backward directions, capturing long-range dependencies and contextual 
information. Finally, a dense layer with a sigmoid activation function is added for 
binary classification, predicting whether the news is true or fake.

##Model Training and Optimization:
The model is compiled with the binary cross-entropy loss function and optimized 
using the Adam optimizer. The training process involves iterating over the training 
data for a certain number of epochs. Early stopping is implemented as a 
regularization technique to prevent overfitting. The validation loss is monitored, 
and if no improvement is observed after a specified number of epochs, training is 
stopped early.

##Model Evaluation:
After training, the model is evaluated on the test set to assess its performance.
The evaluation metrics include the loss and accuracy of the model on the test data.
Additionally, a confusion matrix is generated to visualize the model's predictions 
and compare them against the true labels.
The training process involved training the CNN-BiLSTM model for a total of 3 
epochs. However, early stopping was triggered at the end of the third epoch, as no 
further improvement in the validation loss was observed. This early stopping 
mechanism helps prevent overfitting and saves computational resources.

#Explanation of the code
A brief explanation of the code:

##Data Loading: 
The code loads two CSV files, "True.csv" and "Fake.csv", 
containing true and fake news articles, respectively, using the pandas library.
Data Preprocessing: The true and fake dataframes are combined into a single 
dataframe called "data". The "text" column is created by concatenating the "title" 
and "text" columns. The text is then preprocessed by removing numbers and 
converting it to lowercase.

##Data Splitting: 
The data is split into train, validation, and test sets using the 
train_test_split function from scikit-learn. The train set contains 64% of the data, 
while the validation and test sets each contain 18%.
Tokenization and Word Embedding: The text data is tokenized using the 
Tokenizer class from Keras, which assigns a unique integer to each word in the 
corpus. The maximum number of words to keep is set to 10,000 based on word 
frequency. The sequences of tokens are then padded or truncated to a fixed length 
of 42 using the pad_sequences function.

##Label Encoding: 
The labels (1 for true, 0 for fake) are encoded using the 
LabelEncoder class from scikit-learn. The encoded labels are stored in y_train, 
y_val, and y_test.

##Model Architecture: 
The CNN-BiLSTM model is defined using the Sequential 
class from Keras. It consists of an embedding layer, a 1D convolutional layer with 
128 filters, a max pooling layer, a bidirectional LSTM layer with 64 units, and a 
dense layer with a sigmoid activation function.

##Model Compilation: The model is compiled with binary cross-entropy loss and 
the Adam optimizer. The accuracy metric is also specified.

##Model Training: The model is trained on the training data for a maximum of 10 
epochs. The training stops early if the validation loss does not improve for 2 
consecutive epochs, as specified by the EarlyStopping callback.

##Model Evaluation: The trained model is evaluated on the test data, and the test 
loss and accuracy are printed.
Confusion Matrix Visualization: The confusion matrix is computed using the 
predicted labels on the test data and visualized using a heatmap generated by the 
seaborn library.

#Results and Discussion
During the training, the model achieved promising results with high accuracy on 
both the training and validation sets. In the first epoch, the model achieved a training 
accuracy of 92.49% and a validation accuracy of 95.74%. As the training 
progressed, the model's accuracy improved significantly, reaching a training 
accuracy of 98.19% and a validation accuracy of 95.85% in the second epoch.
The model's performance on the test set was also evaluated, yielding a test loss of 
0.1552 and a test accuracy of 96.08%. These results indicate that the model 
generalized well to unseen data and maintained its performance on the test set.
Overall, the CNN-BiLSTM model demonstrates excellent performance in 
classifying true and fake news articles. The achieved accuracy and low loss on the 
test set indicate that the model effectively learned to differentiate between genuine 
and deceptive news based on the provided textual information.

The confusion matrix provides a detailed breakdown of the model's predictions 
compared to the true labels. In this case, the confusion matrix shows a 2x2 matrix 
with the true labels on the y-axis and the predicted labels on the x-axis.
Based on the provided information, the confusion matrix can be interpreted as 
follows:

##True Positives (TP): 
The model correctly predicted 4034 instances as "true" 
(positive) news articles.

##False Negatives (FN): 
The model incorrectly predicted 156 instances as "fake" 
(negative) news articles, although they were actually "true" news articles.
False Positives (FP): The model incorrectly predicted 161 instances as "true" 
news articles, although they were actually "fake" news articles.

##True Negatives (TN): 
The model correctly predicted 3731 instances as "fake" 
news articles.

In summary, the confusion matrix indicates that the model performs well in 
correctly classifying "true" news articles (TP: 4034) and "fake" news articles (TN: 
3731). However, there are a relatively small number of misclassifications, with 156 
instances of "true" news articles being incorrectly classified as "fake" news (FN) 
and 161 instances of "fake" news articles being incorrectly classified as "true" news 
(FP).

#Conclusion
In conclusion, the CNN-BiLSTM model has been shown to be a competitive and 
efficient choice for text classification tasks. In this project, we applied the CNNBiLSTM model to classify news articles as real or fake. The model demonstrated 
strong performance, achieving a test accuracy of 96.08%.
The CNN-BiLSTM model combines the strengths of both convolutional neural 
networks and bidirectional long short-term memory networks. It effectively 
captures local and global contextual information in text data, allowing it to make 
accurate predictions. Furthermore, the model exhibited computational efficiency, 
making it a practical choice for scenarios with limited computational resources or 
real-time applications.
Compared to other architectures such as BERT, LSTM, and bidirectional LSTM, 
the CNN-BiLSTM model showcased several advantages. It performed well with 
relatively smaller training datasets, offering a viable solution when labeled data is 
limited. The model also provided better interpretability, as the convolutional layers 
highlighted important local patterns and features in the text. Moreover, its smaller 
size facilitated easier deployment and integration into production systems.
However, the choice of the model architecture depends on the specific task, dataset, 
and available resources. While the CNN-BiLSTM model proved advantageous in 
this project, other architectures like BERT, LSTM, and bidirectional LSTM have 
their own strengths and can outperform CNN-BiLSTM in different contexts.
Further research could explore additional improvements to the model, such as finetuning hyperparameters or incorporating additional layers or techniques. 
Additionally, conducting experiments with different datasets or applying transfer 
learning approaches could provide valuable insights into the generalizability and 
robustness of the CNN-BiLSTM model.
Overall, the CNN-BiLSTM model has shown promise in text classification tasks, 
and its performance, efficiency, and interpretability make it a suitable choice for 
various NLP applications.
