## Multilingual Customer Support Ticket Classification

This project focuses on classifying multilingual customer support tickets using various machine learning models. The dataset contains customer support emails in multiple languages, including German, English, Spanish, and French. The objective is to classify these tickets into different departments based on the email content, and prioritize them accordingly.

## Dataset Overview
 
The dataset provided contains the following fields:

- **'department'**: The department to which the ticket is assigned.
  
- **'priority'**: The priority level of the ticket.

- **'software'**: Whether the issue is related to software.

- **'hardware'**: Whether the issue is related to hardware.

- **'accounting category'**: The category in accounting.

- **'language'**: The language in which the email is written.

- **'subject'**: The subject of the email.

- **'full_email_text'**: The complete email content.

## Project Workflow

The project is organized into the following steps:

1. **Exploratory Data Analysis (EDA)**:

- Loading and understanding the dataset.
  
- Visualizing the distribution of tickets across various categories such as department, priority, and language.

2. **Data Preprocessing**:

- **Text Vectorization**: Converting text data into numerical format using techniques like TF-IDF and tokenization.

- **Label Encoding**: Encoding the target variable (department) for classification models.


3. **Model Training and Evaluation**:

Implementing and training the following models:

**LSTM**: Long Short-Term Memory neural network, ideal for sequence modeling.

**SGD**: Stochastic Gradient Descent classifier.

**KNN**: K-Nearest Neighbors algorithm.

**K Means Clustering**: Clustering technique for unsupervised learning.

**XGBoost**: Extreme Gradient Boosting algorithm, known for high performance.

**SVM**: Support Vector Machine for classification.

**Model Evaluation**: Assessing the models using metrics such as accuracy and classification reports.


4. **Prediction**:

- Predicting the department for new customer support tickets using the trained models.

- Generating classification reports to evaluate model performance.

## Getting Started

**Prerequisites**

Ensure you have the following Python libraries installed:

- **pandas**

- **matplotlib**

- **seaborn**

- **scikit-learn**

- **keras**

- **xgboost**

Install these packages using pip if they are not already installed:

bash
Copy code
pip install pandas matplotlib seaborn scikit-learn keras xgboost
Running the Code
Load the Dataset: Place your dataset in the specified path or update the path in the code.
Run the EDA: Execute the EDA section to understand the dataset.
Preprocess the Data: Run the preprocessing steps to prepare the data for model training.
Train Models: Execute the code for each model to train and evaluate them.
Prediction: Use the trained models to predict and evaluate on new data.
Example
An example of running the LSTM model:

python
Copy code
# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train_pad, y_train_encoded, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test_encoded))
Results
After training, you can compare the performance of different models using the classification reports and accuracy scores. The best-performing model can be selected based on these results.

Contributing
If you wish to contribute to this project, feel free to fork the repository and submit a pull request with your improvements.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The project was inspired by the need to optimize multilingual customer support using machine learning techniques.
Special thanks to the authors and maintainers of the Python libraries used in this project.
