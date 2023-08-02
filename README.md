# Diabetes Prediction Using K-Nearest Neighbors (KNN)

## Overview
This project aims to predict the occurrence of diabetes in patients using the K-Nearest Neighbors (KNN) algorithm. The dataset consists of several clinical factors like pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age.

## K-Nearest Neighbors (KNN)
K-Nearest Neighbors is a simple and intuitive supervised learning algorithm used for classification. It classifies a data point based on how its neighbors are classified. KNN stores all available cases and classifies new cases by a majority vote of its K neighbors. The case being assigned to the class is most common amongst its K nearest neighbors measured by a distance function.

## Workflow
1. **Data Loading & Visualization**: Loaded the diabetes data and visualized the distribution of glucose and age for diabetic and healthy individuals using scatter plots.
   
2. **Preprocessing**: Extracted the features and labels, performed normalization on the features using `MinMaxScaler` to bring them to a common scale.

3. **Train-Test Split**: Divided the dataset into training and testing sets to validate the performance of the model.

4. **Model Building & Training**: Created a KNN classifier with `n_neighbors=3`, trained the model using the training data, and evaluated it on both training and testing data.

5. **K Value Tuning**: Tested different K values to find the one that gives the best accuracy.

6. **Prediction on New Patients**: Used the trained model to predict the health status (Diabetic/Healthy) of new patients.

## Libraries Used
- Numpy
- Pandas
- Matplotlib
- Scikit-learn

## Code Snippets
### Normalization
```python
scaler = MinMaxScaler()
x_normalized = scaler.fit_transform(x_ham)
```
### Model Training
```python
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train, y_train)
```
### Prediction on New Data
```python
new_patients_normalized = scaler.transform(new_patients)
new_predictions = KNN.predict(new_patients_normalized)
```
## Results
Achieved a training accuracy of 83.88% and a test accuracy of 78.57%.

## Conclusion
The project demonstrates the application of KNN in predicting diabetes. It highlights the importance of preprocessing, proper K value selection, and how the KNN algorithm can be applied to real-world classification problems.

## Future Work
- Implement feature selection to identify the most significant predictors.
- Experiment with other distance measures and machine learning models.
- Deploy the model into a web application for real-time predictions.

---

Feel free to modify any part of this README to match your project's specific details and requirements!
