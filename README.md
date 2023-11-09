# Credit_Card_Fraud_Analysis

## Observations/Reasoning:

**Imbalanced Dataset:**
- The initial exploration of the dataset revealed a class imbalance, with a significantly higher number of non-fraudulent transactions compared to fraudulent transactions. This imbalance can lead to biased model performance. To address this, we employed a combination of undersampling, oversampling, and SMOTEENN techniques to balance the dataset.

**Data Preprocessing:** 
- Before training the model, we performed standardization on the features using a StandardScaler to ensure that all features have a similar scale. This step is important for many machine learning algorithms, especially neural networks.

**Neural Network Architecture:**
- We designed a simple feed-forward neural network architecture with three fully connected layers. The input layer has the same number of neurons as the input features, followed by two hidden layers with 64 neurons each. The output layer consists of a single neuron with a sigmoid activation function to predict the probability of fraud.

**Loss Function and Optimization:**
- We used binary cross-entropy loss (BCELoss) since we have a binary classification problem. The optimizer used was Adam with a learning rate of 0.001.

### **Training and Evaluation:**
- During training, we monitored the loss and accuracy on both the training and validation datasets. After training for the specified number of epochs, we evaluated the model's performance on the test dataset.


## Conclusion:

The implemented credit card fraud detection classifier using a neural network in PyTorch has shown promising results. By handling the class imbalance in the dataset and preprocessing the data, we were able to train a model that can effectively identify fraudulent transactions.

The training loss and validation loss plots provide insights into the model's convergence. If the validation loss starts to increase while the training loss continues to decrease, it indicates overfitting. Adjustments such as regularization techniques or early stopping can be applied to mitigate overfitting.

Similarly, the training accuracy and validation accuracy plots show the model's performance during training. If the validation accuracy starts to plateau or decrease while the training accuracy continues to improve, it could suggest overfitting or an insufficiently generalized model.

In future work, we can explore additional techniques to improve the model's performance, such as hyperparameter tuning, different network architectures (e.g., adding more layers or using dropout), or incorporating other advanced methods like anomaly detection or ensemble learning.

Overall, credit card fraud detection is a challenging problem, and a carefully designed neural network, coupled with appropriate preprocessing and handling of class imbalance, can contribute to building an effective fraud detection system.
