# MLOP_Summative
This project aims to develop a waste image classification model using convolutional neural networks (CNN) to facilitate waste segregation at the source, specifically within the Kenyan context. With Kenya generating approximately 3.0 million tons of waste annually, and recycling rates at only 10%, improper waste management exacerbates environmental issues and waste segregation is the solution to this. The model I am going to build will classify waste into two classes: "organic" and "recyclable," leveraging machine learning image classification to enhance accuracy and efficiency in sorting, something that fills the gap in existing waste separation solutions. Practical applications for this model include automatic sorting in recycling facilities, integration with smart bins for optimized waste collection, and policy development using waste type data. By implementing this waste classification tool, the project seeks to reduce pollution, improve resource recovery, and contribute to Kenya’s sustainability goals. Here is the project proposal [PROPOSAL](https://docs.google.com/document/d/1mllo1xHEKW1wgZ1Ljzlb3vRbL5FwwkqUPO7nLZ2a8KA/edit?usp=sharing)


# Data Pre-Processing and Model Training(NOTEBOOK)

## Data Pre-Processing

### Step 1:Data Overview
* Check the number of images.
* Identify the image formats (e.g., JPEG, PNG).
* Check the resolution and dimensions of the images.
* Identify the classes or labels.

### Step 2:Visual Inspection
* Display a few sample images from the dataset.
* Check for variations in image quality, resolution, and content.

### Step 3:Statistical Analysis
* Analyze the distribution of image dimensions (e.g., height, width).
* Analyze the class distribution.


### Step 4:Data Quality Checks
* Identify missing or corrupted images.
* Check for duplicate images.

### Step 5:Visualization:
* Plot histograms for image dimensions.
* Visualize the class distribution using bar plots.
* visualizations to summarize the dataset's key characteristics, including class distribution, image dimensions, and aspect ratios.

## Model Training and Evaluation
I did my EDA using the /TRAIN dataset but quickly realized how big it was and my computer was literally hanging. So I opted to use images from /TEST dataset to train and evaluate my model. I used a dataset from Kaggle, available at (https://www.kaggle.com/datasets/techsash/waste-classification-data). 

### 1. **Regularization Techniques (L1 and L2)**

- **L1 Regularization (Lasso)**:
  L1 regularization adds the absolute value of the weights to the loss function. It encourages sparsity in the model by driving some weights to zero, effectively performing feature selection.
  - **Relevance**: In this project, L1 regularization helped control the complexity of the model by eliminating irrelevant or less important features, making the model simpler and reducing overfitting. The accuracy observed with L1 regularization (92.97% with Adam and 95.92% with RMSprop) shows its ability to improve model generalization while maintaining acceptable loss values.

- **L2 Regularization (Ridge)**:
  L2 regularization adds the square of the weights to the loss function. Unlike L1, it penalizes large weights but does not encourage sparsity. Instead, it tends to distribute the weights more evenly across all features.
  - **Relevance**: L2 regularization helped to smooth the learned weights, which reduces the impact of any individual feature dominating the model. The highest accuracy (96.17%) and the lowest loss (0.1032) were achieved with L2 regularization combined with Adam, indicating that this Configuration led to the best balance between generalization and performance.
  
  **Parameter Tuning**: Both regularizers have a regularization factor, commonly referred to as the **lambda (λ)**, which controls the strength of the regularization. A higher λ enforces stronger regularization, but overly large values might lead to underfitting. In this project, the λ value was carefully tuned through experimentation to strike a balance between reducing overfitting and maintaining model performance.

### 2. **Optimizers (Adam and RMSprop)**
Optimizers are algorithms used to update the model weights based on the gradients calculated during backpropagation. Two optimizers were used:

- **Adam (Adaptive Moment Estimation)**:
  Adam combines the benefits of two other optimizers, AdaGrad (which works well with sparse gradients) and RMSprop (which works well in non-stationary environments). Adam maintains a moving average of both the gradients and the squared gradients, which helps in adaptive learning rates.
  - **Relevance**: Adam was used in Configuration 1 and 4. It tends to converge faster and works well in scenarios with noisy gradients or when parameters are updated frequently. Configuration 4 (Adam + L2) resulted in the highest accuracy (96.17%) and the lowest loss (0.1032), demonstrating Adam's ability to effectively find the optimal parameters when combined with L2 regularization.

- **RMSprop (Root Mean Square Propagation)**:
  RMSprop is an adaptive learning rate optimizer that adjusts the learning rate based on the average of recent gradients, which helps in handling the exploding or vanishing gradient problem.
  - **Relevance**: RMSprop was used in Configuration 2 and 3. It performed particularly well with L1 regularization, achieving 95.92% accuracy with a low loss of 0.1391. This shows that RMSprop, when paired with L1 regularization, was effective in preventing overfitting while maintaining high accuracy.

  **Parameter Tuning**: Both optimizers have **learning rate** as a key parameter. The learning rate controls how much the weights are adjusted during each update. A higher learning rate speeds up convergence but can overshoot the optimal point, while a lower learning rate may lead to slow convergence or getting stuck in local minima. In this project, the learning rate was tuned through trial and error, and the default values (Adam: 0.001, RMSprop: 0.001) worked well after experimentation.

### 3. **Early Stopping**
Early stopping is a regularization technique that stops training when the model's performance on a validation set stops improving. This helps in preventing overfitting since the model is stopped before it starts to learn noise from the training data.

- **Relevance**: Early stopping was applied in all Configurations and played a significant role in optimizing the model's performance by stopping training at the optimal point. For instance, Configurations 2 and 4 achieved high accuracy with relatively low loss, and early stopping ensured that these models did not overfit, even when trained for fewer epochs.

  **Parameter Tuning**: The **patience** parameter was tuned to determine how many epochs the model would wait for an improvement before stopping. A patience value of 3-5 was selected after experimentation, which allowed the model to converge to the best performance without prematurely halting training.

### 4. **Dropout**
Dropout is another regularization technique that prevents overfitting by randomly setting a fraction of input units to zero at each update during training. This forces the model to learn more robust features by not relying too heavily on any one node.

- **Relevance**: Dropout was applied in all Configurations to improve the generalization of the model. By randomly dropping units during training, dropout helped to prevent overfitting, which was particularly beneficial when combined with early stopping and regularization. For instance, the Configuration of L2 regularization with dropout in Configuration 4 led to the best overall performance.

  **Parameter Tuning**: The **dropout rate** was tuned to find the optimal level of regularization. A dropout rate of 0.2 to 0.5 is commonly used in practice, and a value of around 0.3 was found to work best in this project, offering a good tradeoff between regularization and model capacity.

### Summary of Results and Conclusion
- **Configuration 1 (L1 + Adam + Early Stopping + Dropout)**: While achieving 92.97% accuracy and a loss of 0.2125, this Configuration had decent performance, but Adam struggled slightly with L1 regularization compared to RMSprop.
  
- **Configuration 2 (L1 + RMSprop + Early Stopping + Dropout)**: This Configuration performed well with 95.92% accuracy and a loss of 0.1391. RMSprop paired better with L1 regularization, leading to better generalization and lower loss.

- **Configuration 3 (L2 + RMSprop + Early Stopping + Dropout)**: L2 regularization with RMSprop yielded good results, with 94.30% accuracy and a loss of 0.1424, but not as strong as Configuration 4.

- **Configuration 4 (L2 + Adam + Early Stopping + Dropout)**: This Configuration resulted in the best performance, with 96.17% accuracy and the lowest loss of 0.1032. The synergy between Adam and L2 regularization, along with early stopping and dropout, led to the most effective model.

Thus, the Configuration of **L2 regularization, Adam optimizer, early stopping, and dropout** was the best optimization strategy for this project. The parameter choices, including the learning rate, regularization strength, dropout rate, and early stopping patience, were tuned carefully through experimentation, leading to this optimal result.

### Error Analysis of the Two Models

#### 1. **Confusion Matrix Comparison**

- **Vanilla Model**:
  - True Organic (O) correctly classified: 122
  - True Organic misclassified as Recyclable (R): 4
  - True Recyclable correctly classified: 150
  - True Recyclable misclassified as Organic: 12

- **Optimized Model**:
  - True Organic (O) correctly classified: 122
  - True Organic misclassified as Recyclable (R): 4
  - True Recyclable correctly classified: 152
  - True Recyclable misclassified as Organic: 10

#### 2. **False Positives and False Negatives**:

- **Vanilla Model**:
  - **False Positives (FP)**: 4 (Organic predicted as Recyclable)
  - **False Negatives (FN)**: 12 (Recyclable predicted as Organic)
  
- **Optimized Model**:
  - **False Positives (FP)**: 4 (Organic predicted as Recyclable)
  - **False Negatives (FN)**: 10 (Recyclable predicted as Organic)

#### 3. **Performance Metrics**:

- **Vanilla Model**:
  - **Accuracy**: 0.9444
  - **Precision**: 0.9740
  - **Recall**: 0.9259
  - **F1 Score**: 0.9494

- **Optimized Model**:
  - **Accuracy**: 0.9514
  - **Precision**: 0.9744
  - **Recall**: 0.9383
  - **F1 Score**: 0.9560


# Python Scripts

## Model Training, retraining and evaluation
/TEST dataset was used for this too for easier handling.

**/ model training.py**

* Loading dataset
* Iterating through each class directory
* Image pre-processing
* Created a Tensorflow dataset
* Defined model parameters
* Create an optimizer instance
* Compile the model
* Train the model
* Save the model

## Model Retraining

**/ retraining.py**

A place where one can upload multiple image files and click retrain, which will return that a model is successfully retrained or throw an error if it is not retrained. 

* Loading trained dataset
* Prepare data for retraining
* Convert images to numpy arrays
* Normalize
* Check null values
* Input and output matching
* Retrain Model
* Save retrained model

## Model Evaluation

A button where one can upload an image and click evaluate, which will predict the class of the image and confidence score from predicting a retrained model

* Load and pre-process image
* Input and output matching
* Making a prediction


# APIs

## Predict API

* Link = [predict API ](https://prediction-eewt.onrender.com)


````
    POST/ predict
    Content type
    application/json
    null
    
````
````
   GET / main
   Content type
   application/json
   null

````

## Retrain and Evaluate API

* Link = [retrain API ](https://retrainapi.onrender.com)

This is how the API works:

```
    POST/ retrain
    Content type
    application/json
    null

```


````
   POST /evaluate
   Content type
   application/json
   null

````

````
  {
  "Model Retrained Successfully!":
  "Predicted Class":
  "Confidence score":
  }

````

# Website Deployment

* Install Docker desktop from their website and install it depending on your OS requirements 
  
* You can alternatively set up nginx or install from CMD

* git clone https://github.com/cynthianekesa/MLOP_Summative.git

* Navigate to the **/ webapp** directory

* Run **docker build -t waste-model .** on the terminal to build a docker container

* Run **docker run -p 800:80 waste-model** on the terminal to run the docker container so as to create a docker image

* Click on the URL created so as to access docker image on the localhost **http://127.0.0.1.800**
    * On creating a docker image the display of my website was a bit spoilt. This is something to look into next time.

* Alternatively, push the docker image to the docker hub so as to access the image url that can be shared across teams.

* Or host the docker image/ website on your cloud service of choice

* Link to hosted docker image on docker hub:  https://hub.docker.com/r/cynthianekesa/waste-model


# Locust Flooding

* Install locust

````
   pip3 install locust
   
````
* Validate Installation
````
locust -V
   
````
* Locate a file named **locustfile-predict.py** in the current directory and then run **locust**:

* Locate a file named **locustfile-retrain.py** in the current directory and the run **locust**:

* Locust will run on **http://localhost:8089** for both endpoints

* Users will begin to be spwaned

* Results of simulation are in the **locust-files** directory

![locust-frontend](https://github.com/user-attachments/assets/c4607013-ea55-42b8-ac2c-43f1c278c307)

# Contributing

Make a pull request before contributing

# License

This code was built and produced without a licence

# WOOW
