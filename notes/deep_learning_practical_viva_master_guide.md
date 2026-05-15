# Deep Learning Practical Examination + Viva Master Guide

Prepared for: Shashank

This guide is based on your uploaded practical codes:
1. Boston Housing Price Prediction using ANN
2. IMDB Sentiment Analysis using Embedding + Dense Network
3. Fashion-MNIST Image Classification using ANN
4. Google Stock Price Prediction using LSTM and GRU

It includes:
- Practical explanation of every code
- Theory behind every concept
- Most probable viva questions
- Detailed answers
- Important formulas
- Common mistakes
- External viva traps
- Deep Learning theory questions

---

# PRACTICAL 1: BOSTON HOUSING PRICE PREDICTION USING ANN

## What the Practical Does
This practical predicts house prices using an Artificial Neural Network (ANN).

Dataset: Boston Housing Dataset
Target Column: MEDV (Median house value)
Type of Problem: Regression

---

# Important Concepts

## What is Regression?
Regression predicts continuous numerical values.

Examples:
- House price prediction
- Temperature prediction
- Stock price prediction

In this practical, the output is a numeric value (house price), therefore it is a regression problem.

---

# Important Viva Questions and Answers

## Q1. Why is this a regression problem and not classification?
Because the output is continuous numerical data.

Example:
- Classification output: Cat or Dog
- Regression output: 25.6 lakhs house price

Since MEDV contains continuous values, regression is used.

---

## Q2. What is ANN?
ANN (Artificial Neural Network) is a computational model inspired by the human brain.

It consists of:
- Input layer
- Hidden layers
- Output layer

Each neuron performs:
1. Weighted sum
2. Activation function
3. Produces output

---

## Q3. What is a neuron?
A neuron is the basic processing unit of a neural network.

Mathematically:

genui{"math_block_widget_always_prefetch_v2":{"content":"y=f(\\sum wx+b)"}}

Where:
- x = input
- w = weight
- b = bias
- f = activation function

---

## Q4. Why do we split data into training and testing?
Training data is used to train the model.
Testing data is used to evaluate performance on unseen data.

This helps check generalization.

Common split:
- 80% training
- 20% testing

---

## Q5. Why is StandardScaler used?
StandardScaler normalizes data.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"z=\\frac{x-\\mu}{\\sigma}"}}

Benefits:
- Faster convergence
- Better training stability
- Prevents large-value dominance

---

## Q6. What is an activation function?
An activation function decides whether a neuron should activate.

Without activation functions, neural networks become linear.

Common activation functions:
- ReLU
- Sigmoid
- Tanh
- Softmax

---

## Q7. Why is ReLU commonly used?
ReLU stands for Rectified Linear Unit.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"f(x)=\\max(0,x)"}}

Advantages:
- Faster training
- Reduces vanishing gradient problem
- Computationally efficient

---

## Q8. What is backpropagation?
Backpropagation is the process of updating weights using gradients.

Steps:
1. Forward propagation
2. Calculate loss
3. Compute gradients
4. Update weights

It uses gradient descent.

---

## Q9. What is gradient descent?
Gradient descent is an optimization algorithm used to minimize loss.

Update rule:

genui{"math_block_widget_always_prefetch_v2":{"content":"w_{new}=w_{old}-\\eta\\frac{\\partial L}{\\partial w}"}}

Where:
- η = learning rate
- L = loss function

---

## Q10. What is loss function?
Loss function measures prediction error.

For regression:
- Mean Squared Error (MSE)

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"MSE=\\frac{1}{n}\\sum(y-\\hat{y})^2"}}

---

## Q11. Why use MSE in regression?
Because it heavily penalizes large errors.

Good for continuous value prediction.

---

## Q12. What is epoch?
One complete pass of the entire training dataset through the model.

---

## Q13. What is batch size?
Number of samples processed before weight update.

Example:
- Dataset = 1000 samples
- Batch size = 100
- 10 iterations per epoch

---

## Q14. What is overfitting?
Model performs well on training data but poorly on testing data.

Causes:
- Too many epochs
- Complex model
- Small dataset

Solutions:
- Dropout
- Early stopping
- Regularization

---

## Q15. What is underfitting?
Model fails to learn training data patterns.

Causes:
- Very simple model
- Insufficient training

---

## Q16. Difference between machine learning and deep learning?

Machine Learning:
- Manual feature extraction
- Works well on smaller data

Deep Learning:
- Automatic feature extraction
- Uses neural networks
- Requires more data and computation

---

## Q17. Why use Dense layers?
Dense layer means every neuron is connected to every neuron in next layer.

Used for:
- General pattern learning
- ANN models

---

## Q18. What optimizer is used?
Adam optimizer.

Adam combines:
- Momentum
- RMSProp

Advantages:
- Fast convergence
- Adaptive learning rate

---

## Q19. Why is linear activation used in output layer for regression?
Because output can be any continuous value.

Sigmoid or softmax restrict output range.

---

## Q20. What evaluation metrics are used in regression?
- MSE
- RMSE
- MAE
- R² Score

---

# PRACTICAL 2: IMDB SENTIMENT ANALYSIS USING EMBEDDING

## What the Practical Does
This practical predicts whether a movie review is positive or negative.

Dataset: IMDB Reviews
Problem Type: Binary Classification

---

# Important Concepts

## What is NLP?
NLP (Natural Language Processing) enables computers to understand human language.

Examples:
- Chatbots
- Translation
- Sentiment analysis
- Voice assistants

---

## Q1. What is sentiment analysis?
Sentiment analysis identifies emotional tone from text.

Outputs:
- Positive
- Negative
- Neutral

---

## Q2. Why is this binary classification?
Because there are only two classes:
- Positive
- Negative

---

## Q3. What is tokenization?
Tokenization converts text into smaller units called tokens.

Example:
"I love AI"
→ ["I", "love", "AI"]

---

## Q4. What is vocabulary?
Vocabulary is the collection of unique words in dataset.

---

## Q5. What is padding?
Padding makes all sequences same length.

Example:
[1,2,3]
→ [0,0,1,2,3]

Used because neural networks require fixed-size input.

---

## Q6. What is word embedding?
Embedding converts words into dense vectors.

Example:
"king" and "queen" have similar vectors.

Advantages:
- Captures semantic meaning
- Reduces dimensionality

---

## Q7. Why use Embedding layer?
Embedding layer learns word representations automatically.

Better than:
- One-hot encoding
- Bag of words

---

## Q8. Difference between one-hot encoding and embeddings?

One-hot Encoding:
- Sparse vectors
- No semantic meaning

Embeddings:
- Dense vectors
- Semantic relationships preserved

---

## Q9. Why use binary_crossentropy?
Used for binary classification.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"L=-(y\\log(\\hat{y})+(1-y)\\log(1-\\hat{y}))"}}

---

## Q10. Why use sigmoid activation in output layer?
Sigmoid gives output between 0 and 1.

Useful for probability.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"\\sigma(x)=\\frac{1}{1+e^{-x}}"}}

---

## Q11. What is LabelEncoder?
Converts categorical labels into numerical form.

Example:
positive → 1
negative → 0

---

## Q12. What is confusion matrix?
Confusion matrix shows prediction performance.

Contains:
- True Positive
- True Negative
- False Positive
- False Negative

---

## Q13. What is precision?
Precision measures correctness of positive predictions.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"Precision=\\frac{TP}{TP+FP}"}}

---

## Q14. What is recall?
Recall measures ability to identify actual positives.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"Recall=\\frac{TP}{TP+FN}"}}

---

## Q15. What is F1-score?
Harmonic mean of precision and recall.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"F1=2\\times\\frac{Precision\\times Recall}{Precision+Recall}"}}

---

## Q16. What is overfitting in NLP?
Model memorizes training text instead of learning patterns.

---

## Q17. What is Bag of Words?
Represents text using word frequency.

Disadvantage:
- Ignores word order
- No semantic understanding

---

## Q18. Difference between stemming and lemmatization?

Stemming:
- Removes suffixes
- Faster
- Less accurate

Example:
playing → play
studies → studi

Lemmatization:
- Converts to root dictionary form
- More accurate

Example:
studies → study

---

## Q19. What is stop word?
Common words removed during preprocessing.

Examples:
- is
- the
- and

---

## Q20. What is sequence length?
Maximum number of words per review.

---

# PRACTICAL 3: FASHION-MNIST IMAGE CLASSIFICATION USING ANN

## What the Practical Does
This practical classifies fashion images.

Classes include:
- Shirt
- Sneaker
- Bag
- Dress
- Sandal

Dataset: Fashion-MNIST

Problem Type: Multi-class Classification

---

# Important Viva Questions

## Q1. What is image classification?
Image classification identifies object category in image.

---

## Q2. Why flatten images?
Images are 2D matrices.

Neural networks require 1D input.

Example:
28×28 image → 784 vector

---

## Q3. What is categorical_crossentropy?
Used for multi-class classification.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"L=-\\sum y_i\\log(\\hat{y_i})"}}

---

## Q4. Why use softmax activation?
Softmax converts outputs into probability distribution.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"Softmax(x_i)=\\frac{e^{x_i}}{\\sum_j e^{x_j}}"}}

---

## Q5. Difference between sigmoid and softmax?

Sigmoid:
- Binary classification
- Independent probabilities

Softmax:
- Multi-class classification
- Sum of probabilities = 1

---

## Q6. What is one-hot encoding?
Converts labels into binary vectors.

Example:
Class 2 in 5 classes:
[0,0,1,0,0]

---

## Q7. Why normalize image pixel values?
Pixel range:
0–255

Normalized:
0–1

Benefits:
- Faster learning
- Stable gradients

---

## Q8. What are hidden layers?
Layers between input and output layers.

They learn features.

---

## Q9. What is dropout?
Dropout randomly disables neurons during training.

Purpose:
- Prevent overfitting

---

## Q10. Why use confusion matrix?
To understand class-wise performance.

---

## Q11. What is validation split?
Part of training data reserved for validation.

Used to monitor overfitting.

---

## Q12. What is accuracy?
Percentage of correct predictions.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"Accuracy=\\frac{Correct Predictions}{Total Predictions}"}}

---

## Q13. What is CNN?
CNN (Convolutional Neural Network) is specialized for image processing.

Advantages over ANN:
- Better feature extraction
- Fewer parameters
- Spatial understanding

---

## Q14. Why is CNN better than ANN for images?
ANN ignores spatial relationships.
CNN preserves local features.

---

## Q15. What is convolution?
Convolution applies filters to extract features.

Examples:
- Edges
- Shapes
- Textures

---

## Q16. What is pooling?
Pooling reduces image dimensions.

Types:
- Max pooling
- Average pooling

---

## Q17. What is feature extraction?
Detecting important patterns from data.

---

## Q18. What is data augmentation?
Generating modified training images.

Examples:
- Rotation
- Flip
- Zoom

Purpose:
- Improve generalization

---

# PRACTICAL 4: GOOGLE STOCK PRICE PREDICTION USING LSTM AND GRU

## What the Practical Does
Predicts future stock prices using sequential data.

Models Used:
- LSTM
- GRU

Problem Type:
Time Series Forecasting

---

# Important Viva Questions

## Q1. Why use LSTM for stock prediction?
LSTM handles sequential dependencies and remembers past information.

Useful for:
- Time series
- NLP
- Speech recognition

---

## Q2. What is RNN?
RNN (Recurrent Neural Network) processes sequential data.

Output depends on previous inputs.

---

## Q3. Why traditional ANN fails for sequence data?
ANN has no memory.

RNN remembers previous information.

---

## Q4. What is vanishing gradient problem?
Gradients become extremely small during backpropagation.

Result:
- Network fails to learn long-term dependencies.

---

## Q5. How does LSTM solve vanishing gradient?
Using memory cells and gates.

---

## Q6. What are gates in LSTM?
Three gates:
1. Forget gate
2. Input gate
3. Output gate

They control information flow.

---

## Q7. What is GRU?
GRU (Gated Recurrent Unit) is simplified version of LSTM.

Advantages:
- Faster training
- Fewer parameters

---

## Q8. Difference between LSTM and GRU?

LSTM:
- More complex
- Better for long sequences
- More parameters

GRU:
- Simpler
- Faster
- Less memory

---

## Q9. Why use MinMaxScaler?
Scales data between 0 and 1.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"x' = \\frac{x-x_{min}}{x_{max}-x_{min}}"}}

---

## Q10. What is time series data?
Data collected over time.

Examples:
- Stock prices
- Weather data
- Sensor readings

---

## Q11. Why reshape data for LSTM?
LSTM expects 3D input:

(samples, timesteps, features)

---

## Q12. What is timestep?
Number of previous observations used for prediction.

---

## Q13. Why use return_sequences=True?
Returns output for every timestep.

Needed when stacking recurrent layers.

---

## Q14. What is sequence modeling?
Learning patterns from ordered data.

---

## Q15. What is RMSE?
Root Mean Squared Error.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"RMSE=\\sqrt{\\frac{1}{n}\\sum(y-\\hat{y})^2}"}}

---

## Q16. What is MAE?
Mean Absolute Error.

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"MAE=\\frac{1}{n}\\sum|y-\\hat{y}|"}}

---

## Q17. Why use dropout in LSTM?
To reduce overfitting.

---

## Q18. What is sequence dependency?
Future values depend on previous values.

---

## Q19. Can LSTM be used in NLP?
Yes.

Applications:
- Translation
- Chatbots
- Speech recognition

---

# VERY IMPORTANT THEORY QUESTIONS FOR EXTERNAL VIVA

## Q1. What is Deep Learning?
Deep learning is a subset of machine learning using multi-layer neural networks to automatically learn features from data.

---

## Q2. Difference between AI, ML, and DL?

AI:
Machines mimicking intelligence.

ML:
Machines learning from data.

DL:
Subset of ML using deep neural networks.

---

## Q3. What are weights and biases?
Weights determine importance of inputs.
Bias shifts activation function.

---

## Q4. What is forward propagation?
Process of passing inputs through network to produce output.

---

## Q5. What is learning rate?
Controls step size during weight updates.

Small learning rate:
- Slow training

Large learning rate:
- Overshooting

---

## Q6. What is exploding gradient?
Gradients become extremely large.

Causes unstable training.

---

## Q7. What is TensorFlow?
Open-source deep learning framework developed by entity["company","Google","Mountain View, CA, USA"].

---

## Q8. What is Keras?
High-level API for building deep learning models.

Runs on TensorFlow.

---

## Q9. What is GPU acceleration?
GPUs perform parallel computation.

Deep learning training becomes faster.

---

## Q10. What is hyperparameter?
Parameters set before training.

Examples:
- Learning rate
- Epochs
- Batch size

---

## Q11. What is model compilation?
Defines:
- Optimizer
- Loss function
- Metrics

---

## Q12. What is model.fit()?
Used for training model.

---

## Q13. What is model.evaluate()?
Used for testing model performance.

---

## Q14. What is model.predict()?
Generates predictions from trained model.

---

## Q15. What is transfer learning?
Using pre-trained model for new task.

---

## Q16. What is fine-tuning?
Adjusting pre-trained model weights.

---

## Q17. What is precision vs recall?
Precision:
How many predicted positives are correct.

Recall:
How many actual positives were found.

---

## Q18. What is ROC curve?
Graph between:
- True Positive Rate
- False Positive Rate

---

## Q19. What is AUC?
Area under ROC curve.

Higher AUC means better model.

---

## Q20. Why deep learning needs large data?
Because many parameters must be learned.

---

# COMMON EXTERNAL VIVA TRAPS

## Trap 1: “Why not use machine learning instead of deep learning?”
Answer:
Deep learning automatically extracts features and performs better on unstructured data like text, images, and sequences.

---

## Trap 2: “What happens if learning rate is extremely high?”
Answer:
Model overshoots minima and training becomes unstable.

---

## Trap 3: “What happens if epochs are too high?”
Answer:
Overfitting may occur.

---

## Trap 4: “Why use softmax instead of sigmoid in Fashion-MNIST?”
Answer:
Because Fashion-MNIST is multi-class classification.

---

## Trap 5: “Why use sigmoid in IMDB sentiment analysis?”
Answer:
Because output has only two classes.

---

# IMPORTANT DIFFERENCES TABLE

| Concept | Regression | Classification |
|---|---|---|
| Output | Continuous | Categorical |
| Example | House price | Cat/Dog |
| Loss | MSE | Crossentropy |

---

| Concept | LSTM | GRU |
|---|---|---|
| Complexity | High | Lower |
| Parameters | More | Fewer |
| Speed | Slower | Faster |

---

| Concept | Sigmoid | Softmax |
|---|---|---|
| Use | Binary | Multi-class |
| Output Range | 0–1 | Probabilities sum to 1 |

---

# LAST MINUTE QUICK REVISION

Remember these:

## Regression
- Continuous output
- MSE loss
- Linear activation

## Binary Classification
- 2 classes
- Sigmoid
- Binary crossentropy

## Multi-class Classification
- Multiple classes
- Softmax
- Categorical crossentropy

## LSTM
- Sequential data
- Memory cells
- Handles long dependencies

## Embedding Layer
- Converts words to vectors
- Learns semantic meaning

## CNN
- Best for images
- Uses convolution filters

---

# MOST IMPORTANT FORMULAS

## ReLU

genui{"math_block_widget_always_prefetch_v2":{"content":"f(x)=\\max(0,x)"}}

## Sigmoid

genui{"math_block_widget_always_prefetch_v2":{"content":"\\sigma(x)=\\frac{1}{1+e^{-x}}"}}

## Softmax

genui{"math_block_widget_always_prefetch_v2":{"content":"Softmax(x_i)=\\frac{e^{x_i}}{\\sum_j e^{x_j}}"}}

## MSE

genui{"math_block_widget_always_prefetch_v2":{"content":"MSE=\\frac{1}{n}\\sum(y-\\hat{y})^2"}}

## RMSE

genui{"math_block_widget_always_prefetch_v2":{"content":"RMSE=\\sqrt{\\frac{1}{n}\\sum(y-\\hat{y})^2}"}}

## MAE

genui{"math_block_widget_always_prefetch_v2":{"content":"MAE=\\frac{1}{n}\\sum|y-\\hat{y}|"}}

---

# FINAL EXAM TIPS

1. First explain the type of problem:
   - Regression
   - Binary classification
   - Multi-class classification
   - Time series forecasting

2. Always explain:
   - Dataset
   - Preprocessing
   - Model architecture
   - Activation functions
   - Loss function
   - Optimizer
   - Evaluation metrics

3. If external asks “Why this activation?” always connect it to problem type.

4. If external asks “Why scaling?” answer:
   - Faster convergence
   - Stable training
   - Better gradient flow

5. Speak confidently even if answer is short.

6. If you forget exact formula, explain concept clearly.

7. Most externals focus on:
   - Activation functions
   - Loss functions
   - Optimizers
   - Difference between ANN/CNN/RNN/LSTM
   - Overfitting
   - Preprocessing

---

# SUPER IMPORTANT ONE-LINE ANSWERS

## What is deep learning?
Deep learning uses multi-layer neural networks for automatic feature learning.

## What is ANN?
A neural network inspired by the human brain.

## What is CNN?
A neural network specialized for image processing.

## What is RNN?
A neural network specialized for sequential data.

## What is LSTM?
An advanced RNN that remembers long-term dependencies.

## What is GRU?
A simplified and faster version of LSTM.

## What is overfitting?
Model memorizes training data and performs poorly on unseen data.

## What is dropout?
Technique to randomly disable neurons during training.

## What is epoch?
One complete pass through training data.

## What is batch size?
Number of samples processed before updating weights.

## What is optimizer?
Algorithm used to minimize loss.

## What is Adam?
Adaptive optimizer combining Momentum and RMSProp.

---

End of Guide.

