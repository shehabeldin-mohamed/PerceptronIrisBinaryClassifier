# Perceptron Iris Binary Classifier

A simple **Perceptron** implementation in Python that classifies iris flowers into two species:
- *Iris Versicolor*
- *Iris Virginica*

The project uses a **binary subset** of the classic **Iris dataset**, excluding *Iris Setosa*.

---

## How It Works

The perceptron is a **linear classifier** that learns to separate two classes using a weight vector and bias.  
During training, it updates weights based on misclassified examples using the **Perceptron Learning Rule**.

You can easily modify parameters such as:
- Learning rate  
- Number of epochs  
- Training and testing dataset files  

---

## Installation Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/shehabeldin-mohamed/PerceptronIrisBinaryClassifier.git
   cd PerceptronIrisBinaryClassifier
2. **Move to src and run the program by specifying the learning rate, training dataset, and test dataset
   ```bash
   cd src
   python Main.py 0.01 perceptron.data perceptron.test.data
