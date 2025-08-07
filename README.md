#LEARNING 

 Regression Modeling, Evaluation Metrics, Model Interpretation
 
 PYTHON lIBRARIES:
 
 Pandas: 
 
 Pandas is a Python library used to work with data in tables (like Excel sheets).It lets you load, clean, organize, and explore data easily using rows and columns.Without Pandas, working with real-world datasets would be slow and messy.
    
    
 Scikit-learn (sklearn):
 
Scikit-learn is a powerful machine learning library in Python.It provides ready-made tools to build, train, and test models like Linear Regression.Instead of doing complex calulations , Scikit-learn lets the computer do it with simple commands.   
    
    
Matplotlib :

Matplotlib is a library used to draw graphs and charts.It helps visualize data and predictions, making it easier to understand trends and model performance.


CONCEPT BUILDING 

        | Concept              | Explanation                                            |
        | -------------------- | ------------------------------------------------------ |
        | `import`             | Brings in external libraries (like Pandas, Sklearn)    |
        | `DataFrame`          | Table-like structure from Pandas                       |
        | `function()`         | Pre-built code that does a specific task               |
        | `variables`          | Names to store data                                    |
        | `list` and `dict`    | Built-in Python structures for storing multiple values |
        | `train_test_split()` | Function that randomly splits data                     |
        | `fit()`              | Trains the model on the training data                  |
        | `predict()`          | Predicts results using the trained model               |
        | `print()`            | Displays output                                        |
        | `plot()`             | Draws graphs using Matplotlib                          |
        
            
    
    
             FLOW CHART 

    
              Start
                ↓
          Read CSV File ("Housing.csv")
                ↓
       Apply One-Hot Encoding (get_dummies)
                ↓
      Display Head and Columns of Dataset
                ↓
      Select Features (X) and Target (y)
                ↓
      Split Data (Train/Test)
                ↓
       Train Linear Regression Model
                ↓
        Predict on Test Data
                ↓
      Evaluate Model (MAE, MSE, R²)
                ↓
       Plot Actual vs Predicted Graph
                ↓
               End
