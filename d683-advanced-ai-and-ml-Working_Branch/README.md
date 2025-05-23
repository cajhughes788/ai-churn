Customer Churn Prediction Project

1. Requirements

Software:

-Python 3.11
-Libraries: pandas, scikit-learn
(Task 1 listed more libraries, but only ended up using these two)

Hardware:

-Dell Latitude 7400 laptop 
-Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz 
-16 GB RAM 

Development Environment:
-PyCharm Community Edition
-Windows 11 Pro (64-bit)


2. Instructions to Run the Application

A. Clone the GitLab repository to your local machine.
B. Open the project in PyCharm.
C. Install required libraries:
D. Run `preprocess.py` to preprocess the raw dataset and create `preprocessed_data.csv`.
E. Run `train_model.py` to:
- Build the model
- Train the model
- Evaluate model accuracy, precision, recall, and F1 score
- Apply 5-fold cross-validation
- Tune hyperparameters using GridSearchCV

(All outputs will be printed to the terminal after running `train_model.py`.)


Note

Model building (B2) and training (B3) steps were performed together in a single script and commit for simplicity. Both are included and properly documented in the `train_model.py` file.
