# Spark Decision Tree Hiring Predictor

A simple Apache Spark MLlib application that uses decision tree classification to predict hiring decisions based on candidate attributes.

## Overview

This project demonstrates how to use **Apache Spark's MLlib** to build a decision tree classifier that predicts whether a job candidate will be hired based on their profile characteristics such as experience, education, and employment history.

## Dataset

The model is trained on historical hiring data (`PastHires.csv`) containing the following features:

| Feature | Description | Values |
|---------|-------------|---------|
| Years Experience | Number of years of work experience | Integer |
| Employed? | Currently employed status | Y/N |
| Previous employers | Number of previous employers | Integer |
| Level of Education | Highest degree obtained | BS/MS/PhD |
| Top-tier school | Graduated from top-tier school | Y/N |
| Interned | Completed internship | Y/N |
| Hired | Target variable - was candidate hired | Y/N |

## Requirements

- Apache Spark 2.x or higher
- Python 3.x
- PySpark
- NumPy

## Installation & Setup

```bash
# Install required packages
pip install pyspark numpy

# Ensure Spark is properly configured
export SPARK_HOME=/path/to/spark
export PATH=$SPARK_HOME/bin:$PATH
```

## Usage

1. Place `PastHires.csv` in the same directory as the script
2. Run the decision tree classifier:

```bash
spark-submit SparkDecisionTree.py
```

## Output

The program will output:
- **Prediction** for the test candidate (hired/not hired)
- **Decision tree model** structure showing the learned classification rules

## Sample Output

```
Hire prediction:
1.0
Learned classification tree model:
DecisionTreeModel classifier of depth 4 with 9 nodes
  If (feature 1 in {0.0})
   If (feature 5 in {0.0})
    If (feature 0 <= 0.5)
     If (feature 3 in {1.0})
      Predict: 0.0
     Else (feature 3 not in {1.0})
      Predict: 1.0
    Else (feature 0 > 0.5)
     Predict: 0.0
   Else (feature 5 not in {0.0})
    Predict: 1.0
  Else (feature 1 not in {0.0})
   Predict: 1.0
```

## Model Parameters

- **Algorithm**: Decision Tree Classifier
- **Impurity Measure**: Gini
- **Max Depth**: 5
- **Max Bins**: 32
- **Categorical Features**: Employed, Education Level, Top-tier School, Interned

## Code Structure

The main script (`SparkDecisionTree.py`) includes:

1. **Data preprocessing functions** for converting categorical data to numerical format
2. **Feature engineering** to create MLlib-compatible data structures
3. **Model training** using Spark's DecisionTree classifier
4. **Prediction** on test candidates
5. **Model interpretation** showing the learned decision rules

## Test Case

The script includes a test candidate with the following profile:
- 10 years experience
- Currently employed
- 3 previous employers
- BS degree
- Not from top-tier school  
- No internship experience

## Contributing

Feel free to fork this project and submit pull requests for improvements such as:
- Additional feature engineering
- Model evaluation metrics
- Cross-validation
- Different algorithms comparison

## License

This project is open source and available under the MIT License.
