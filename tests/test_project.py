import project as pr 
import time
import pandas as pd
import numpy as np
import pytest

excel_file = "train.xlsx"
df = pd.read_excel(excel_file)

# Test case to ensure the correct loading of data
def test_load_data():
    data = pr.loadData()
    startTime = time.time()
    assert isinstance(data, pd.DataFrame), "Loaded data should be a DataFrame."

    expected_columns = ["User_ID", "Product_ID", "Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status", "Product_Category_1", "Product_Category_2", "Product_Category_3", "Purchase"]

    for col in expected_columns:
        assert col in data.columns, f"Expected column {col} not found in loaded data."
    endTime = time.time()
    print(endTime - startTime)


# Test case to ensure the correct shape of data
def test_shapeData():
    input_df, output_df, scaled_input_df,scaled_output_df, scalerInput, scalerOutput = pr.preprocessData()

    
    startTime = time.time()
    
    assert scaled_input_df.shape[1] == 5, "Expected 5 features in the X data after preprocessing."
    assert scaled_output_df.shape[1] == 1, "Expected Y data to have a single column."
    endTime = time.time()
    print(endTime - startTime)


# Test case to ensure the correct columns for Input
def test_colInput():
    input_df, output_df, scaled_input_df,scaled_output_df, scalerInput, scalerOutput = pr.preprocessData()
    
    startTime = time.time()
    
    input_columns = ["Gender", "Age", "Occupation", "City_Category", "Marital_Status"]
    # Convert the NumPy array to a DataFrame
    X_df = pd.DataFrame(input_df, columns=input_columns)
    # Check if X_df is a DataFrame
    assert isinstance(X_df, pd.DataFrame)
    # Check that the columns have been dropped for X
    assert "User_ID" not in X_df.columns
    assert "Product_ID" not in X_df.columns
    assert "Stay_In_Current_City_Years" not in X_df.columns
    assert "Product_Category_1" not in X_df.columns
    assert "Product_Category_2" not in X_df.columns
    assert "Product_Category_3" not in X_df.columns
    assert "Purchase" not in X_df.columns    
    
    endTime = time.time()
    print(endTime - startTime)
    
    
# Test case to ensure the correct columns for Output
def test_colOutput():
    input_df, output_df, scaled_input_df,scaled_output_df, scalerInput, scalerOutput = pr.preprocessData()
    
    startTime = time.time()
    # Convert the NumPy array to a DataFrame
    # Allows to use coloumns, numpy array doesn't have coloumns
    output_df = pd.DataFrame(output_df, columns=['Car Purchase Amount'])
    
    # Check if Y_df is a DataFrame and has the correct column name
    assert isinstance(output_df, pd.DataFrame)
    assert output_df.columns == 'Car Purchase Amount'
    
    endTime = time.time()
    print(endTime - startTime)
    

# Test case to ensure the correct range of data
def test_rangeData():
    input_df, output_df, scaled_input_df,scaled_output_df, scalerInput, scalerOutput = pr.preprocessData()
    x_train, x_test, y_train, y_test = pr.splitData()
    
    startTime = time.time()
    range_input_result = len(input_df)
    range_output_result = len(output_df)
    expected_range = len(df)
    
    assert range_input_result and range_output_result == expected_range
    
    assert 0 <= np.min(scaled_input_df) <= 1, "X min should be scaled between 0 and 1."
    assert 0 <= np.min(scaled_output_df) <= 1, "Y min should be scaled between 0 and 1."
    assert 0 <= np.max(scaled_input_df) <= 1, "X max should be scaled between 0 and 1."
    assert 0 <= np.max(scaled_output_df) <= 1, "Y max should be scaled between 0 and 1."
    
    endTime = time.time()
    print(endTime - startTime)


# Test case to ensure the correct splitting of data
def test_splitData():

    x_train, x_test, y_train, y_test = pr.splitData()
    input_df, output_df, scaled_input_df,scaled_output_df, scalerInput, scalerOutput = pr.preprocessData()
    
    startTime = time.time()
    
    size_result = len(np.concatenate((x_train, x_test)))
    expected_size = len(df)
    assert size_result == expected_size
    
    # Check proportions for train-test split
    # Check proportions for train-test split
    assert x_train.shape[0] / scaled_input_df.shape[0] == pytest.approx(0.8, 0.1)
    assert x_test.shape[0] / scaled_input_df.shape[0] == pytest.approx(0.2, 0.1)
    assert y_train.shape[0] / scaled_output_df.shape[0] == pytest.approx(0.8, 0.1)
    assert y_test.shape[0] / scaled_output_df.shape[0] == pytest.approx(0.2, 0.1)
    
    endTime = time.time()
    
    print(endTime - startTime)
