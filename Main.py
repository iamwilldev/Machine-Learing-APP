import subprocess

# List of required modules
modules = [
    'streamlit',
    'pandas',
    'matplotlib',
    'plotly',
    'seaborn',
    'scikit-learn'
]

# Install the modules
for module in modules:
    subprocess.call(['pip', 'install', module])

# Import the modules
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics


def main():
    st.sidebar.image("https://github.com/iamwilldev/iamwilldev.github.io/blob/gh-pages/_static/logo.png?raw=true")
    st.sidebar.markdown("<h1 style='text-align: center;'>Muhammad Muqtafin Nuha</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>210411100218</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>Penambangan Data C</p>", unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    
    navigation = st.sidebar.radio("Go to", ["Upload", "Data Normalization", "Data Setting", "Visualization", "Evaluate", "Implementation"])
    
    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv", index_col=None)
    
    if navigation == "Upload":
        st.title("Upload File")
        file = st.file_uploader("Upload Your Dataset Here", type=['csv', 'xlsx'])
        if file:
            if file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                df = pd.read_excel(file, engine='openpyxl')
            else:
                df = pd.read_csv(file, index_col=None)
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)
    
    elif navigation == "Data Normalization":
        st.title("Data Normalization with MinMaxScalar")
        selected_columns = st.multiselect("Select Columns for Normalization", df.columns[:-1])

        if st.button("Normalize Data"):
            scaler = MinMaxScaler()
            normalized_data = pd.DataFrame(scaler.fit_transform(df[selected_columns]), columns=selected_columns)
            
            # Add the target column back to the normalized data
            target_column = df.columns[-1]
            normalized_data[target_column] = df[target_column]

            st.dataframe(normalized_data)
            normalized_data.to_csv("normalized_data.csv", index=None)
            st.success("Normalized data saved successfully.")

    elif navigation == "Data Setting":
        st.title("Data Setting")
        if os.path.exists("normalized_data.csv"):
            normalized_data = pd.read_csv("normalized_data.csv")
            num_folds = st.sidebar.selectbox("Number of Folds", [2, 3, 5, 10, 20], key="num_folds")

            if st.button("Split Data & Save Folds"):
                X = normalized_data
                kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
                fold_count = 1
                for train_index, test_index in kf.split(X):
                    st.write(f"Fold {fold_count}")
                    fold_count += 1
                    train_data = normalized_data.iloc[train_index]
                    test_data = normalized_data.iloc[test_index]
                    train_percent = len(train_data) / len(normalized_data) * 100
                    test_percent = len(test_data) / len(normalized_data) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"Train Data: {len(train_data)} rows ({train_percent:.2f}%)")
                        st.dataframe(train_data)
                    with col2:
                        st.write(f"Test Data: {len(test_data)} rows ({test_percent:.2f}%)")
                        st.dataframe(test_data)
                    
                with open("num_folds.txt", "w") as file:
                    file.write(str(num_folds))
                st.success("Number of folds saved successfully.")
        else:
            st.warning("Please upload a dataset, perform data normalization, and save the normalized data first.")

    elif navigation == "Visualization":
        st.title("Data Visualization")
        if os.path.exists("normalized_data.csv"):
            normalized_data = pd.read_csv("normalized_data.csv")
            
            # Plot histograms for all columns
            with st.expander("Features Distribution"):
                fig, axs = plt.subplots(len(normalized_data.columns), 1, figsize=(10, 6 * len(normalized_data.columns)))
                axs = axs.flatten()

                for i, column in enumerate(normalized_data.columns):
                    axs[i].hist(normalized_data[column], bins=15)
                    axs[i].set_title(f"Features - {column}")

                fig_distplot = px.histogram(normalized_data, x=normalized_data.columns)
                st.plotly_chart(fig_distplot)
            
            # Plot pairwise correlation heatmap
            with st.expander("Pairwise Correlation Heatmap"):
                corr_matrix = normalized_data.corr()
                fig_heatmap = plt.figure(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
                st.pyplot(fig_heatmap)
            
            # Plot scatter plots for selected columns
            selected_columns_scatter = st.multiselect("Select Columns for Scatter Plot", normalized_data.columns)
            if selected_columns_scatter:
                with st.expander("Scatter Plot"):
                    fig_scatter = px.scatter(normalized_data, x=selected_columns_scatter[0], y=selected_columns_scatter[1])
                    st.plotly_chart(fig_scatter)
                
        else:
            st.warning("Please upload a dataset, perform data normalization, and save the normalized data first.")

    elif navigation == "Evaluate":
        st.title("Evaluate Accuracy Data Test and Training")
        algorithm = st.sidebar.selectbox("Select Algorithm", ["KNN", "Naive Bayes", "Decision Tree", "Random Forest", "SVM", "All"])

        if os.path.exists("normalized_data.csv"):
            normalized_data = pd.read_csv("normalized_data.csv")
            
            if algorithm == "KNN":
                st.write("You selected the KNN algorithm")
                # Kode untuk algoritma KNN
                col1, col2 = st.columns([1, 1])
                target = col1.selectbox("Select Your Target", normalized_data.columns, key="target")
                features = normalized_data.drop(target, axis=1)
                k = col2.slider("Number of Neighbors (k)", 1, 10)
                
                if st.button("Test and Score"):
                    # Read the number of folds from the file
                    with open("num_folds.txt", "r") as file:
                        num_folds = int(file.read().strip())

                    # Split the data using k-fold cross-validation
                    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                    accuracy_train_list = []
                    accuracy_test_list = []

                    for train_index, test_index in kf.split(features):
                        # Split the data into training and testing sets
                        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                        y_train, y_test = normalized_data[target].iloc[train_index], normalized_data[target].iloc[test_index]

                        # Create and fit the KNN classifier
                        knn = KNeighborsClassifier(n_neighbors=k)
                        knn.fit(X_train, y_train)

                        # Predict the target variable for the training and testing sets
                        y_train_pred = knn.predict(X_train)
                        y_test_pred = knn.predict(X_test)

                        # Calculate accuracy data
                        # Calculating and printing the accuracy for the training set
                        accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
                        accuracy_train_list.append(accuracy_train)

                        # Calculating and printing the accuracy for the test set
                        accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
                        accuracy_test_list.append(accuracy_test)

                    # Calculate the mean accuracy across all folds
                    mean_accuracy_train = round(sum(accuracy_train_list) / len(accuracy_train_list), 2)
                    mean_accuracy_test = round(sum(accuracy_test_list) / len(accuracy_test_list), 2)

                    # Display the accuracy scores
                    st.write("Mean Accuracy (Training Set):", mean_accuracy_train)
                    st.write("Mean Accuracy (Test Set):", mean_accuracy_test)
            
            elif algorithm == "Naive Bayes":
                st.write("You selected the Naive Bayes algorithm")
                # Kode untuk algoritma Naive Bayes
                col1, col2 = st.columns([1, 1])
                target = col1.selectbox("Select Your Target", normalized_data.columns, key="target")
                features = normalized_data.drop(target, axis=1)
                
                if st.button("Test and Score"):
                    # Read the number of folds from the file
                    with open("num_folds.txt", "r") as file:
                        num_folds = int(file.read().strip())

                    # Split the data using k-fold cross-validation
                    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                    accuracy_train_list = []
                    accuracy_test_list = []

                    for train_index, test_index in kf.split(features):
                        # Split the data into training and testing sets
                        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                        y_train, y_test = normalized_data[target].iloc[train_index], normalized_data[target].iloc[test_index]

                        # Create and fit the Naive Bayes
                        nb = GaussianNB()
                        nb.fit(X_train, y_train)

                        # Predict the target variable for the training and testing sets
                        y_train_pred = nb.predict(X_train)
                        y_test_pred = nb.predict(X_test)

                        # Calculate accuracy data
                        # Calculating and printing the accuracy for the training set
                        accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
                        accuracy_train_list.append(accuracy_train)

                        # Calculating and printing the accuracy for the test set
                        accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
                        accuracy_test_list.append(accuracy_test)

                    # Calculate the mean accuracy across all folds
                    mean_accuracy_train = round(sum(accuracy_train_list) / len(accuracy_train_list), 2)
                    mean_accuracy_test = round(sum(accuracy_test_list) / len(accuracy_test_list), 2)

                    # Display the accuracy scores
                    st.write("Mean Accuracy (Training Set):", mean_accuracy_train)
                    st.write("Mean Accuracy (Test Set):", mean_accuracy_test)
            
            elif algorithm == "Decision Tree":
                st.write("You selected the Decision Tree algorithm")
                # Kode untuk algoritma Decision Tree
                col1, col2 = st.columns([1, 1])
                target = col1.selectbox("Select Your Target", normalized_data.columns, key="target")
                features = normalized_data.drop(target, axis=1)
                
                if st.button("Test and Score"):
                    # Read the number of folds from the file
                    with open("num_folds.txt", "r") as file:
                        num_folds = int(file.read().strip())

                    # Split the data using k-fold cross-validation
                    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                    accuracy_train_list = []
                    accuracy_test_list = []

                    for train_index, test_index in kf.split(features):
                        # Split the data into training and testing sets
                        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                        y_train, y_test = normalized_data[target].iloc[train_index], normalized_data[target].iloc[test_index]

                        # Create and fit the Decision Tree
                        dt = DecisionTreeClassifier()
                        dt.fit(X_train, y_train)

                        # Predict the target variable for the training and testing sets
                        y_train_pred = dt.predict(X_train)
                        y_test_pred = dt.predict(X_test)

                        # Calculate accuracy data
                        # Calculating and printing the accuracy for the training set
                        accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
                        accuracy_train_list.append(accuracy_train)

                        # Calculating and printing the accuracy for the test set
                        accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
                        accuracy_test_list.append(accuracy_test)

                    # Calculate the mean accuracy across all folds
                    mean_accuracy_train = round(sum(accuracy_train_list) / len(accuracy_train_list), 2)
                    mean_accuracy_test = round(sum(accuracy_test_list) / len(accuracy_test_list), 2)

                    # Display the accuracy scores
                    st.write("Mean Accuracy (Training Set):", mean_accuracy_train)
                    st.write("Mean Accuracy (Test Set):", mean_accuracy_test)
            
            elif algorithm == "Random Forest":
                st.write("You selected the Random Forest algorithm")
                # Kode untuk algoritma Random Forest
                col1, col2 = st.columns([1, 1])
                target = col1.selectbox("Select Your Target", normalized_data.columns, key="target")
                features = normalized_data.drop(target, axis=1)
                
                if st.button("Test and Score"):
                    # Read the number of folds from the file
                    with open("num_folds.txt", "r") as file:
                        num_folds = int(file.read().strip())

                    # Split the data using k-fold cross-validation
                    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                    accuracy_train_list = []
                    accuracy_test_list = []

                    for train_index, test_index in kf.split(features):
                        # Split the data into training and testing sets
                        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                        y_train, y_test = normalized_data[target].iloc[train_index], normalized_data[target].iloc[test_index]

                        # Create and fit the Random Forest
                        rf = RandomForestClassifier()
                        rf.fit(X_train, y_train)

                        # Predict the target variable for the training and testing sets
                        y_train_pred = rf.predict(X_train)
                        y_test_pred = rf.predict(X_test)

                        # Calculate accuracy data
                        # Calculating and printing the accuracy for the training set
                        accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
                        accuracy_train_list.append(accuracy_train)

                        # Calculating and printing the accuracy for the test set
                        accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
                        accuracy_test_list.append(accuracy_test)

                    # Calculate the mean accuracy across all folds
                    mean_accuracy_train = round(sum(accuracy_train_list) / len(accuracy_train_list), 2)
                    mean_accuracy_test = round(sum(accuracy_test_list) / len(accuracy_test_list), 2)

                    # Display the accuracy scores
                    st.write("Mean Accuracy (Training Set):", mean_accuracy_train)
                    st.write("Mean Accuracy (Test Set):", mean_accuracy_test)
            
            elif algorithm == "SVM":
                st.write("You selected the SVM algorithm")
                # Kode untuk algoritma SVM
                col1, col2 = st.columns([1, 1])
                target = col1.selectbox("Select Your Target", normalized_data.columns, key="target")
                features = normalized_data.drop(target, axis=1)
                
                if st.button("Test and Score"):
                    # Read the number of folds from the file
                    with open("num_folds.txt", "r") as file:
                        num_folds = int(file.read().strip())

                    # Split the data using k-fold cross-validation
                    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                    accuracy_train_list = []
                    accuracy_test_list = []

                    for train_index, test_index in kf.split(features):
                        # Split the data into training and testing sets
                        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                        y_train, y_test = normalized_data[target].iloc[train_index], normalized_data[target].iloc[test_index]

                        # Create and fit the SVM
                        svm = SVC()
                        svm.fit(X_train, y_train)

                        # Predict the target variable for the training and testing sets
                        y_train_pred = svm.predict(X_train)
                        y_test_pred = svm.predict(X_test)

                        # Calculate accuracy data
                        # Calculating and printing the accuracy for the training set
                        accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
                        accuracy_train_list.append(accuracy_train)

                        # Calculating and printing the accuracy for the test set
                        accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
                        accuracy_test_list.append(accuracy_test)

                    # Calculate the mean accuracy across all folds
                    mean_accuracy_train = round(sum(accuracy_train_list) / len(accuracy_train_list), 2)
                    mean_accuracy_test = round(sum(accuracy_test_list) / len(accuracy_test_list), 2)
                    
                    # Display the accuracy scores
                    st.write("Mean Accuracy (Training Set):", mean_accuracy_train)
                    st.write("Mean Accuracy (Test Set):", mean_accuracy_test)
            
            elif algorithm == "All":
                st.write("You selected the All algorithm")
                col1, col2 = st.columns([1, 1])
                selected_algorithms = col1.multiselect("Select Algorithms for comparison", ["KNN", "Naive Bayes", "Decision Tree", "Random Forest", "SVM"])
                target = col2.selectbox("Select Your Target", normalized_data.columns, key="target")
                features = normalized_data.drop(target, axis=1)

                if st.button("Test and Score"):
                    if os.path.exists("normalized_data.csv"):
                        normalized_data = pd.read_csv("normalized_data.csv")

                        # Read the number of folds from the file
                        with open("num_folds.txt", "r") as file:
                            num_folds = int(file.read().strip())

                        results = {}

                        for algorithm in selected_algorithms:
                            if algorithm == "KNN":
                                # Code for KNN algorithm
                                # Split the data using k-fold cross-validation
                                kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                                accuracy_train_list = []
                                accuracy_test_list = []

                                for train_index, test_index in kf.split(features):
                                    # Split the data into training and testing sets
                                    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                                    y_train, y_test = normalized_data[target].iloc[train_index], normalized_data[target].iloc[test_index]

                                    # Create and fit the KNN classifier
                                    knn = KNeighborsClassifier(n_neighbors=5)
                                    knn.fit(X_train, y_train)

                                    # Predict the target variable for the training and testing sets
                                    y_train_pred = knn.predict(X_train)
                                    y_test_pred = knn.predict(X_test)

                                    # Calculate accuracy data
                                    # Calculating and printing the accuracy for the training set
                                    accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
                                    accuracy_train_list.append(accuracy_train)

                                    # Calculating and printing the accuracy for the test set
                                    accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
                                    accuracy_test_list.append(accuracy_test)

                                # Calculate the mean accuracy across all folds
                                mean_accuracy_train = round(sum(accuracy_train_list) / len(accuracy_train_list), 2)
                                mean_accuracy_test = round(sum(accuracy_test_list) / len(accuracy_test_list), 2)
                                pass
                            
                            elif algorithm == "Naive Bayes":
                                # Code for Naive Bayes algorithm
                                # Split the data using k-fold cross-validation
                                kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                                accuracy_train_list = []
                                accuracy_test_list = []

                                for train_index, test_index in kf.split(features):
                                    # Split the data into training and testing sets
                                    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                                    y_train, y_test = normalized_data[target].iloc[train_index], normalized_data[target].iloc[test_index]

                                    # Create and fit the Naive Bayes
                                    nb = GaussianNB()
                                    nb.fit(X_train, y_train)

                                    # Predict the target variable for the training and testing sets
                                    y_train_pred = nb.predict(X_train)
                                    y_test_pred = nb.predict(X_test)

                                    # Calculate accuracy data
                                    # Calculating and printing the accuracy for the training set
                                    accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
                                    accuracy_train_list.append(accuracy_train)

                                    # Calculating and printing the accuracy for the test set
                                    accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
                                    accuracy_test_list.append(accuracy_test)

                                # Calculate the mean accuracy across all folds
                                mean_accuracy_train = round(sum(accuracy_train_list) / len(accuracy_train_list), 2)
                                mean_accuracy_test = round(sum(accuracy_test_list) / len(accuracy_test_list), 2)
                                pass
                            
                            elif algorithm == "Decision Tree":
                                # Code for Decision Tree algorithm
                                # Split the data using k-fold cross-validation
                                kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                                accuracy_train_list = []
                                accuracy_test_list = []

                                for train_index, test_index in kf.split(features):
                                    # Split the data into training and testing sets
                                    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                                    y_train, y_test = normalized_data[target].iloc[train_index], normalized_data[target].iloc[test_index]

                                    # Create and fit the Decision Tree
                                    dt = DecisionTreeClassifier()
                                    dt.fit(X_train, y_train)

                                    # Predict the target variable for the training and testing sets
                                    y_train_pred = dt.predict(X_train)
                                    y_test_pred = dt.predict(X_test)

                                    # Calculate accuracy data
                                    # Calculating and printing the accuracy for the training set
                                    accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
                                    accuracy_train_list.append(accuracy_train)

                                    # Calculating and printing the accuracy for the test set
                                    accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
                                    accuracy_test_list.append(accuracy_test)

                                # Calculate the mean accuracy across all folds
                                mean_accuracy_train = round(sum(accuracy_train_list) / len(accuracy_train_list), 2)
                                mean_accuracy_test = round(sum(accuracy_test_list) / len(accuracy_test_list), 2)
                                pass
                            
                            elif algorithm == "Random Forest":
                                # Code for Random Forest algorithm
                                # Split the data using k-fold cross-validation
                                kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                                accuracy_train_list = []
                                accuracy_test_list = []

                                for train_index, test_index in kf.split(features):
                                    # Split the data into training and testing sets
                                    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                                    y_train, y_test = normalized_data[target].iloc[train_index], normalized_data[target].iloc[test_index]

                                    # Create and fit the Random Forest
                                    rf = RandomForestClassifier()
                                    rf.fit(X_train, y_train)

                                    # Predict the target variable for the training and testing sets
                                    y_train_pred = rf.predict(X_train)
                                    y_test_pred = rf.predict(X_test)

                                    # Calculate accuracy data
                                    # Calculating and printing the accuracy for the training set
                                    accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
                                    accuracy_train_list.append(accuracy_train)

                                    # Calculating and printing the accuracy for the test set
                                    accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
                                    accuracy_test_list.append(accuracy_test)

                                # Calculate the mean accuracy across all folds
                                mean_accuracy_train = round(sum(accuracy_train_list) / len(accuracy_train_list), 2)
                                mean_accuracy_test = round(sum(accuracy_test_list) / len(accuracy_test_list), 2)
                                pass
                            
                            elif algorithm == "SVM":
                                # Code for SVM algorithm
                                # Split the data using k-fold cross-validation
                                kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                                accuracy_train_list = []
                                accuracy_test_list = []

                                for train_index, test_index in kf.split(features):
                                    # Split the data into training and testing sets
                                    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                                    y_train, y_test = normalized_data[target].iloc[train_index], normalized_data[target].iloc[test_index]

                                    # Create and fit the SVM
                                    svm = SVC()
                                    svm.fit(X_train, y_train)

                                    # Predict the target variable for the training and testing sets
                                    y_train_pred = svm.predict(X_train)
                                    y_test_pred = svm.predict(X_test)

                                    # Calculate accuracy data
                                    # Calculating and printing the accuracy for the training set
                                    accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
                                    accuracy_train_list.append(accuracy_train)

                                    # Calculating and printing the accuracy for the test set
                                    accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
                                    accuracy_test_list.append(accuracy_test)

                                # Calculate the mean accuracy across all folds
                                mean_accuracy_train = round(sum(accuracy_train_list) / len(accuracy_train_list), 2)
                                mean_accuracy_test = round(sum(accuracy_test_list) / len(accuracy_test_list), 2)
                                pass

                            # Store the results for the current algorithm
                            results[algorithm] = {
                                "Mean Accuracy (Training Set)": mean_accuracy_train,
                                "Mean Accuracy (Test Set)": mean_accuracy_test
                            }

                        # Display the results
                        for algorithm, result in results.items():
                            st.write(f"Algorithm: {algorithm}")
                            st.write("Mean Accuracy (Training Set):", result["Mean Accuracy (Training Set)"])
                            st.write("Mean Accuracy (Test Set):", result["Mean Accuracy (Test Set)"])
                    else:
                        st.warning("Please upload a dataset, perform data normalization, and save the normalized data first.")
                
    elif navigation == "Implementation":
        st.title("Implementation")
        st.info("Prediction")
        
        if os.path.exists("normalized_data.csv"):
            normalized_data = pd.read_csv("normalized_data.csv")

        # Select algorithm
        col1, col2 = st.columns([1, 1])
        algorithm = col1.selectbox("Select Your Algorithm", ["KNN", "Naive Bayes", "Decision Tree", "Random Forest", "SVM"], key="algorithm1")

        # Select Target
        target = col2.selectbox("Select Your Target", df.columns, key="target1")

        st.info("Enter the value column")

        # Enter the value column
        selected_columns = []
        num_columns = len(normalized_data.columns[1:]) - 1  # Subtract 1 to remove the last column
        num_columns_per_col = num_columns // 2
        with st.form(key='input_form'):
            col1, col2 = st.columns(2)
            for i, column in enumerate(normalized_data.columns[0:-1]):  # Exclude the last column
                if i < num_columns_per_col:
                    with col1:
                        value = st.text_input(f"Enter value for {column}")
                else:
                    with col2:
                        value = st.text_input(f"Enter value for {column}")
                if value:
                    value = value.replace(',', '.')  # Replace comma with period
                    try:
                        selected_columns.append({
                            "column": column,
                            "value": float(value.strip()),
                        })
                    except ValueError:
                        st.error(f"Invalid value entered for {column}. Please enter a valid numeric value.")

            # Add "Check Prediction" button
            check_prediction = st.form_submit_button("Check Prediction")

        # Perform prediction when "Check Prediction" button is clicked
        if check_prediction:
            # Prepare the data for prediction
            X = normalized_data[[col['column'] for col in selected_columns]]
            y = normalized_data[target]
            
            if algorithm == "KNN":
                # Create and train the KNN classifier
                knn = KNeighborsClassifier()
                knn.fit(X, y)

                # Prepare the input data for prediction
                input_data = [[col['value'] for col in selected_columns]]

                # Perform prediction
                prediction = knn.predict(input_data)

                # Display the prediction result
                st.info(f"Prediction with KNN Classifier: {prediction}")
                
            elif algorithm == "Naive Bayes":
                # Create and train the Naive Bayes classifier
                nb = GaussianNB()
                nb.fit(X, y)

                # Prepare the input data for prediction
                input_data = [[col['value'] for col in selected_columns]]

                # Perform prediction
                prediction = nb.predict(input_data)

                # Display the prediction result
                st.info(f"Prediction with Naive Bayes Classifier: {prediction}")
            
            elif algorithm == "Decision Tree":
                # Create and train the Decision Tree classifier
                dt = DecisionTreeClassifier()
                dt.fit(X, y)

                # Prepare the input data for prediction
                input_data = [[col['value'] for col in selected_columns]]

                # Perform prediction
                prediction = dt.predict(input_data)

                # Display the prediction result
                st.info(f"Prediction with Decision Tree Classifier: {prediction}")
            
            elif algorithm == "Random Forest":
                # Create and train the Random Forest classifier
                rf = RandomForestClassifier()
                rf.fit(X, y)

                # Prepare the input data for prediction
                input_data = [[col['value'] for col in selected_columns]]

                # Perform prediction
                prediction = rf.predict(input_data)

                # Display the prediction result
                st.info(f"Prediction with Random Forest Classifier: {prediction}")
            
            elif algorithm == "SVM":
                # Create and train the SVM classifier
                svm = SVC()
                svm.fit(X, y)

                # Prepare the input data for prediction
                input_data = [[col['value'] for col in selected_columns]]

                # Perform prediction
                prediction = svm.predict(input_data)

                # Display the prediction result
                st.info(f"Prediction with SVM Classifier: {prediction}")

    st.sidebar.info("This is a Machine Learning web application for predicting data based on input data.")

if __name__ == "__main__":
    main()
