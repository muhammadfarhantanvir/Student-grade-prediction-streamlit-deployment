import streamlit as st
import pandas as pd
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib for model saving/loading
import matplotlib.pyplot as plt
# Load data function
def load_data(file):
    df = pd.read_csv(file)
    return df

# Data preprocessing function
def preprocess_data(df):
    # Your existing data preprocessing code here
    return df

# Train model function
def train_model(X_train_scaled, y_train):
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    return model_lr

# Save model function
def save_model(model, filename):
    joblib.dump(model, filename)

# Load model function
def load_model(filename):
    return joblib.load(filename)

# Main function
def main():
    st.title("Student's grade prediction")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
            # Load and preprocess data
            df = load_data(uploaded_file)
            df = preprocess_data(df)

            # Check if the required columns exist
            required_columns = ['Abgabe1', 'Abgabe2', 'Abgabe3', 'Anz_Zugriffe', 'Anz_Forum', 'Anz_Post', 'Anz_Quiz_Pruefung']
            if not all(col in df.columns for col in required_columns):
                st.warning("The selected data set does not contain the columns required for analysis. Please select another data set.")
                return

            # Checkbox to show/hide raw data
            show_raw_data = st.checkbox("Show raw data")
            if show_raw_data:
                # Display the raw data
                st.subheader("Raw data")
                st.write(df)

            # Annahme: Auswertung von Lernaufgaben basiert auf der durchschnittlichen Bewertung der Aufgaben
            df['Auswertung_Lernaufgaben'] = df[['Abgabe1', 'Abgabe2', 'Abgabe3']].mean(axis=1)

            # Annahme: Lernaktivitäten basieren auf der Summe verschiedener Aktivitäten
            df['Lernaktivitaeten'] = df[['Anz_Zugriffe', 'Anz_Forum', 'Anz_Post', 'Anz_Quiz_Pruefung']].sum(axis=1)
            # Split data and train model
            features = [
                'Auswertung_Lernaufgaben', 'Lernaktivitaeten',
                'Anz_Anmeldungen', 'Anz_Zugriffe', 'Anz_Forum', 'Anz_Post', 'Anz_Quiz_Pruefung',
                'Abgabe1_spaet', 'Abgabe2_spaet', 'Abgabe3_spaet',
                'Abgabe1_stunden', 'Abgabe2_stunden', 'Abgabe3_stunden',
                'Abgabe_mittel'
            ]
            selected_features = st.multiselect("Choose features for building the new model", features, default=features)

            # Check if at least 5 features are selected
            if len(selected_features) < 5:
                st.warning("At least five features must be selected for training the model.")
                return

            # Exclude selected features
            
            target = 'Abschlussnote'

            X = df[selected_features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Train the model
            model_lr = train_model(X_train_scaled, y_train)

            # Save the model
            save_model(model_lr, "linear_regression_model.joblib")

            # Quality of the trained Model
            st.subheader("Quality of the model")
            y_pred = model_lr.predict(scaler.transform(X_test))
            mse_lr = mean_squared_error(y_test, y_pred)
            st.write(f'Mean Squared Error (Linear Regression): {mse_lr:.3f}')

            # Button to show Mean Squared Error explanation
            if st.checkbox("What is Mean Squared Error?"):
                st.write("Mean Squared Error (MSE) measures the average of the squares of the errors, i.e. H. the average squared difference between the estimated values and the actual value. It is a common metric for evaluating the performance of regression models.")

            # Filter for Each Student
            st.subheader("Make a prediction")
            student_list = df['Student_ID'].unique()
            selected_student = st.selectbox("Select a Student ID", student_list)

            # Filter data for the selected student
            filtered_data = df[df['Student_ID'] == selected_student]
            
            # Checkbox to show/hide filtered data
            show_filtered_data = st.checkbox("Show filtered data")
            if show_filtered_data:
                # Display filtered data
                st.write(filtered_data)

            # Predictions for the selected student
            X_selected = filtered_data[selected_features]
            y_selected_pred = model_lr.predict(scaler.transform(X_selected))

            # Display predictions with a bigger font
            prediction_text = f"<h2>Expected final grade for {selected_student}: {y_selected_pred[0]:.3f}</h2>"
            st.write(prediction_text, unsafe_allow_html=True)


            if st.checkbox("Show histograms"):
                # Plot histograms for all numeric variables
                st.subheader("Histograms for numerical variables")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                numeric_cols = df.select_dtypes(include='number').columns
                for col in numeric_cols:
                    plt.hist(df[col], bins=20, color='blue', alpha=0.7)
                    plt.title(f'Histogram for {col}')
                    plt.xlabel(col)
                    plt.ylabel('frequency')
                    st.pyplot()


        

        

    

if __name__ == "__main__":
    main()
