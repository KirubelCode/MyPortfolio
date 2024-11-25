# Produced by: Kirubel Temesgen
# College ID: C00260396
# Date: 10/11/2024
# Description: To use linear aggression to predict employee salaries
#              given the business sector and one other user-selected parameter, 
#              i.e region, business size.


from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import io
import base64

app = Flask(__name__)

# Route to get column names from uploaded CSV
@app.route('/columns', methods=['POST'])
def columns():
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    df = pd.read_csv(file)

    # Prepare the data
    df.columns = df.columns.str.strip()
    return jsonify(df.columns.tolist())

# Route to get unique employee sizes from the selected column
@app.route('/employee_sizes', methods=['POST'])
def employee_sizes():
    column = request.args.get('column')
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    df = pd.read_csv(file)
    employee_sizes = df[column].dropna().unique().tolist()
    return jsonify(employee_sizes)

# Main page route
@app.route('/')
def index():
    return render_template('indexs.html')

@app.route('/error')
def error_page():
    return render_template('low_score.html', message="An error occurred. Please try again.")
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the uploaded file
        file = request.files['file']
    
        # Read the file into a DataFrame
        df = pd.read_csv(file)

        # Get user input for required fields
        salary_column = request.form.get('salary_column')
        sector_column = request.form.get('sector_column')

        # Ensure required fields are provided
        if not salary_column or not sector_column:
            return render_template('error.html', message="Salary Column and Sector Column are required.")

        # Optional filtering by employee size
        employee_size_column = request.form.get('employee_size_column')
        employee_size = request.form.get('employee_size')

        if employee_size_column and employee_size:
            df = df[df[employee_size_column] == employee_size]

        # Data cleaning/manipulation
        df[salary_column] = pd.to_numeric(df[salary_column], errors='coerce')
        df = df[df[salary_column] > 0].dropna()

        # One-hot encoding of the sector column
        df_encoded = pd.get_dummies(df[[sector_column]], drop_first=True)

        # Define features and target
        X = df_encoded
        y = df[salary_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Handle sectors for plotting
        _, _, _, _, _, sectors_test = train_test_split(
            X, y, df[sector_column], test_size=0.2, random_state=42
        )


        # Redirect to low score page if R² is too low
        if r2 < 0.5:
            return render_template('low_score.html')

        unique_sectors = sectors_test.unique()
        colors = get_cmap('tab20', len(unique_sectors))

        # Create the scatter plot
        plt.figure(figsize=(17, 8))
        for i, sector in enumerate(unique_sectors):
            sector_filter = sectors_test == sector
            plt.scatter(
                y_test[sector_filter],
                y_pred[sector_filter],
                color=colors(i),
                label=sector
            )

        # Add labels and legend
        plt.xlabel('Actual Salary')
        plt.ylabel('Predicted Salary')
        plt.title('Actual vs Predicted Salary (Linear Regression) with Sectors')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Sectors")
        plt.tight_layout()

        # Convert plot to an image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('result.html', mse=mse, r2=r2, plot_url=plot_url)

    except Exception as e:
        # Redirect to the error page with a generic message
        return render_template('low_score.html', message="An error occurred while processing your submission. Please check your data and try again.")


if __name__ == '__main__':
    app.run(debug=True)
