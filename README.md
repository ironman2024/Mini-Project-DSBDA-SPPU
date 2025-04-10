# Data Science and Big Data Analytics Mini Project

Welcome to the **Data Science and Big Data Analytics (DSBDA) Mini Project** repository. This project is designed to assist third-year Computer Engineering students at Savitribai Phule Pune University (SPPU) in their DSBDA coursework. By providing comprehensive datasets and Python code, this repository aims to facilitate a deeper understanding of data analysis techniques and their practical applications.

## Project Overview

This project focuses on analyzing crime-related data to extract meaningful insights. The datasets encompass various aspects of crime statistics, including property theft, violent crimes, and custodial deaths. Through this project, students will learn to preprocess data, perform exploratory data analysis (EDA), and visualize findings to draw informed conclusions.

## Datasets

The repository includes the following datasets in CSV format:

- **10_Property_stolen_and_recovered.csv**: Details of stolen and recovered property cases.
- **20_Victims_of_rape.csv**: Statistics on rape victims categorized by age and region.
- **25_Complaints_against_police.csv**: Records of complaints filed against police personnel.
- **28_Trial_of_violent_crimes_by_courts.csv**: Information on court trials related to violent crimes.
- **29_Period_of_trials_by_courts.csv**: Duration statistics of various court trials.
- **30_Auto_theft.csv**: Data on reported auto theft incidents.
- **31_Serious_fraud.csv**: Records of serious fraud cases reported.
- **32_Murder_victim_age_sex.csv**: Demographic details of murder victims.
- **33_CH_not_murder_victim_age_sex.csv**: Data on culpable homicide cases not amounting to murder, with victim demographics.
- **35_Human_rights_violation_by_police.csv**: Instances of human rights violations attributed to police actions.
- **36_Police_housing.csv**: Information on housing facilities provided to police personnel.
- **39_Specific_purpose_of_kidnapping_and_abduction.csv**: Categorization of kidnapping and abduction cases based on intent.
- **40_01_Custodial_death_person_remanded.csv**: Details of custodial deaths of remanded individuals.
- **40_02_Custodial_death_person_not_remanded.csv**: Records of custodial deaths of individuals not on remand.
- **40_03_Custodial_death_during_production.csv**: Cases of custodial deaths occurring during court productions.

## Project Structure

The repository is organized as follows:

- **crime/**: Directory containing Python scripts for data analysis.
- **datasets/**: Folder housing all the CSV files mentioned above.
- **notebooks/**: Jupyter notebooks demonstrating data analysis and visualization techniques.

## Getting Started with the DSBDA Mini Project

To effectively utilize this repository, follow the steps below to set up your environment, run the Streamlit application, and optionally deploy it for wider accessibility.

### 1. Clone the Repository

Begin by cloning this repository to your local machine:

```bash
git clone https://github.com/ironman2024/Mini-Project-DSBDA-SPPU.git
```

### 2. Set Up the Environment

Ensure you have Python 3.x installed on your system. It's recommended to create a virtual environment to manage dependencies:

```bash
cd Mini-Project-DSBDA-SPPU
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes all necessary libraries, such as:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `streamlit`

### 3. Run the Streamlit Application

This project includes a Streamlit application for interactive data analysis. To launch the app:

```bash
streamlit run app.py
```

This command will start a local development server and open the application in your default web browser. citeturn0search0

### 4. Explore the Jupyter Notebooks

For detailed data analysis and visualization, navigate to the `notebooks` directory:

```bash
cd notebooks
```

Open the Jupyter notebooks using:

```bash
jupyter notebook
```

These notebooks provide step-by-step analyses and visualizations of the datasets.

### 5. Deploying the Streamlit Application

To make the Streamlit application accessible online, consider deploying it using Streamlit Community Cloud:

1. **Prepare Your Repository**: Ensure your project is pushed to a public GitHub repository.

2. **Sign Up on Streamlit Community Cloud**: Create an account at [Streamlit Community Cloud](https://share.streamlit.io/).

3. **Deploy the App**:
   - Click on "New app" and connect your GitHub repository.
   - Select the repository and branch containing your `app.py`.
   - Click "Deploy."

Your application will be live and accessible via a unique URL. 

By following these steps, you can set up, run, and deploy the DSBDA Mini Project, enhancing your data analysis skills and sharing your work with others. 
## Code Explanation

Below is a sample Python script demonstrating data loading and basic analysis:


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'datasets/10_Property_stolen_and_recovered.csv'
data = pd.read_csv(file_path)

# Display the first few rows
print(data.head())

# Basic statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualization: Property stolen vs. recovered
plt.figure(figsize=(10, 6))
sns.barplot(x='State/UT', y='Number of cases property stolen', data=data, color='red', label='Stolen')
sns.barplot(x='State/UT', y='Number of cases property recovered', data=data, color='green', label='Recovered')
plt.xticks(rotation=90)
plt.title('Property Stolen vs. Recovered by State/UT')
plt.legend()
plt.show()
```


**Explanation**:

1. **Import Libraries**: The script imports necessary libraries: `pandas` for data manipulation, and `matplotlib.pyplot` and `seaborn` for data visualization.

2. **Load Dataset**: The dataset `10_Property_stolen_and_recovered.csv` is loaded into a DataFrame.

3. **Inspect Data**: The first few rows and basic statistics of the dataset are displayed to understand its structure and contents.

4. **Check Missing Values**: The script checks for any missing values in the dataset to ensure data quality.

5. **Visualization**: A bar plot is created to compare the number of property theft cases versus recovered cases across different States/UTs. This helps in visualizing and comparing the effectiveness of property recovery efforts regionally.

## Contributing

Contributions to enhance this project are welcome. Feel free to fork the repository, make modifications, and submit pull requests. Your contributions can help fellow students and practitioners in their data science journey.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this code as per the license terms.

---

*By providing this repository, we aim to support students in their DSBDA coursework and foster a collaborative learning environment. If you find this project helpful, consider starring the repository to show your support.* 
