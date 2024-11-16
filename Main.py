import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.gaussian_process.kernels import Matern
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize

st.header("Mental Health Survey")

df=pd.read_csv("/Users/sparshmehta/Downloads/code/survey code/Dataset1.csv")

st.write("Shape of Dataset - ", df.shape)

st.write("Data Header - ", df.head(10))

column = st.sidebar.multiselect("Choose String columns to drop", df.columns)
NC = df.drop(column, axis=1)

if 'Gender' or 'gender' in df.columns:
    if False:
        df.isnull().sum().sort_values(ascending=False)
        male = ['male', 'Male', 'M', 'm', 'Male-ish', 'maile', 'Cis Male', 'Mal', 'Male (CIS)', 'Make', 'Male ', 'Man',
                'msle', 'Mail', 'cis male', 'Malr', 'Cis Man']
        female = ['Female', 'female', 'Cis Female', 'F', 'Woman', 'f', 'Femake', 'woman', 'Female ', 'cis-female/femme',
                  'Female (cis)', 'femail']
        other = ['Trans-female', 'something kinda male?', 'queer/she/they', 'non-binary', 'Nah', 'All', 'Enby', 'fluid',
                 'Genderqueer', 'Androgyne', 'Agender', 'Guy (-ish) ^_^', 'male leaning androgynous', 'Trans woman',
                 'Neuter', 'Female (trans)', 'queer', 'ostensibly male, unsure what that really means']
        dropped = ['A little about you', 'p']
        df = df[(df['Gender'] != 'A little about you') & (df['Gender'] != 'p')]
        df.replace(to_replace=male, value='male', inplace=True)
        df.replace(to_replace=other, value='other', inplace=True)
        df.replace(to_replace=female, value='female', inplace=True)
if 'Age' or 'age' in df.columns:
    if False:
        df['Age'].fillna(df['Age'].median(), inplace=True)

        # Fill with media() values < 18 and > 120
        s = pd.Series(df['Age'])
        s[s < 18] = df['Age'].median()
        df['Age'] = s
        s = pd.Series(df['Age'])
        s[s > 120] = df['Age'].median()
        df['Age'] = s

        # Ranges of Age
        df['age_range'] = pd.cut(df['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"],
                                 include_lowest=True)
        df['self_employed'] = df['self_employed'].replace([defaultString], 'No')
        df['work_interfere'] = df['work_interfere'].replace([defaultString], 'Don\'t know')
        df = df.drop(['Country'], axis=1)

stress_symptoms = [
    'Feeling overwhelmed or unable to cope with daily tasks',
    'Feeling tense or unable to relax',
    'Frequent headaches or body aches',
    'Changes in appetite (eating more or less than usual)',
    'Difficulty concentrating due to stress',
    'Sleep disturbances due to stress (e.g., difficulty falling asleep, waking up frequently)',
    'Avoiding activities or situations due to stress'
]
anxiety_symptoms = [
    'Feeling restless or on edge',
    'Excessive worry about various things',
    'Difficulty controlling worry',
    'Feeling easily fatigued',
    'Irritability',
    'Muscle tension',
    'Sleep disturbances (e.g., difficulty falling or staying asleep, restless sleep)',
    'Difficulty concentrating or mind going blank'
]
depression_symptoms = [
    'Feeling down, depressed, or hopeless',
    'Little interest or pleasure in doing things',
    'Significant weight loss or gain or change in appetite',
    'Trouble falling asleep, staying asleep, or sleeping too much',
    'Feeling tired or having low energy',
    'Difficulty concentrating or making decisions',
    'Thoughts of death or suicide'
]
life_affecting_symptoms=[
    'Ability to work or attend school',
    'Interpersonal relationships (e.g., family, friends)',
    'Self-care (e.g., grooming, eating habits)',
    'Overall quality of life'
]
boolean_symptoms=[
    'Have you ever been diagnosed with an anxiety disorder, depression, or stress-related condition?',
    'Have you sought professional help for your mental health concerns?',
    'Have you experienced any significant life events or stressors recently?',
    'Have you experienced any traumatic events in the past?',
    'Do you have a history of mental health conditions in your family?'
]
symptom_categories = {
    'Stress Symptoms': stress_symptoms,
    'Anxiety Symptoms': anxiety_symptoms,
    'Depression Symptoms': depression_symptoms,
    'Life-Affecting Symptoms': life_affecting_symptoms,
    'Boolean Symptoms': boolean_symptoms
}

df['Stress_Score'] = df[stress_symptoms].sum(axis=1)
df['Anxiety_Score'] = df[anxiety_symptoms].sum(axis=1)
df['Depression_Score'] = df[depression_symptoms].sum(axis=1)
df['Life_Affecting_Score'] = df[life_affecting_symptoms].sum(axis=1)

# Assuming boolean answers are coded as 0 and 1, you can simply sum them
df['Boolean_Symptoms_Score'] = df[boolean_symptoms].sum(axis=1)

def assign_grade(score):
    if 0 <= score <= 4:
        return 'Minimal'
    elif 5 <= score <= 9:
        return 'Mild'
    elif 10 <= score <= 14:
        return 'Moderate'
    elif 15 <= score:
        return 'Severe'

selected_symptom_groups = st.sidebar.multiselect("Choose Symptom Groups", list(symptom_categories.keys()))

selected_symptoms = []
for group in selected_symptom_groups:
    selected_symptoms += symptom_categories[group]

df = pd.read_csv("/Users/sparshmehta/Downloads/code/survey code/Dataset1.csv")

# Sidebar - User input options
st.sidebar.title("Filter Data")
selected_gender = st.sidebar.multiselect("Select Gender(s)", df['Gender'].unique(), default=df['Gender'].unique())
selected_age_range = st.sidebar.slider("Select Age Range", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=(int(df['Age'].min()), int(df['Age'].max())))
selected_education = st.sidebar.multiselect("Select Education Level(s)", df['Education Level'].unique(), default=df['Education Level'].unique())
selected_marital_status = st.sidebar.multiselect("Select Marital Status(es)", df['Marital Status'].unique(), default=df['Marital Status'].unique())
selected_relationship_status = st.sidebar.multiselect("Select Relationship Status(es)", df['Relationship Status'].unique(), default=df['Relationship Status'].unique())

# Filter the DataFrame based on user input
df_filtered = df[
    (df['Gender'].isin(selected_gender)) &
    (df['Age'].between(selected_age_range[0], selected_age_range[1])) &
    (df['Education Level'].isin(selected_education)) &
    (df['Marital Status'].isin(selected_marital_status)) &
    (df['Relationship Status'].isin(selected_relationship_status) &
     df['Diagnosis'] & 
     df['Grade'])
]

df_final = df_filtered[['Age', 'Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'] + selected_symptoms]

st.write("Updated Dataframe:")
st.dataframe(df_final)

df['Stress_Score'] = df[stress_symptoms].sum(axis=1)
df['Anxiety_Score'] = df[anxiety_symptoms].sum(axis=1)
df['Depression_Score'] = df[depression_symptoms].sum(axis=1)
df['Life_Affecting_Score'] = df[life_affecting_symptoms].sum(axis=1)

# Assuming boolean answers are coded as 0 and 1, you can simply sum them
df['Boolean_Symptoms_Score'] = df[boolean_symptoms].sum(axis=1)

st.write("DataFrame with Scores and Grades of symptoms: ")

df['Stress_Grade'] = df['Stress_Score'].apply(assign_grade)
df['Anxiety_Grade'] = df['Anxiety_Score'].apply(assign_grade)
df['Depression_Grade'] = df['Depression_Score'].apply(assign_grade)
df['Life_Affecting_Grade'] = df['Life_Affecting_Score'].apply(assign_grade)
df['Boolean_Symptoms_Grade'] = df['Boolean_Symptoms_Score'].apply(assign_grade)

df.head()

st.dataframe(df)

X_interactions = df_final.drop(columns=['Gender','Education Level',
       'Marital Status', 'Relationship Status','Diagnosis', 'Grade'])
y_diag = df_final['Diagnosis']
y_grade = df_final['Grade']

compn = st.number_input('Input number of components for PCA')

compon = int(compn)

pca = PCA(n_components=compon)
X_pca = pca.fit_transform(X_interactions)

explained_variance = pca.explained_variance_ratio_.sum()

st.write("Explained Variance after PCA - ",explained_variance)

z = int(st.slider("Select the Test Split value", 60.0, 80.0, 70.0))

sta = int(st.slider("Select Random state value", 16, 64, 42))

df_modified = pd.read_csv('/Users/sparshmehta/Downloads/code/survey code/Dataset1.csv')

symptoms_columns = stress_symptoms + anxiety_symptoms + depression_symptoms + life_affecting_symptoms + boolean_symptoms
symptoms_data = df_modified[symptoms_columns]
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
interaction_data = poly.fit_transform(symptoms_data)
interaction_df = pd.DataFrame(interaction_data, columns=poly.get_feature_names_out(symptoms_columns))
df_interactions = df_modified.drop(columns=symptoms_columns)
df_interactions = pd.concat([df_interactions, interaction_df], axis=1)

st.write('Anxiety value count:\t\t\t',df_interactions['Diagnosis'].value_counts()['Anxiety'])
st.write('Stress value count:\t\t\t',df_interactions['Diagnosis'].value_counts()['Stress'])
st.write('Depression value count:\t\t\t',df_interactions['Diagnosis'].value_counts()['Depression'])
st.write('Normal value count:\t\t\t',df_interactions['Diagnosis'].value_counts()['Normal'])
st.write('Anxiety and Depression value count:\t',df_interactions['Diagnosis'].value_counts()['Anxiety and Depression'])
st.write('Depression and Stress value count:\t',df_interactions['Diagnosis'].value_counts()['Depression and Stress'])
st.write('Anxiety and Stress value count:\t\t',df_interactions['Diagnosis'].value_counts()['Anxiety and Stress'])

selected_symptom_groups_for_corr = st.sidebar.multiselect("Choose Symptom Categories for relationship between them", list(symptom_categories.keys()), default=["Stress Symptoms", "Anxiety Symptoms"])

if len(selected_symptom_groups_for_corr) != 2:
    st.warning("Please select exactly two symptom categories.")
else:
    selected_symptoms_for_corr = []
    for group in selected_symptom_groups_for_corr:
        selected_symptoms_for_corr += symptom_categories[group]

    df_filtered_for_corr = df[['Age', 'Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'] + selected_symptoms_for_corr]

    corr_mat = df_filtered_for_corr[selected_symptoms_for_corr].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.write(fig)

grade_column = st.selectbox("Select Grade classification:", ['binary classification','three class','Four class','multiclass classification'])

unique_grades = df_final['Grade'].unique()

if grade_column == 'binary classification':
    selected_grades = [unique_grades[0], unique_grades[4]] 
    df_finale = df_final[df_final['Grade'].isin(selected_grades)]

elif grade_column == 'three class':
    selected_grades = [unique_grades[0], unique_grades[2], unique_grades[4]] 
    df_finale = df_final[df_final['Grade'].isin(selected_grades)]
    
elif grade_column == 'Four class':
    selected_grades = [unique_grades[4] ,unique_grades[1] , unique_grades[2], unique_grades[0]] 
    df_finale = df_final[df_final['Grade'].isin(selected_grades)]

elif grade_column == 'multiclass classification':
    df_finale = df_final
    st.title('Multiclass Classification: All Grades')

st.write(df_finale)

model = st.selectbox('Model Selection', ['Support Vector Classifier', 'SVC + smote','SVC + Adaboost','SVC + Bagging','SVC + Adaboost + SMOTE','SVC + Bagging + SMOTE','Random Forest Classifier',
                                         'Random Forest + Smote','Random Forest + Adaboost','Random Forest + Bagging','K-nearestNeighbor', 'KNN + Smote', 
                                         'Decision Tree', 'Decision Tree + Smote', 'Decision Tree + Adaboost', 'Decision Tree + Bagging', 'Gradient Boost Classifier', 'Gradient Boost + Smote', 'Gradient Boost + Adaboost', 'Gradient Boost + Bagging',
                                         'Logistic Regression', 'Logistic + Smote', 'Logistic + AdaBoost', 'Logistic + Bagging','XGBoost','Deep Learning'])

kfold = st.selectbox("Select k-fold validation method",['no fold','5 fold','10 fold'])

#Support Vector Classifier
if model == 'Support Vector Classifier':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import classification_report

    kernel_params = st.selectbox('Kernel Parameters', ['poly', 'sigmoid', 'rbf', 'linear'])

    # Initialize lists to store error rates, degrees, training accuracy, and test accuracy
    degrees = st.slider('Select Degree', 0, 10, 3)

    class_weights = 'balanced'

    df_mod = df_finale

    # Extract features and target
    X_interactions = df_mod.drop(columns=['Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Specify which columns to encode
    categorical_columns = ['Gender', 'Education Level', 'Marital Status', 'Relationship Status']

    # Create a column transformer for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(), categorical_columns)
        ],
        remainder='passthrough'
    )

    # Create a pipeline with the column transformer and the SVM classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(C=1.0, kernel=kernel_params, degree=degrees, gamma='scale', class_weight=class_weights))
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train_diag, y_test_diag = train_test_split(X_interactions, y_diag, test_size=z, random_state=sta)

    try:
        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train_diag)

        # Make predictions on the training and test sets
        predict_train_i = pipeline.predict(X_train)
        predict_test_i = pipeline.predict(X_test)

        # Calculate error rates and accuracy
        error_rate = np.mean(predict_train_i != y_train_diag)
        train_accuracy = accuracy_score(y_train_diag, predict_train_i)
        test_accuracy = accuracy_score(y_test_diag, predict_test_i)

        st.write("Error Rate:", error_rate)
        st.write("Training Accuracy:", train_accuracy)
        st.write("Test Accuracy:", test_accuracy)

    except ValueError:
        st.write("Error: Invalid combination of parameters")

    svc_train_pred = pipeline.predict(X_train)
    svc_test_pred = pipeline.predict(X_test)
    train_accuracy = accuracy_score(y_train_diag, svc_train_pred)
    test_accuracy = accuracy_score(y_test_diag, svc_test_pred)

    df_report = (classification_report(y_test_diag, svc_test_pred, output_dict=True))
    report = pd.DataFrame(df_report)
    st.subheader("Classification Report:")
    st.dataframe(report)

    # Correlation Heatmap
    numeric_df = df_mod.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()

    conf_matrix = confusion_matrix(y_test_diag, svc_test_pred)
    st.write("Confusion Matrix for Test Set:")
    st.write(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display the plot within Streamlit
    st.pyplot(plt.gcf())

    # Function to plot ROC curve for multiclass classification
    def plot_roc_curve_multiclass(y_true, y_scores, n_classes, label):
        # Convert y_true to one-hot encoding
        y_true_onehot = pd.get_dummies(y_true)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot.iloc[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.values.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot micro and macro-average ROC curves
        plt.figure(figsize=(8, 8))
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (AUC = {:.2f})'.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (AUC = {:.2f})'.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        # Plot ROC curve for each class
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve (AUC = {:.2f}) for class {}'.format(roc_auc[i], i))

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - {}'.format(label))
        plt.legend(loc="lower right")
        plt.show()

    # AUC curve and ROC curve for the training set
    st.subheader("AUC and ROC Curve (SVC):")
    y_scores_train = pipeline.decision_function(X_train)
    plot_roc_curve_multiclass(y_train_diag, y_scores_train, n_classes=len(np.unique(y_train_diag)), label='Training Set')
    plt.savefig('roc_curve.png')
    
    st.image('roc_curve.png')

    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

    conf_matrix = confusion_matrix(y_test_diag, svc_test_pred)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#SVC + Smote
elif model == "SVC + smote":
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, classification_report
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
    from imblearn.over_sampling import SMOTE
    import streamlit as st

    # Assuming df_final is your original dataframe
    df_mod = df_finale

    # Extract features and target
    X_interactions = df_mod.drop(columns=['Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Specify which columns to encode
    categorical_columns = ['Gender', 'Education Level', 'Marital Status', 'Relationship Status']

    # Create a column transformer for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(), categorical_columns)
        ],
        remainder='passthrough'
    )

    # Create a pipeline with the column transformer, SMOTE, and the SVM classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', SVC(C=1.0, kernel='linear', gamma='scale', class_weight='balanced'))
    ])

    from imblearn.over_sampling import SMOTENC

    X_train, X_test, y_train_diag, y_test_diag = train_test_split(X_interactions, y_diag, test_size=0.2, random_state=42)

    # Identify categorical columns
    categorical_columns = ['Gender', 'Education Level', 'Marital Status', 'Relationship Status']
    categorical_feature_indices = [X_train.columns.get_loc(col) for col in categorical_columns]

    # Apply SMOTENC to the training set
    smote = SMOTENC(categorical_features=categorical_columns, random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_diag)

    # Create a pipeline with the column transformer and the SVM classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(C=1.0, kernel='linear', gamma='scale', class_weight='balanced'))
    ])

    try:
        # Fit the pipeline on the resampled training data
        pipeline.fit(X_train_resampled, y_train_resampled)

        # Make predictions on the training and test sets
        predict_train_i = pipeline.predict(X_train)
        predict_test_i = pipeline.predict(X_test)

        # Calculate error rates and accuracy
        error_rate = np.mean(predict_train_i != y_train_diag)
        train_accuracy = accuracy_score(y_train_diag, predict_train_i)
        test_accuracy = accuracy_score(y_test_diag, predict_test_i)

        st.write("Error Rate:", error_rate)
        st.write("Training Accuracy:", train_accuracy)
        st.write("Test Accuracy:", test_accuracy)

    except ValueError:
        st.write("Error: Invalid combination of parameters")

    svc_train_pred = pipeline.predict(X_train)
    svc_test_pred = pipeline.predict(X_test)
    train_accuracy = accuracy_score(y_train_diag, svc_train_pred)
    test_accuracy = accuracy_score(y_test_diag, svc_test_pred)

    df_report = (classification_report(y_test_diag, svc_test_pred, output_dict=True))
    report = pd.DataFrame(df_report)
    st.subheader("Classification Report:")
    st.dataframe(report)

    
    conf_matrix = confusion_matrix(y_test_diag, svc_test_pred)
    st.subheader("Confusion Matrix:")
    st.write("Confusion Matrix for Test Set:")
    st.write(conf_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap:")
    numeric_df = df_mod.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()

    # Adjust the size of the heatmap and the font size of annotations
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, annot_kws={"size": 10})

    # Display the heatmap in Streamlit
    st.pyplot(heatmap.figure)

    # Function to plot ROC curve for multiclass classification
    def plot_roc_curve_multiclass(y_true, y_scores, n_classes, label):
        # Convert y_true to one-hot encoding
        y_true_onehot = pd.get_dummies(y_true)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot.iloc[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.values.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot micro and macro-average ROC curves
        plt.figure(figsize=(8, 8))
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (AUC = {:.2f})'.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (AUC = {:.2f})'.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        # Plot ROC curve for each class
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve (AUC = {:.2f}) for class {}'.format(roc_auc[i], i))

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - {}'.format(label))
        plt.legend(loc="lower right")
        plt.show()

    # AUC curve and ROC curve for the training set
    st.subheader("AUC and ROC Curve (SVM):")
    y_scores_train = pipeline.decision_function(X_train)
    plot_roc_curve_multiclass(y_train_diag, y_scores_train, n_classes=len(np.unique(y_train_diag)), label='Training Set')
    plt.savefig('roc_curve.png')

    st.image('roc_curve.png')

    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

    conf_matrix = confusion_matrix(y_test_diag, svc_test_pred)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#SVC wit Adaboost
elif model == 'SVC + Adaboost':
    from sklearn.svm import SVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
    import seaborn as sns
    import matplotlib.pyplot as plt

    X = df_finale.drop(columns=['Diagnosis'])
    y = df_finale['Diagnosis']
    
    categorical_cols = ['Age', 'Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Grade']

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'  # This includes non-categorical columns as they are
    )

    X_encoded = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    n_estimators = st.slider("Select Number of Estimators", 1, 100, 50)
    random_state = st.slider("Select Random State", 0, 100, 42)
    kernel = st.selectbox("Select Kernel", ['linear', 'poly', 'rbf', 'sigmoid'])

    adaboost_svm_clf = AdaBoostClassifier(estimator=SVC(probability=True, kernel=kernel), n_estimators=n_estimators, random_state=random_state)

    adaboost_svm_clf.fit(X_train, y_train)
    y_pred = adaboost_svm_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy  
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    st.write(f"Accuracy:", accuracy)
    st.write(f"Error Rate:",  error_rate)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for SVC + Adaboost')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())

    st.write("Classification Report:")
    st.dataframe(classification_rep)

    # ROC Curve
    n_classes = len(adaboost_svm_clf.classes_)
    y_scores = adaboost_svm_clf.predict_proba(X_test)
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test == adaboost_svm_clf.classes_[i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (class {adaboost_svm_clf.classes_[i]}) (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (SVC + Adaboost)')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#SVC + Bagging
elif model == "SVC + Bagging":
    from sklearn.ensemble import BaggingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    X = df_filtered.drop(columns=['Diagnosis', 'Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Grade'])
    y = df_filtered['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    n_estimators = st.slider("Select Number of Estimators", 1, 100, 50)
    random_state = st.slider("Select Random State", 0, 100, 42)

    # Create base SVC classifier
    base_classifier = SVC(probability=True)
    
    # Create BaggingClassifier with SVC as estimator
    bagging_classifier = BaggingClassifier(
        estimator=base_classifier,
        n_estimators=n_estimators,
        random_state=random_state
    )

    # Fit the model
    bagging_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = bagging_classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    # Display results
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Error Rate: {error_rate:.2f}")
    st.write("\nConfusion Matrix:")
    st.write(conf_matrix)
    report = pd.DataFrame(classification_rep)
    st.write("\nClassification Report:")
    st.write(report)

    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy and error rate
    plt.subplot(1, 2, 1)
    plt.plot([accuracy] * n_estimators, label='Accuracy', marker='o')
    plt.plot([error_rate] * n_estimators, label='Error Rate', marker='o')
    plt.ylabel('Metric')
    plt.title('Accuracy and Error Rate')
    plt.legend()

    # Plot confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    st.pyplot()

#SVC + Adaboost + SMOTE
elif model == 'SVC + Adaboost + SMOTE':
    from sklearn.svm import SVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
    from imblearn.over_sampling import SMOTE
    import seaborn as sns
    import matplotlib.pyplot as plt

    X = df_finale.drop(columns=['Diagnosis'])
    y = df_finale['Diagnosis']
    
    categorical_cols = ['Age', 'Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Grade']

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'  # This includes non-categorical columns as they are
    )

    X_encoded = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    n_estimators = st.slider("Select Number of Estimators", 1, 100, 50)
    random_state = st.slider("Select Random State", 0, 100, 42)
    kernel = st.selectbox("Select Kernel", ['linear', 'poly', 'rbf', 'sigmoid'])

    adaboost_svm_clf = AdaBoostClassifier(estimator=SVC(probability=True, kernel=kernel), n_estimators=n_estimators, random_state=random_state)

    adaboost_svm_clf.fit(X_train_resampled, y_train_resampled)
    y_pred = adaboost_svm_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy  
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    st.write(f"Accuracy:", accuracy)
    st.write(f"Error Rate:",  error_rate)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for SVC + Adaboost + SMOTE')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())

    st.write("Classification Report:")
    st.dataframe(classification_rep)

    # ROC Curve
    n_classes = len(adaboost_svm_clf.classes_)
    y_scores = adaboost_svm_clf.predict_proba(X_test)
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test == adaboost_svm_clf.classes_[i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (class {adaboost_svm_clf.classes_[i]}) (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (SVC + Adaboost + SMOTE)')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#SVC + Bagging + Smote
elif model == "SVC + Bagging + SMOTE":
    from sklearn.svm import SVC
    from sklearn.ensemble import BaggingClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Prepare the data
    X = df_filtered.drop(columns=['Diagnosis', 'Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Grade'])
    y = df_filtered['Diagnosis']
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    
    # Create base SVC classifier
    svc = SVC(probability=True, random_state=42)
    
    # Create and train Bagging classifier
    bagging_classifier = BaggingClassifier(
        estimator=svc,
        n_estimators=10,
        random_state=42
    )
    bagging_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = bagging_classifier.predict(X_test)
    y_proba = bagging_classifier.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    # Display results
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("\nConfusion Matrix:")
    st.write(conf_matrix)
    
    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for SVC + Bagging + SMOTE')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt.gcf())
    
    # Display classification report
    st.write("\nClassification Report:")
    report = pd.DataFrame(classification_rep)
    st.write(report)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(np.unique(y)):
        fpr, tpr, _ = roc_curve(y_test == class_label, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for SVC + Bagging + SMOTE')
    plt.legend()
    st.pyplot(plt.gcf())

#Random Forest Classifier
elif model == 'Random Forest Classifier':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    import streamlit as st
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Assuming df_final is your dataset
    df_mod = df_finale

    # Extract features and target
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_interactions, y_diag, test_size=z, random_state=sta)

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict probabilities for each class
    y_probs = rf_classifier.predict_proba(X_test)
    
    # Other evaluation metrics
    y_pred = rf_classifier.predict(X_test)
    st.write('Accuracy:', accuracy_score(y_test, y_pred))
    st.write('Confusion Matrix:')
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Random Forest Classifier')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())
    st.write('Classification Report:')
    st.dataframe(classification_report(y_test, y_pred, output_dict=True))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y_diag))):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(pd.get_dummies(y_test).values.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y_diag))):
        plt.plot(fpr[i], tpr[i], label='ROC curve (AUC = {:.2f}) for class {}'.format(roc_auc[i], i))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {:.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig('roc_curve_rf.png')

    # Display the plot
    st.pyplot()

    # Display the AUC values
    st.write("AUC for each class:", roc_auc)

    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Random Forest Classifier')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#RFR + Smote
elif model == "Random Forest + Smote":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, mean_squared_error, mean_absolute_error
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    import streamlit as st
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from imblearn.over_sampling import SMOTE

    # Assuming df_final is your dataset
    df_mod = df_finale

    # Extract features and target
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_interactions, y_diag)

    # PCA for dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_resampled)

    # Split the resampled data into training and testing sets
    X_train_pca, X_test_pca, y_train_diag, y_test_diag = train_test_split(X_pca, y_resampled, test_size=z, random_state=sta)

    # Slider for the number of estimators
    n_estimators = st.slider("Number of Estimators", min_value=1, max_value=200, value=100)
    # Slider for random state
    random_state = st.slider("Random State", min_value=1, max_value=100, value=42)

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_classifier.fit(X_train_pca, y_train_diag)
    rf_pred_diag = rf_classifier.predict(X_test_pca)

    # Evaluate the model
    rf_accuracy = accuracy_score(y_test_diag, rf_pred_diag)
    rf_confusion = confusion_matrix(y_test_diag, rf_pred_diag)
    rf_classification_report = classification_report(y_test_diag, rf_pred_diag, output_dict=True)

    # Display results
    st.write(f"Random Forest Accuracy: {rf_accuracy*100.0:.2f}%")
    st.write("Random Forest Confusion Matrix:")
    st.write(rf_confusion)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(rf_confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Random Forest + SMOTE')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())

    st.write('Classification Report:')
    report = pd.DataFrame(rf_classification_report)
    st.dataframe(report)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap (Random Forest):")
    numeric_df_rf = pd.concat([X_interactions, y_diag], axis=1).select_dtypes(include='number')
    corr_matrix_rf = numeric_df_rf.corr()
    sns.heatmap(corr_matrix_rf, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()

    y_probs = rf_classifier.predict_proba(X_test_pca)
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y_diag))):
        fpr[i], tpr[i], _ = roc_curve(y_test_diag == i, y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(pd.get_dummies(y_test_diag).values.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y_diag))):
        plt.plot(fpr[i], tpr[i], label='ROC curve (AUC = {:.2f}) for class {}'.format(roc_auc[i], i))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {:.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig('roc_curve_rf.png')

    # Display the plot
    st.pyplot()

    # Display the AUC values
    st.write("AUC for each class:", roc_auc)

    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

    conf_matrix = confusion_matrix(y_test_diag, rf_pred_diag)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#Random Forest with Adaboost
elif model == 'Random Forest + Adaboost':
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
    import matplotlib.pyplot as plt

    X = df_finale.drop(columns=['Diagnosis'])
    y = df_finale['Diagnosis']

    categorical_cols = ['Age', 'Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Grade']

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )

    X_encoded = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)


    depths = st.slider("Select Max Depth", 1, 5, 3)
    n_estimators = st.slider("Select Number of Estimators", 1, 100, 50)

    accuracies = []
    error_rates = []

    for depth in range(1, depths + 1):
        base_estimator = RandomForestClassifier(max_depth=depth, random_state=42)

        adaboost_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=42)

        adaboost_clf.fit(X_train, y_train)

        y_pred = adaboost_clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1 - accuracy

        accuracies.append(accuracy)
        error_rates.append(error_rate)

        st.write(f"Depth: {depth}, Accuracy: {accuracy:.2f}, Error Rate: {error_rate:.2f}")

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_error_rate = sum(error_rates) / len(error_rates)

    st.write(f"\nAverage Accuracy: {avg_accuracy:.2f}")
    st.write(f"Average Error Rate: {avg_error_rate:.2f}")

    # Classification Report
    classification_rep = classification_report(y_test, y_pred, output_dict = True)
    report = pd.DataFrame(classification_rep)
    st.write("Classification Report:")
    st.write(report)

    conf_matrix = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix:")
    st.write("Confusion Matrix for Test Set:")
    st.write(conf_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Random Forest + Adaboost')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())

    # Visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, depths + 1), accuracies, marker='o')
    plt.title('Accuracy vs Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, depths + 1), error_rates, marker='o', color='red')
    plt.title('Error Rate vs Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Error Rate')

    st.pyplot()

    y_probs = adaboost_clf.predict_proba(X_test)
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y_diag))):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(pd.get_dummies(y_test).values.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y_diag))):
        plt.plot(fpr[i], tpr[i], label='ROC curve (AUC = {:.2f}) for class {}'.format(roc_auc[i], i))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {:.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig('roc_curve_rf.png')

    # Display the plot
    st.pyplot()

    # Display the AUC values
    st.write("AUC for each class:", roc_auc)

    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#Random Forest with Bagging
elif model == 'Random Forest + Bagging':
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
    from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
    import matplotlib.pyplot as plt

    X = df_finale.drop(columns=['Diagnosis'])
    y = df_finale['Diagnosis']

    categorical_cols = ['Age', 'Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Grade']

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )

    X_encoded = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    depths = st.slider("Select Max Depth", 1, 5, 3)
    n_estimators = st.slider("Select Number of Estimators", 1, 100, 50)

    accuracies = []
    error_rates = []

    for depth in range(1, depths + 1):
        base_estimator = RandomForestClassifier(max_depth=depth, random_state=42)

        # Updated parameter name from base_estimator to estimator
        bagging_clf = BaggingClassifier(
            estimator=base_estimator, 
            n_estimators=n_estimators, 
            random_state=42
        )

        bagging_clf.fit(X_train, y_train)
        y_pred = bagging_clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1 - accuracy

        accuracies.append(accuracy)
        error_rates.append(error_rate)

        st.write(f"Depth: {depth}, Accuracy: {accuracy:.2f}, Error Rate: {error_rate:.2f}")

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_error_rate = sum(error_rates) / len(error_rates)

    st.write(f"\nAverage Accuracy: {avg_accuracy:.2f}")
    st.write(f"Average Error Rate: {avg_error_rate:.2f}")

    # Classification Report
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(classification_rep)
    st.write("Classification Report:")
    st.write(report)

    conf_matrix = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix:")
    st.write("Confusion Matrix for Test Set:")
    st.write(conf_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Random Forest + Bagging')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt.gcf())
    
    # Visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, depths + 1), accuracies, marker='o')
    plt.title('Accuracy vs Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, depths + 1), error_rates, marker='o', color='red')
    plt.title('Error Rate vs Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Error Rate')

    st.pyplot()

    # ROC Curve
    y_probs = bagging_clf.predict_proba(X_test)
    
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y_diag))):
        fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f}) for class {i}')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest + Bagging')
    plt.legend(loc="lower right")
    st.pyplot()

#K Nearest Neighbor
elif model == "K-nearestNeighbor":
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize

    # Prepare the data
    df_mod = df_finale
    X_interactions = df_mod.drop(
        columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Convert labels to numerical format
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_diag = le.fit_transform(y_diag)
    
    # Dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_interactions)

    # Split the data
    X_train_pca, X_test_pca, y_train_diag, y_test_diag = train_test_split(X_pca, y_diag, test_size=z, random_state=sta)

    # Parameters for grid search
    k_values = list(range(1, 21))
    distance_methods = ['euclidean', 'manhattan']
    tree_methods = ['kd_tree', 'ball_tree']
    voting_methods = ['uniform', 'distance']

    accuracy_records = []

    # Grid search
    for k in k_values:
        for distance_method in distance_methods:
            for tree_method in tree_methods:
                for voting_method in voting_methods:
                    knn_classifier = KNeighborsClassifier(
                        n_neighbors=k, 
                        algorithm=tree_method,
                        metric=distance_method, 
                        weights=voting_method
                    )
                    knn_classifier.fit(X_train_pca, y_train_diag)
                    knn_pred_diag = knn_classifier.predict(X_test_pca)
                    accuracy = accuracy_score(y_test_diag, knn_pred_diag)
                    accuracy_records.append({
                        'K': k, 
                        'Distance Method': distance_method, 
                        'Tree Method': tree_method,
                        'Voting Method': voting_method, 
                        'Accuracy': accuracy
                    })

    accuracy_df = pd.DataFrame(accuracy_records)

    # Find optimal hyperparameters
    optimal_hyperparameters = accuracy_df.loc[accuracy_df['Accuracy'].idxmax()]
    optimal_k = int(optimal_hyperparameters['K'])
    optimal_distance_method = optimal_hyperparameters['Distance Method']
    optimal_tree_method = optimal_hyperparameters['Tree Method']
    optimal_voting_method = optimal_hyperparameters['Voting Method']

    # User selection
    selected_k = st.selectbox('Select Number of Neighbors (K):', k_values, index=k_values.index(optimal_k))
    selected_distance_method = st.selectbox('Select Distance Method:', distance_methods, index=distance_methods.index(optimal_distance_method))
    selected_tree_method = st.selectbox('Select Tree Method:', tree_methods, index=tree_methods.index(optimal_tree_method))
    selected_voting_method = st.selectbox('Select Voting Method:', voting_methods, index=voting_methods.index(optimal_voting_method))

    # Display optimal parameters
    st.write(f"Optimal Number of Neighbors (K): {optimal_k}")
    st.write(f"Optimal Distance Method: {optimal_distance_method}")
    st.write(f"Optimal Tree Method: {optimal_tree_method}")
    st.write(f"Optimal Voting Method: {optimal_voting_method}")

    # Train final model with selected parameters
    knn_classifier = KNeighborsClassifier(
        n_neighbors=selected_k,
        algorithm=selected_tree_method,
        metric=selected_distance_method,
        weights=selected_voting_method
    )
    knn_classifier.fit(X_train_pca, y_train_diag)

    # Predictions
    knn_pred_diag = knn_classifier.predict(X_test_pca)
    y_proba = knn_classifier.predict_proba(X_test_pca)

    # Metrics
    knn_confusion = confusion_matrix(y_test_diag, knn_pred_diag)
    knn_accuracy = accuracy_score(y_test_diag, knn_pred_diag)
    knn_classification_report = classification_report(y_test_diag, knn_pred_diag, output_dict=True)

    # Display results
    st.write("KNN Confusion Matrix:")
    st.write(knn_confusion)

    plt.figure(figsize=(8, 6))
    sns.heatmap(knn_confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for KNN')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt.gcf())

    st.write("KNN Classification Report:")
    report = pd.DataFrame(knn_classification_report)
    st.write(report)

    # ROC Curve
    n_classes = len(np.unique(y_diag))
    y_test_bin = label_binarize(y_test_diag, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["micro"]))
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for KNN Classification')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())

    st.write("AUC for each class:", roc_auc)

#KNN + Smote
elif model == "KNN + Smote":
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
    from imblearn.over_sampling import SMOTE
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize, LabelEncoder
    import seaborn as sns

    # Prepare the data - use only numerical columns
    df_mod = df_finale
    X = df_mod.select_dtypes(include=['float64', 'int64'])  # Select only numerical columns
    y = df_mod['Diagnosis']

    # Convert labels to numerical format
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_resampled)

    # Split the resampled data
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y_resampled, test_size=z, random_state=sta)

    # Parameters for grid search
    k_values = list(range(1, 21))
    distance_methods = ['euclidean', 'manhattan']
    tree_methods = ['kd_tree', 'ball_tree']
    voting_methods = ['uniform', 'distance']

    accuracy_records = []

    # Grid search
    for k in k_values:
        for distance_method in distance_methods:
            for tree_method in tree_methods:
                for voting_method in voting_methods:
                    knn_classifier = KNeighborsClassifier(
                        n_neighbors=k, 
                        algorithm=tree_method,
                        metric=distance_method, 
                        weights=voting_method
                    )
                    knn_classifier.fit(X_train_pca, y_train)
                    knn_pred = knn_classifier.predict(X_test_pca)
                    accuracy = accuracy_score(y_test, knn_pred)
                    accuracy_records.append({
                        'K': k, 
                        'Distance Method': distance_method, 
                        'Tree Method': tree_method,
                        'Voting Method': voting_method, 
                        'Accuracy': accuracy
                    })

    accuracy_df = pd.DataFrame(accuracy_records)

    # Find optimal hyperparameters
    optimal_hyperparameters = accuracy_df.loc[accuracy_df['Accuracy'].idxmax()]
    optimal_k = int(optimal_hyperparameters['K'])
    optimal_distance_method = optimal_hyperparameters['Distance Method']
    optimal_tree_method = optimal_hyperparameters['Tree Method']
    optimal_voting_method = optimal_hyperparameters['Voting Method']

    # User selection
    selected_k = st.selectbox('Select Number of Neighbors (K):', k_values, index=k_values.index(optimal_k))
    selected_distance_method = st.selectbox('Select Distance Method:', distance_methods, index=distance_methods.index(optimal_distance_method))
    selected_tree_method = st.selectbox('Select Tree Method:', tree_methods, index=tree_methods.index(optimal_tree_method))
    selected_voting_method = st.selectbox('Select Voting Method:', voting_methods, index=voting_methods.index(optimal_voting_method))

    # Display optimal parameters
    st.write(f"Optimal Number of Neighbors (K): {optimal_k}")
    st.write(f"Optimal Distance Method: {optimal_distance_method}")
    st.write(f"Optimal Tree Method: {optimal_tree_method}")
    st.write(f"Optimal Voting Method: {optimal_voting_method}")

    # Train final model with selected parameters
    final_knn = KNeighborsClassifier(
        n_neighbors=selected_k,
        algorithm=selected_tree_method,
        metric=selected_distance_method,
        weights=selected_voting_method
    )
    final_knn.fit(X_train_pca, y_train)

    # Predictions
    y_pred = final_knn.predict(X_test_pca)
    y_proba = final_knn.predict_proba(X_test_pca)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Display results
    st.write(f"Model Accuracy: {accuracy:.4f}")
    
    st.write("\nConfusion Matrix:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for KNN + SMOTE')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt.gcf())

    st.write("\nClassification Report:")
    st.write(pd.DataFrame(class_report))

    # ROC Curve
    n_classes = len(np.unique(y))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for KNN + SMOTE Classification')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())

    st.write("AUC for each class:", roc_auc)

#Gradient Boost
elif model == "Gradient Boost Classifier":
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_mod = df_finale

    # Extract features and target
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # PCA for dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_interactions)

    # Split the data into training and testing sets
    X_train_pca, X_test_pca, y_train_diag, y_test_diag = train_test_split(X_pca, y_diag, test_size=z, random_state=sta)

    gbm_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbm_classifier.fit(X_train_pca, y_train_diag)

    gbm_pred_diag = gbm_classifier.predict(X_test_pca)
    gbm_accuracy_diag = accuracy_score(y_test_diag, gbm_pred_diag)
    gbm_pred_train = gbm_classifier.predict(X_train_pca)
    gbm_accuracy_diag_train = accuracy_score(y_train_diag, gbm_pred_train)

    gbm_confusion = confusion_matrix(y_test_diag, gbm_pred_diag)

    st.write("GBM Train Accuracy:", gbm_accuracy_diag_train * 100.0)
    st.write("GBM Test Accuracy:", gbm_accuracy_diag * 100.0)

    st.write("GBM Confusion Matrix:")
    st.write(gbm_confusion)
    plt.figure(figsize=(8, 6))

    sns.heatmap(gbm_confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for GBM')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())

    def plot_confusion_matrix_heatmap(conf_matrix, class_names, title):
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)

    st.subheader("Correlation Heatmap for GBM:")
    class_names = np.unique(y_test_diag)

    plot_confusion_matrix_heatmap(gbm_confusion, class_names, title="GBM Confusion Matrix")
    numeric_columns = df_mod.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df_mod[numeric_columns].corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.show()
    st.pyplot()
    n_classes_gbm = len(np.unique(y_test_diag))
    y_score = gbm_classifier.predict_proba(X_test_pca)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Assuming you have binary classification, modify the loop if it's multi-class
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_diag == class_names[i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 8))
    colors = ['darkorange', 'green', 'blue', 'purple', 'brown']  # You may need to adjust the colors based on the number of classes

    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    st.subheader("ROC Curve:")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("ROC.png")
    st.image("ROC.png")

    
    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

    conf_matrix = confusion_matrix(y_test_diag, gbm_pred_diag)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#GBC + SMOTE
elif model == "Gradient Boost + Smote":
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming df_mod is your dataset
    df_mod = df_finale

    # Extract features and target
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_interactions, y_diag)

    # PCA for dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_resampled)

    # Split the data into training and testing sets
    X_train_pca, X_test_pca, y_train_diag, y_test_diag = train_test_split(X_pca, y_resampled, test_size=z, random_state=sta)

    gbm_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbm_classifier.fit(X_train_pca, y_train_diag)

    gbm_pred_diag = gbm_classifier.predict(X_test_pca)
    gbm_accuracy_diag = accuracy_score(y_test_diag, gbm_pred_diag)
    gbm_pred_train = gbm_classifier.predict(X_train_pca)
    gbm_accuracy_diag_train = accuracy_score(y_train_diag, gbm_pred_train)

    gbm_confusion = confusion_matrix(y_test_diag, gbm_pred_diag)

    st.write("GBM Train Accuracy:", gbm_accuracy_diag_train * 100.0)
    st.write("GBM Test Accuracy:", gbm_accuracy_diag * 100.0)

    st.write("GBM Confusion Matrix:")
    st.write(gbm_confusion)

    def plot_confusion_matrix_heatmap(conf_matrix, class_names, title):
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)

    st.subheader("Correlation Heatmap for GBM:")
    class_names = np.unique(y_test_diag)

    plot_confusion_matrix_heatmap(gbm_confusion, class_names, title="GBM Confusion Matrix")
    numeric_columns = df_mod.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df_mod[numeric_columns].corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
    st.pyplot()

    n_classes_gbm = len(np.unique(y_test_diag))
    y_score = gbm_classifier.predict_proba(X_test_pca)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Assuming you have binary classification, modify the loop if it's multi-class
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_diag == class_names[i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 8))
    colors = ['darkorange', 'green', 'blue', 'purple', 'brown']  # You may need to adjust the colors based on the number of classes

    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    st.subheader("ROC Curve:")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("ROC.png")
    st.image("ROC.png")
    
    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

    conf_matrix = confusion_matrix(y_test_diag, gbm_pred_diag)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#GBM + Adaboost 
elif model == "Gradient Boost + Adaboost":
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    df_mod = df_finale

    # Extract features and target
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # PCA for dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_interactions)

    # Split the data into training and testing sets
    X_train_pca, X_test_pca, y_train_diag, y_test_diag = train_test_split(X_pca, y_diag, test_size=z, random_state=sta)

    # Create a base Gradient Boosting Classifier
    base_gbm_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # AdaBoost with Gradient Boosting Classifier as the base estimator
    adaboost_gbm_classifier = AdaBoostClassifier(base_gbm_classifier, n_estimators=50, random_state=sta)
    adaboost_gbm_classifier.fit(X_train_pca, y_train_diag)

    # Predictions and Metrics
    adaboost_gbm_pred_train = adaboost_gbm_classifier.predict(X_train_pca)
    adaboost_gbm_accuracy_train = accuracy_score(y_train_diag, adaboost_gbm_pred_train)

    adaboost_gbm_pred_test = adaboost_gbm_classifier.predict(X_test_pca)
    adaboost_gbm_accuracy_test = accuracy_score(y_test_diag, adaboost_gbm_pred_test)

    adaboost_gbm_confusion = confusion_matrix(y_test_diag, adaboost_gbm_pred_test)

    st.write("AdaBoost on Gradient Boosting Classifier Train Accuracy:", adaboost_gbm_accuracy_train * 100.0)
    st.write("AdaBoost on Gradient Boosting Classifier Test Accuracy:", adaboost_gbm_accuracy_test * 100.0)

    st.write("AdaBoost on Gradient Boosting Classifier Confusion Matrix:")
    st.write(adaboost_gbm_confusion)

    plt.figure(figsize=(8, 6))
    sns.heatmap(adaboost_gbm_confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Gradient Boost Classifier + Adaboost')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())

    def plot_confusion_matrix_heatmap(conf_matrix, class_names, title):
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)

    st.subheader("Correlation Heatmap for AdaBoost on GBM:")
    class_names = np.unique(y_test_diag)

    plot_confusion_matrix_heatmap(adaboost_gbm_confusion, class_names, title="AdaBoost on GBM Confusion Matrix")
    numeric_columns = df_mod.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df_mod[numeric_columns].corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
    st.pyplot()
    n_classes_adaboost_gbm = len(np.unique(y_test_diag))
    adaboost_gbm_y_score = adaboost_gbm_classifier.predict_proba(X_test_pca)
    adaboost_gbm_fpr = dict()
    adaboost_gbm_tpr = dict()
    adaboost_gbm_roc_auc = dict()

    # Assuming you have binary classification, modify the loop if it's multi-class
    for i in range(len(class_names)):
        adaboost_gbm_fpr[i], adaboost_gbm_tpr[i], _ = roc_curve(y_test_diag == class_names[i], adaboost_gbm_y_score[:, i])
        adaboost_gbm_roc_auc[i] = auc(adaboost_gbm_fpr[i], adaboost_gbm_tpr[i])

    plt.figure(figsize=(8, 8))
    colors = ['darkorange', 'green', 'blue', 'purple', 'brown']  # You may need to adjust the colors based on the number of classes

    for i, color in zip(range(len(class_names)), colors):
        plt.plot(adaboost_gbm_fpr[i], adaboost_gbm_tpr[i], color=color, lw=2,
                 label=f'Class {class_names[i]} (AUC = {adaboost_gbm_roc_auc[i]:.2f})')

    st.subheader("ROC Curve for AdaBoost on GBM:")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for AdaBoost on GBM')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("ROC_AdaBoost_on_GBM.png")
    st.image("ROC_AdaBoost_on_GBM.png")
    
    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

    conf_matrix = confusion_matrix(y_test_diag, adaboost_gbm_pred_test)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#GBM + Bagging 
elif model == "Gradient Boost + Bagging":
    from sklearn.ensemble import BaggingClassifier

    df_mod = df_finale

    # Extract features and target
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # PCA for dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_interactions)

    # Split the data into training and testing sets
    X_train_pca, X_test_pca, y_train_diag, y_test_diag = train_test_split(X_pca, y_diag, test_size=z, random_state=sta)

    # Gradient Boosting Classifier
    gbm_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Bagging with Gradient Boosting Classifier
    bagging_gbm_classifier = BaggingClassifier(base_estimator=gbm_classifier, n_estimators=10, random_state=42)
    bagging_gbm_classifier.fit(X_train_pca, y_train_diag)

    # Predictions and Metrics
    bagging_gbm_pred_diag = bagging_gbm_classifier.predict(X_test_pca)
    bagging_gbm_accuracy_diag = accuracy_score(y_test_diag, bagging_gbm_pred_diag)
    bagging_gbm_pred_train = bagging_gbm_classifier.predict(X_train_pca)
    bagging_gbm_accuracy_diag_train = accuracy_score(y_train_diag, bagging_gbm_pred_train)

    bagging_gbm_confusion = confusion_matrix(y_test_diag, bagging_gbm_pred_diag)

    st.write("Bagging with GBM Train Accuracy:", bagging_gbm_accuracy_diag_train * 100.0)
    st.write("Bagging with GBM Test Accuracy:", bagging_gbm_accuracy_diag * 100.0)

    st.write("Bagging with GBM Confusion Matrix:")
    st.write(bagging_gbm_confusion)
    plt.figure(figsize=(8, 6))
    sns.heatmap(bagging_gbm_confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Gradient Boost Classifier + Bagging')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())

    def plot_confusion_matrix_heatmap(conf_matrix, class_names, title):
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)

    st.subheader("Correlation Heatmap for Bagging with GBM:")
    class_names = np.unique(y_test_diag)

    plot_confusion_matrix_heatmap(bagging_gbm_confusion, class_names, title="Bagging with GBM Confusion Matrix")
    numeric_columns = df_mod.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df_mod[numeric_columns].corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
    st.pyplot()
    n_classes_gbm = len(np.unique(y_test_diag))
    y_score = bagging_gbm_classifier.predict_proba(X_test_pca)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Assuming you have binary classification, modify the loop if it's multi-class
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_diag == class_names[i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 8))
    colors = ['darkorange', 'green', 'blue', 'purple', 'brown']  # You may need to adjust the colors based on the number of classes

    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    st.subheader("ROC Curve:")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("ROC_Bagging_with_GBM.png")
    st.image("ROC_Bagging_with_GBM.png")
    
    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

    conf_matrix = confusion_matrix(y_test_diag, bagging_gbm_pred_diag)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#Decision Tree 
elif model == 'Decision Tree':
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
    import matplotlib.pyplot as plt

    df_mod = df_finale

    # Extract features and target
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # PCA for dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_interactions)

    # Split the data into training and testing sets
    X_train_pca, X_test_pca, y_train_diag, y_test_diag = train_test_split(X_pca, y_diag, test_size=z, random_state=sta)


    criterion = st.selectbox("Select Criterion", ['gini', 'entropy'])
    max_depth_user = st.slider("Select Max Depth", 1, 10, 5)

    # Decision Tree Classifier for user-selected depth
    decision_tree_user = DecisionTreeClassifier(max_depth=max_depth_user, criterion=criterion)
    decision_tree_user.fit(X_train_pca, y_train_diag)

    # Make predictions on test set (not training set)
    y_pred = decision_tree_user.predict(X_test_pca)
    
    # Calculate metrics using test predictions
    accuracy_test = accuracy_score(y_test_diag, y_pred)
    error_rate_test = 1 - accuracy_test

    # Display metrics
    st.write(f"User-Selected Max Depth: {max_depth_user}")
    st.write(f"Test Accuracy: {accuracy_test*100:.2f}%")
    st.write(f"Test Error Rate: {error_rate_test*100:.2f}%")

    # Calculate and display confusion matrix
    conf_matrix = confusion_matrix(y_test_diag, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Decision Tree')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt.gcf())

    # Rest of your code...

#Decision Tree with Smote
elif model == "Decision Tree + Smote":
    from imblearn.over_sampling import SMOTE
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
    import matplotlib.pyplot as plt

    df_mod = df_finale

    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Apply SMOTE before splitting the data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_interactions, y_diag)

    # PCA after SMOTE
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_resampled)

    # Split the resampled data
    X_train_pca, X_test_pca, y_train_resampled, y_test_resampled = train_test_split(X_pca, y_resampled, test_size=0.2, random_state=42)

    criterion = st.selectbox("Select Criterion", ['gini', 'entropy'])
    max_depth_user = st.slider("Select Max Depth", 1, 10, 5)

    decision_tree_user = DecisionTreeClassifier(max_depth=max_depth_user, criterion=criterion)
    decision_tree_user.fit(X_train_pca, y_train_resampled)

    # Make predictions on test set
    y_pred = decision_tree_user.predict(X_test_pca)
    
    # Calculate metrics using test predictions
    accuracy_test = accuracy_score(y_test_resampled, y_pred)
    error_rate_test = 1 - accuracy_test

    st.write(f"User-Selected Max Depth: {max_depth_user}")
    st.write(f"Test Accuracy: {accuracy_test*100:.2f}%")
    st.write(f"Test Error Rate: {error_rate_test*100:.2f}%")

    # Calculate and display confusion matrix
    conf_matrix = confusion_matrix(y_test_resampled, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Decision Tree + SMOTE')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt.gcf())

    # Rest of your visualization code...

#Decision Tree with Adaboost
elif model == "Decision Tree + Adaboost":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import seaborn as sns

    # Prepare the data
    X = df_filtered.drop(columns=['Diagnosis', 'Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Grade'])
    y = df_filtered['Diagnosis']

    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    # Create and train AdaBoost classifier
    base_estimator = DecisionTreeClassifier(max_depth=3)
    adaboost_clf = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        random_state=42
    )
    adaboost_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = adaboost_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    # Display results
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Error Rate: {error_rate:.2f}")
    st.write("\nConfusion Matrix:")
    st.write(conf_matrix)
    report = pd.DataFrame(classification_rep)
    st.write("\nClassification Report:")
    st.write(report)

    # Plotting accuracy and error rate
    plt.figure(figsize=(12, 5))
    
    # First subplot for metrics
    plt.subplot(1, 2, 1)
    metrics = ['Accuracy', 'Error Rate']
    values = [accuracy, error_rate]
    colors = ['green', 'red']
    
    plt.bar(metrics, values, color=colors)
    plt.ylim(0, 1)  # Set y-axis limits between 0 and 1
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    
    # Add value labels on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')

    # Second subplot for confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    st.pyplot()

#Decision tree with Bagging
elif model == "Decision Tree + Bagging" :
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt

    X = df_filtered.drop(columns=['Diagnosis', 'Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Grade'])
    y = df_filtered['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    n_estimators = st.slider("Select Number of Estimators", 1, 100, 50)
    random_state = st.slider("Select Random State", 0, 100, 42)

    bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=n_estimators, random_state=random_state)

    bagging_clf.fit(X_train, y_train)
    y_pred = bagging_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy  
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict= True)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Error Rate: {error_rate:.2f}")
    st.write("\nConfusion Matrix:")
    st.write(conf_matrix)
    report = pd.DataFrame(classification_rep)
    st.write("\nClassification Report:")
    st.write(report)


    # First plot - Metrics
    plt.figure(figsize=(8, 6))
    plt.plot([accuracy] * n_estimators, label='Accuracy', marker='o')
    plt.plot([error_rate] * n_estimators, label='Error Rate', marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Metric')
    plt.title('Accuracy and Error Rate vs Number of Estimators')
    plt.legend()
    st.pyplot(plt)
    plt.close()

    # Second plot - Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Decision Tree + Bagging')
    st.pyplot(plt)
    plt.close()
    
    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])

#Logistic Regression
elif model == 'Logistic Regression':
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, 
        confusion_matrix, 
        classification_report,
        roc_curve,
        auc
    )
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare the data
    df_mod = df_finale
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_diag)

    # PCA for dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_interactions)

    # Split the data
    X_train_pca, X_test_pca, y_train_encoded, y_test_encoded = train_test_split(X_pca, y_encoded, test_size=z, random_state=sta)

    # Train model
    logreg = LogisticRegression(multi_class='multinomial', max_iter=1000)
    logreg.fit(X_train_pca, y_train_encoded)

    # Make predictions
    y_pred_train = logreg.predict(X_train_pca)
    y_pred_test = logreg.predict(X_test_pca)
    y_pred_proba = logreg.predict_proba(X_test_pca)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train_encoded, y_pred_train)
    test_accuracy = accuracy_score(y_test_encoded, y_pred_test)
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_test)

    # Display results
    st.write("Logistic Regression Metrics:")
    st.write(f"Train Accuracy: {train_accuracy*100:.2f}%")
    st.write(f"Test Accuracy: {test_accuracy*100:.2f}%")
    st.write(f"Train Error Rate: {(1-train_accuracy)*100:.2f}%")
    st.write(f"Test Error Rate: {(1-test_accuracy)*100:.2f}%")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt.gcf())
    plt.close()

    # Plot ROC curves for each class
    plt.figure(figsize=(8, 6))
    n_classes = len(np.unique(y_encoded))
    
    for i in range(n_classes):
        y_test_binary = (y_test_encoded == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        fpr, tpr, _ = roc_curve(y_test_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())
    plt.close()

    # Display classification report
    class_report = classification_report(y_test_encoded, y_pred_test, output_dict=True)
    st.write("\nClassification Report:")
    st.write(pd.DataFrame(class_report).transpose())

    # Display confusion matrix details
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("\nConfusion Matrix Details:")
        st.write(f"True Negative: {tn}")
        st.write(f"False Positive: {fp}")
        st.write(f"False Negative: {fn}")
        st.write(f"True Positive: {tp}")
    else:  # Multiclass
        st.write("\nConfusion Matrix Details:")
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} predicted as Class {j}: {conf_matrix[i][j]}")

#Logistic + Smote
elif model == "Logistic + Smote":
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from imblearn.over_sampling import SMOTE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import streamlit as st

    df_mod = df_finale

    # Extract features and target
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_interactions, y_diag)

    # Label encode the target
    le = LabelEncoder()
    y_resampled_encoded = le.fit_transform(y_resampled)

    # PCA for dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_resampled)

    # Split the data
    X_train_pca, X_test_pca, y_train_encoded, y_test_encoded = train_test_split(X_pca, y_resampled_encoded, test_size=z, random_state=sta)

    # Train model
    logreg = LogisticRegression(multi_class='multinomial', max_iter=1000)
    logreg.fit(X_train_pca, y_train_encoded)

    # Make predictions
    y_pred_train = logreg.predict(X_train_pca)
    y_pred_test = logreg.predict(X_test_pca)
    y_pred_proba = logreg.predict_proba(X_test_pca)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train_encoded, y_pred_train)
    test_accuracy = accuracy_score(y_test_encoded, y_pred_test)

    # Display metrics
    st.write("Logistic Regression with SMOTE Metrics:")
    st.write(f"Train Accuracy: {train_accuracy*100:.2f}%")
    st.write(f"Test Accuracy: {test_accuracy*100:.2f}%")
    st.write(f"Train Error Rate: {(1-train_accuracy)*100:.2f}%")
    st.write(f"Test Error Rate: {(1-test_accuracy)*100:.2f}%")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Logistic Regression + SMOTE')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt.gcf())
    plt.close()

    # ROC Curves
    plt.figure(figsize=(8, 6))
    n_classes = len(np.unique(y_resampled_encoded))
    
    for i in range(n_classes):
        y_test_binary = (y_test_encoded == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        fpr, tpr, _ = roc_curve(y_test_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Logistic Regression + SMOTE')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())
    plt.close()

    # Classification Report
    class_report = classification_report(y_test_encoded, y_pred_test, output_dict=True)
    st.write("\nClassification Report:")
    st.write(pd.DataFrame(class_report).transpose())

    # Confusion Matrix Details
    st.write("\nConfusion Matrix Details:")
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[0])):
            st.write(f"Class {i} predicted as Class {j}: {conf_matrix[i][j]}")

#Logistic + Bagging
elif model == 'Logistic + Bagging':
    from sklearn.ensemble import BaggingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Prepare the data
    df_mod = df_finale
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_diag)

    # PCA for dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_interactions)

    # Split the data
    X_train_pca, X_test_pca, y_train_encoded, y_test_encoded = train_test_split(X_pca, y_encoded, test_size=z, random_state=sta)

    # Create base classifier
    base_classifier = LogisticRegression(multi_class='multinomial', max_iter=1000)
    
    # Create and train Bagging classifier
    bagging_classifier = BaggingClassifier(
        estimator=base_classifier,
        n_estimators=10,
        random_state=sta
    )
    bagging_classifier.fit(X_train_pca, y_train_encoded)

    # Make predictions
    y_pred_train = bagging_classifier.predict(X_train_pca)
    y_pred_test = bagging_classifier.predict(X_test_pca)
    y_pred_proba = bagging_classifier.predict_proba(X_test_pca)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train_encoded, y_pred_train)
    test_accuracy = accuracy_score(y_test_encoded, y_pred_test)

    # Display metrics
    st.write("Logistic Regression with Bagging Metrics:")
    st.write(f"Train Accuracy: {train_accuracy*100:.2f}%")
    st.write(f"Test Accuracy: {test_accuracy*100:.2f}%")
    st.write(f"Train Error Rate: {(1-train_accuracy)*100:.2f}%")
    st.write(f"Test Error Rate: {(1-test_accuracy)*100:.2f}%")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Logistic Regression + Bagging')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt.gcf())
    plt.close()

    # ROC Curves for each class
    plt.figure(figsize=(8, 6))
    n_classes = len(np.unique(y_encoded))
    
    for i in range(n_classes):
        y_test_binary = (y_test_encoded == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        fpr, tpr, _ = roc_curve(y_test_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Logistic Regression + Bagging')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())
    plt.close()

    # Classification Report
    class_report = classification_report(y_test_encoded, y_pred_test, output_dict=True)
    st.write("\nClassification Report:")
    st.write(pd.DataFrame(class_report).transpose())

    # Confusion Matrix Details
    st.write("\nConfusion Matrix Details:")
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[0])):
            st.write(f"Class {i} predicted as Class {j}: {conf_matrix[i][j]}")

    # Add feature importance if available
    st.write("\nModel Parameters:")
    st.write(f"Number of base estimators: {bagging_classifier.n_estimators}")
    st.write(f"Base estimator: Logistic Regression")

#Logistic + Adaboost
elif model == 'Logistic + AdaBoost':
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Prepare the data
    df_mod = df_finale
    X_interactions = df_mod.drop(columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_diag)

    # PCA for dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_interactions)

    # Split the data
    X_train_pca, X_test_pca, y_train_encoded, y_test_encoded = train_test_split(X_pca, y_encoded, test_size=z, random_state=sta)

    # Create base classifier
    base_classifier = LogisticRegression(multi_class='multinomial', max_iter=1000)
    
    # Create and train AdaBoost classifier with updated parameter name
    adaboost_classifier = AdaBoostClassifier(
        estimator=base_classifier,
        n_estimators=50,
        random_state=sta
    )
    adaboost_classifier.fit(X_train_pca, y_train_encoded)

    # Make predictions
    y_pred_train = adaboost_classifier.predict(X_train_pca)
    y_pred_test = adaboost_classifier.predict(X_test_pca)
    y_pred_proba = adaboost_classifier.predict_proba(X_test_pca)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train_encoded, y_pred_train)
    test_accuracy = accuracy_score(y_test_encoded, y_pred_test)

    # Display metrics
    st.write("Logistic Regression with AdaBoost Metrics:")
    st.write(f"Train Accuracy: {train_accuracy*100:.2f}%")
    st.write(f"Test Accuracy: {test_accuracy*100:.2f}%")
    st.write(f"Train Error Rate: {(1-train_accuracy)*100:.2f}%")
    st.write(f"Test Error Rate: {(1-test_accuracy)*100:.2f}%")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Logistic Regression + AdaBoost')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt.gcf())
    plt.close()

    # ROC Curves for each class
    plt.figure(figsize=(8, 6))
    n_classes = len(np.unique(y_encoded))
    
    for i in range(n_classes):
        y_test_binary = (y_test_encoded == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        fpr, tpr, _ = roc_curve(y_test_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Logistic Regression + AdaBoost')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())
    plt.close()

    # Classification Report
    class_report = classification_report(y_test_encoded, y_pred_test, output_dict=True)
    st.write("\nClassification Report:")
    st.write(pd.DataFrame(class_report).transpose())

    # Confusion Matrix Details
    st.write("\nConfusion Matrix Details:")
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[0])):
            st.write(f"Class {i} predicted as Class {j}: {conf_matrix[i][j]}")

    # Add feature importance if available
    st.write("\nModel Parameters:")
    st.write(f"Number of base estimators: {adaboost_classifier.n_estimators}")
    st.write(f"Base estimator: Logistic Regression")

#XGBoost 
elif model == "XGBoost":
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
    from imblearn.over_sampling import SMOTE
    import xgboost as xgb
    import matplotlib.pyplot as plt

    # Assuming df_final is your dataset
    df_mod = df_finale

    X_interactions = df_mod.drop(
        columns=['Gender', 'Education Level', 'Marital Status', 'Relationship Status', 'Diagnosis', 'Grade'])
    y_diag = df_mod['Diagnosis']

    # Apply SMOTE to address class imbalance
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_interactions, y_diag)

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_resampled)

    from sklearn.preprocessing import LabelEncoder

    # Convert categorical labels to numeric values
    label_encoder = LabelEncoder()
    y_resampled_encoded = label_encoder.fit_transform(y_resampled)

    # Train-test split with the encoded target variable
    X_train_pca, X_test_pca, y_train_encoded, y_test_encoded = train_test_split(X_pca, y_resampled_encoded, test_size=z, random_state=sta)

    # Train XGBoost model
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X_train_pca, y_train_encoded)


    # Predictions
    xgb_pred_diag = xgb_classifier.predict(X_test_pca)

    # Evaluate the model
    xgb_accuracy = accuracy_score(y_test_encoded, xgb_pred_diag)
    xgb_confusion = confusion_matrix(y_test_encoded, xgb_pred_diag)
    xgb_classification_report = classification_report(y_test_encoded, xgb_pred_diag, output_dict=True)
    st.write('XGBoost Accuracy:', xgb_accuracy)
    st.write("XGBoost Confusion Matrix:")
    st.write(xgb_confusion)

    plt.figure(figsize=(8, 6))
    sns.heatmap(xgb_confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for XGBoost')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Display the plot within Streamlit
    st.pyplot(plt.gcf())
    
    st.write("XGBoost Classification Report:")
    report = pd.DataFrame(xgb_classification_report)
    st.write(report)
    
    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

    conf_matrix = confusion_matrix(y_test_encoded, xgb_pred_diag)

        # Plot ROC curve
    y_probs = xgb_classifier.predict_proba(X_test_pca)
    n_classes = len(np.unique(y_resampled))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_encoded == i, y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(pd.get_dummies(y_test_encoded).values.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve (AUC = {:.2f}) for class {}'.format(roc_auc[i], i))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {:.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - XGBoost')
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig('roc_curve_xgb.png')

    # Display the plot
    st.pyplot()

    # Display the AUC values
    st.write("AUC for each class:", roc_auc)

    # Unpack all four values from the confusion matrix
    if len(conf_matrix) == 2:  # Binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        st.write("True Negative:", tn)
        st.write("False Positive:", fp)
        st.write("False Negative:", fn)
        st.write("True Positive:", tp)
    else:
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                st.write(f"Class {i} and Predicted as Class {j}:", conf_matrix[i][j])
    
#Deep Learning
elif model == "Deep Learning" :
    import numpy as np
    import pandas as pd
    import streamlit as st
    import tensorflow as tf
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import accuracy_score
    from keras.utils import plot_model

    st.title("Convolutional Neural Network Model")

    # Prepare features
    # Select only numerical columns initially
    numerical_features = df_finale.select_dtypes(include=['float64', 'int64']).columns
    X_interactions = df_finale[numerical_features].copy()
    
    # One-hot encode categorical variables
    categorical_features = ['Education Level', 'Marital Status', 'Relationship Status', 'Gender']
    for feature in categorical_features:
        if feature in df_finale.columns:
            dummies = pd.get_dummies(df_finale[feature], prefix=feature)
            X_interactions = pd.concat([X_interactions, dummies], axis=1)

    # Target variable
    y_diag = df_finale['Diagnosis']

    # Standardize numerical features
    scaler = StandardScaler()
    X_interactions[numerical_features] = scaler.fit_transform(X_interactions[numerical_features])

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X_interactions, y_diag)

    # Split the data
    test = st.slider("Test Split value", min_value=0.30, max_value=0.90, step=0.01, value=0.70)    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test, random_state=42)

    # Create and train CNN
    def create_and_train_cnn(X, y, batch_size=32, epochs=50, dropout_rate=0.2):
        # One-hot encode the labels
        encoder = OneHotEncoder(sparse_output=False)
        y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

        # Convert data to float32 type and reshape
        X_reshaped = X.values.astype('float32').reshape((X.shape[0], X.shape[1], 1))
        y_encoded = y_encoded.astype('float32')

        # Convert to TensorFlow tensors
        X_tensor = tf.convert_to_tensor(X_reshaped)
        y_tensor = tf.convert_to_tensor(y_encoded)

        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in kf.split(X_reshaped):
            X_train_fold = tf.convert_to_tensor(X_reshaped[train_idx], dtype=tf.float32)
            y_train_fold = tf.convert_to_tensor(y_encoded[train_idx], dtype=tf.float32)
            X_val_fold = tf.convert_to_tensor(X_reshaped[val_idx], dtype=tf.float32)
            y_val_fold = tf.convert_to_tensor(y_encoded[val_idx], dtype=tf.float32)

            model.fit(X_train_fold, y_train_fold, 
                     epochs=epochs, 
                     batch_size=batch_size,
                     validation_data=(X_val_fold, y_val_fold),
                     verbose=0)
            
            _, acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            cv_scores.append(acc)

        # Final training on whole dataset
        model.fit(X_tensor, y_tensor, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Make predictions
        y_pred = model.predict(X_tensor)
        final_accuracy = accuracy_score(np.argmax(y_encoded, axis=1), np.argmax(y_pred, axis=1))

        return model, cv_scores, final_accuracy
    
    # Train and evaluate
    trained_model, accuracies, final_accuracy = create_and_train_cnn(X_train, y_train)

    # Display results
    st.write(f"Final Accuracy on Test Set: {final_accuracy * 100:.2f}%")
    st.write(f"Cross-Validation Accuracies: {accuracies}")
    st.write(f"Average CV Accuracy: {np.mean(accuracies) * 100:.2f}%")

    # Model information
    st.subheader("Model Information:")
    st.write("Input Shape:", trained_model.input_shape)
    st.write("Output Shape:", trained_model.output_shape)
    st.write("Number of Layers:", len(trained_model.layers))
