# Age 15-18
# Gender 0 male, 1 female
# Ethnicity 0 Caucasian, 1 African American, 2 Asian, 3 Other
# Parental Education 0 None, 1 High School, 2 Some College, 3 Bachelors, 4 Higher
# Study time 0-20
# Absences 0-30
# Tutoring 0 No, 1 Yes
# Parental Support 0 None, 1 Low, 2 Moderate, 3 High, 4 Very High
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import check_normal_dist as cnd
import filtering as f
import seaborn as sns

import pandas as pd

df = pd.read_csv("Student_performance_data.csv")

#print(df.isnull().sum()) # There is not a single missing value

#print(df.duplicated().sum()) # There is no duplicated rows



# Pair Plot
sns.pairplot(data=df, vars=['GPA', 'ParentalSupport', 'GradeClass'])
#plt.show()

# Correlation Matrix Heatmap
correlation_matrix = df[['GPA', 'ParentalEducation','StudyTimeWeekly','Absences','Tutoring','ParentalSupport','Extracurricular','Sports','Music','Volunteering','GradeClass']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.show()

for col in df.columns:
    cnd.plot_histogram_qq(df, col)

for col in df.columns:
    cnd.check_normality(df, col)

for col in df.columns:
    cnd.check_skewness_kurtosis(df, col)

print(df.describe())
# Not a normal dist so we will use IQR
data_filtered = df.copy()
for col in data_filtered.columns:
    data_filtered = f.IQR(data_filtered, col)
    data_filtered = f.percentile_filter(data_filtered, col)
#print(data_filtered.describe())
#print(df)
#print(data_filtered)
# So after I performed filtering with IQR I got a lot of NaN values (outliers) which I need to remove now

nan_columns = data_filtered.columns[data_filtered.isna().any()].tolist()
print(nan_columns)
print(data_filtered[nan_columns].isna().sum())

nan_rows = data_filtered.isna().sum(axis=1)
print(nan_rows[nan_rows > 0])
# 840 rows are NaN after filtering

data_filtered.dropna(inplace=True)
#print(data_filtered)

#for col in data_filtered.columns:
    #cnd.plot_histogram_qq(data_filtered, col)

#for col in data_filtered.columns:
    #cnd.check_normality(data_filtered, col)

features = ['ParentalEducation','StudyTimeWeekly','Absences','Tutoring','ParentalSupport','Extracurricular','Sports','Music','Volunteering','GradeClass']
#features = ['StudyTimeWeekly', 'Tutoring', 'ParentalSupport', 'GradeClass', 'Absences']
X = data_filtered[features]
y = data_filtered['GPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

#model = RandomForestRegressor()
#model = LinearRegression()
model = Ridge()
#model = Lasso()
#model = GradientBoostingRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# Perform cross-validation
cv_mse = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')

# Convert negative MSE to positive
cv_mse = -cv_mse

print(f"Cross-Validation MSE: {cv_mse.mean()} ± {cv_mse.std()}")
print(f"Cross-Validation R-squared: {cv_r2.mean()} ± {cv_r2.std()}")

#now explain residual analysis and how to interpret it
residuals = y_test - y_pred

# Plot Residuals vs Fitted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
#plt.show()

# Histogram of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=20)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
#plt.show()

# Q-Q Plot of Residuals
from scipy import stats
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
#plt.show()

# Scale-Location Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Sqrt(|Residuals|)')
plt.title('Scale-Location Plot')
#plt.show()