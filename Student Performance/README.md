# Predicting Student GPA Using Machine Learning



## Project Overview

This project aims to predict student GPA (Grade Point Average) based on various features such as study time, absences, parental education, and more. Using a combination of data preprocessing, feature engineering, and machine learning models, the project seeks to create a robust predictive model with high accuracy. The Ridge Regression model is used, and the performance is evaluated using cross-validation techniques, with further steps suggested for model improvement.



## Motivation

Understanding the factors that influence student performance is crucial for educators. By predicting GPA, we can identify students who may need additional support and resources, thereby enhancing their academic outcomes. This project showcases my ability to work with real-world data, preprocess it, build predictive models, and interpret the results, making it a valuable addition to my portfolio.


### Data Description

**Data Source:**  
The dataset used in this project contains information on students, including features such as:

- **Age:** (15-18 years)
- **Gender:** (0 for male, 1 for female)
- **Ethnicity:** (0 for Caucasian, 1 for African American, 2 for Asian, 3 for Other)
- **Parental Education:** (0 for None, 1 for High School, 2 for Some College, 3 for Bachelor's, 4 for Higher)
- **Study Time:** (0-20 hours per week)
- **Absences:** (0-30 days)
- **Tutoring:** (0 for No, 1 for Yes)
- **Parental Support:** (0 for None, 1 for Low, 2 for Moderate, 3 for High, 4 for Very High)
- **Extracurricular Activities:** (0 for No, 1 for Yes)
- **Sports:** (0 for No, 1 for Yes)
- **Music:** (0 for No, 1 for Yes)
- **Volunteering:** (0 for No, 1 for Yes)
- **Grade Class:** (0 for Freshman, 1 for Sophomore, 2 for Junior, 3 for Senior)
- **GPA:** (Target variable)



## Data Visualization

To better understand the distribution of features and the target variable, several visualizations were created during the exploratory data analysis phase:

- **Histograms:** 
    - Histogram of each feature to understand its distribution.
    - Example: Histogram of GPA, Study Time, and Absences.
  
- **Q-Q Plots:** 
    - Q-Q Plots to check for normality of the distribution of the features.
    - Example: Q-Q Plot of GPA Residuals.
  
- **HeatMap:**
    - Grid of all the connections between features.
  
**[Placeholder for Images]**  


![Histogram and QQPlot of GPA](/Student%20Performance/Figures/Histogram-QQPlot-GPA.png)
![Histogram and QQPlot of Age](/Student%20Performance/Figures/Histogram-QQPlot-Age.png)
![Histogram and QQPlot of Absences](/Student%20Performance/Figures/Histogram-QQPlot-Absences.png)
![Histogram and QQPlot of Ethnicity](/Student%20Performance/Figures/Histogram-QQPlot-Ethnicity.png)
![Histogram and QQPlot of Extracurricular](/Student%20Performance/Figures/Histogram-QQPlot-Extracurricular.png)
![Histogram and QQPlot of Gender](/Student%20Performance/Figures/Histogram-QQPlot-Gender.png)
![Histogram and QQPlot of GradeClass](/Student%20Performance/Figures/Histogram-QQPlot-GradeClass.png)
![Histogram and QQPlot of Music](/Student%20Performance/Figures/Histogram-QQPlot-Musci.png)
![Histogram and QQPlot of ParentalEducation](/Student%20Performance/Figures/Histogram-QQPlot-ParentalEducation.png)
![Histogram and QQPlot of Sports](/Student%20Performance/Figures/Histogram-QQPlot-Sports.png)
![Histogram and QQPlot of StudentID](/Student%20Performance/Figures/Histogram-QQPlot-StudentID.png)
![Histogram and QQPlot of SttudyTime](/Student%20Performance/Figures/Histogram-QQPlot-SttudyTime.png)
![Histogram and QQPlot of Tutoring](/Student%20Performance/Figures/Histogram-QQPlot-Tutoring.png)
![Histogram and QQPlot of Volunteering](/Student%20Performance/Figures/Histogram-QQPlot-Volunteering.png)
![HeatMap for all features](/Student%20Performance/Figures/HeatMap-of-All.png)



## Data Preprocessing

The data preprocessing steps involved the following:

1. **Data Loading:**
   - The dataset was loaded into a pandas DataFrame, and initial checks for missing values and duplicates were performed to ensure data integrity.

2. **Data Filtering:**
   - Outliers were first handled using the Interquartile Range (IQR) method. After identifying that some data points were too extreme, additional filtering was applied using percentile thresholds. The functions for these filtering techniques are implemented in the `filtering.py` script.

3. **Feature Scaling:**
   - Numerical features, including categorical variables that were already encoded as numbers, were scaled using `StandardScaler` to standardize the data, ensuring that all features contribute equally to the model. No additional encoding techniques (like one-hot or ordinal encoding) were necessary because the categorical values were already represented numerically.

4. **Normality and Distribution Checks:**
   - The check_normal_dist.py script was used to plot histograms and Q-Q plots for each feature. Additionally, it performed normality tests (Shapiro-Wilk and D'Agostino's K-squared) and checked for skewness and kurtosis to evaluate how well each feature conformed to a normal distribution.

5. **Data Splitting:**
   - The dataset was split into training and testing sets to evaluate the model's performance. The training set was used to train the model, and the test set was used for final evaluation.



## Modeling

### Model Selection and Training

- Several models were considered, including Random Forest, Linear Regression, Ridge Regression, Lasso, and Gradient Boosting. After initial evaluation, Ridge Regression was selected due to its balance between simplicity and performance.
- The Ridge Regression model was trained on the training dataset, which included both numerical and categorical features (all represented as numbers).
- Cross-validation was used to evaluate the model's performance, ensuring that the results were robust and not dependent on a single train-test split.



## Model Evaluation

### Performance Metrics

- **Cross-Validation MSE:** 0.036059 ± 0.002883
- **Cross-Validation R-squared:** 0.953098 ± 0.003956

The low Mean Squared Error (MSE) and high R-squared values indicate that the model is accurate and explains a significant portion of the variance in the target variable (GPA).

### Residual Analysis

To ensure that the model's assumptions were met and to check for any potential issues, several residual analysis plots were generated:

- **Residuals vs. Fitted Values:** 
    - The residuals are randomly scattered around zero, indicating that the model’s predictions are unbiased and there are no obvious patterns in the errors.

    ![Residuals vs. Fitted](/Student%20Performance/Figures/Residuals_vs._Fitted.png)
  
- **Histogram of Residuals:** 
    - The histogram shows that the residuals are approximately normally distributed, with a slight skew. This suggests that most predictions are close to the actual values, though there may be some systematic under or over-predictions.

    ![Histogram of Residuals](/Student%20Performance/Figures/Histogram_Residuals.png)

- **Q-Q Plot of Residuals:** 
    - The Q-Q plot indicates that the residuals generally follow a normal distribution, with slight deviations at the tails. This is typical and suggests that the model is reasonably well-specified.

    ![Q-Q Plot of Residuals](/Student%20Performance/Figures/QQPlot_Residuals.png)

- **Scale-Location Plot:**
    - The Scale-Location plot shows that the residuals have a consistent spread across different levels of the fitted values, indicating homoscedasticity (constant variance of residuals).

    ![Scale-Location Plot](/Student%20Performance/Figures/ScaleLocationPlot.png)



## Further Improvements

While the current model demonstrates strong performance, there are several avenues for potential enhancement:

1. **Advanced Hyperparameter Tuning:**
   - Conduct a thorough hyperparameter tuning process using techniques like GridSearchCV or RandomizedSearchCV to optimize the model's parameters, such as the regularization strength in Ridge Regression.

2. **Exploring Alternative Models:**
   - Experiment with more complex models, such as Random Forest, Gradient Boosting, or even neural networks. These models might capture non-linear relationships that are not fully addressed by Ridge Regression.

3. **Additional Feature Engineering:**
   - Investigate the creation of interaction terms between variables, polynomial features, or other derived features that could improve model accuracy.

4. **Ensemble Methods:**
   - Consider combining multiple models (e.g., via stacking, bagging, or boosting) to improve predictive performance and reduce the risk of overfitting.

5. **External Validation:**
   - Validate the model using an external dataset to assess its generalizability. This could involve data from a different cohort or time period to ensure the model performs well on unseen data.

6. **Addressing Residual Skewness:**
   - Further investigate the slight skewness observed in the residuals. Applying transformations or adjusting the model to better account for this could lead to improved predictions.

7. **Incorporating Domain Expertise:**
   - Collaborate with domain experts to identify any potential features or patterns that may not be immediately obvious but could significantly enhance the model.

These strategies could further refine the model, enhance its accuracy, and ensure it performs well in a variety of contexts.



## Conclusion

This project demonstrates the successful development of a predictive model for student GPA using machine learning techniques. Through careful data preprocessing, feature selection, and model evaluation, the Ridge Regression model was able to achieve high accuracy and explain a significant portion of the variance in the target variable.

Key findings from this project include the strong negative correlation between absences and GPA, and the importance of considering both numerical and categorical features when building predictive models. The residual analysis confirmed that the model's assumptions were largely met, although there are opportunities for further refinement.

The use of the check_normal_dist.py script for normality testing and distribution analysis was crucial in ensuring that the data was appropriately prepared for modeling.

Overall, this project highlights my skills in working with real-world data, applying machine learning techniques, and interpreting model results. The potential areas for further improvement suggest that, while the model is already robust, there are avenues to explore for even better performance.

This project is a valuable addition to my portfolio, showcasing my ability to tackle complex data problems and build effective predictive models.
