"""
Insurance Linear Reg ML Project

Author: Ray Lopez
Last Updated: 2/12/24

"""
#Importing Libraries
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from matplotlib.pyplot import figure
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Defining function to plot our ROC curve
def plot_roc_curve(fpr, tpr, name):
    fig = figure(figsize=(4,3))
    fig.set_dpi(300)
    plt.plot(fpr, tpr, color="blue", label="ROC")
    plt.plot([0,1], [0,1], color="red", linestyle="--", label="Guessing")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(name + " ROC Curve")
    plt.legend()
    plt.show()

#Defining function to plot our Confusion Matrix
def plot_conf_matrix(conf_mat, name):
    fig, ax = plt.subplots()
    fig.set_size_inches(3,3)
    fig.set_dpi(300)
    ax = sns.heatmap(conf_mat, annot=True, fmt=str())
    plt.title(name + " Confusion Matrix")
    
#Defining function to plot single value regression.
def plot_single_val_reg(x_vals, y_vals):
    output = sns.lmplot(x=x_vals, y=y_vals, data=df, aspect=1.5, height=6)
    plt.xlabel(x_vals)
    plt.ylabel(y_vals)
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=df[x_vals], y=df[y_vals])
    plt.title('%s vs. %s' % (x_vals, y_vals), fontsize=36, ha='center')
    plt.suptitle('y = ' + str(round(intercept,3)) + ' + ' + str(round(slope,3)) + 'x' + ' | r = ' + str(round(r,3)), fontsize=10, ha='center')
    plt.show()
    plt.clf()

#Defining function to plot a correlation Matrix. Function also performs mean normalization on the data.
def plot_correlation_matrix(data_frame):
    normalized_df = (data_frame-data_frame.mean())/data_frame.std()
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    fig.set_dpi(300)
    ax = sns.heatmap(normalized_df.corr(), cmap = 'Wistia', annot=True)
    plt.title("Correlation Matrix", fontsize=32)
    plt.show()
    plt.clf()
    
#Defining function to create a violin plot.
def plot_violin_plot(x_var, y_var):
    fig = plt.figure()
    ax = fig.subplots()
    fig.set_size_inches(8,8)
    fig.set_dpi(300)
    ax = sns.violinplot(x=x_var, y=y_var, data=df, palette='flare')
    ax.set_title("Violin Plot: " + x_var + ' vs. ' + y_var)
    plt.show()
    plt.clf()
    
#Defining function to create split violin part.
def plot_violin_plot_w_hue(x_var, y_var, hue_var):
    fig = plt.figure()
    ax = fig.subplots()
    fig.set_size_inches(8,8)
    fig.set_dpi(300)
    ax = sns.violinplot(x=x_var, y=y_var, hue=hue_var, data=df, palette='pastel', split=True)
    ax.set_title("Violin Plot: " + x_var + ' vs. ' + y_var + ' | split by ' + hue_var)
    plt.show()
    plt.clf()
    
#Defining function to plot the linearity of a linear regression model    
def plot_linear_reg_linearity(y_test_var, y_pred_var, c):
    fig = plt.figure()
    ax = fig.subplots()
    fig.set_size_inches(8,8)
    fig.set_dpi(300)
    ax = sns.scatterplot(x=y_test_var, y=y_pred_var, color=c)
    ax.set_title('Check For Linearity\n Actual vs. Predicted Values')
    plt.grid()
    plt.show()
    plt.clf()

#Defining function to plot residual error normality/dist.
def plot_linear_reg_residuals(y_test_var, y_pred_var, c, c2):
    fig = plt.figure()
    ax = fig.subplots()
    fig.set_size_inches(8,8)
    fig.set_dpi(300)
    ax = sns.distplot((y_test_var - y_pred_var), color=c, kde=True)
    ax.axvline((y_test_var - y_pred_var).mean(), color = c2, linestyle='--')
    ax.set_title('Check for Residual Normality & Mean\nResidual Error')
    plt.grid()
    plt.show()
    plt.clf()

#Defining function to plot multivariant normality of a lin reg model.
def plot_linear_reg_multivar_norm(y_test_var, y_pred_var):
    fig = plt.figure()
    ax = fig.subplots()
    fig.set_size_inches(8,8)
    fig.set_dpi(300)
    _,(_,_,r) = scipy.stats.probplot((y_test_var-y_pred_var), fit=True, plot=ax)
    ax.set_title('Multivariant Normality\nQ-Q Plot')
    plt.grid()
    plt.show()
    plt.clf()
    
#Defining function to plot homoscedasticity of a linear reg model.    
def plot_linear_reg_homoscedasticity(y_test_var, y_pred_var, c):
    fig = plt.figure()
    ax = fig.subplots()
    fig.set_size_inches(8,8)
    fig.set_dpi(300)
    ax = sns.scatterplot(y=(y_test_var-y_pred_var), x=y_pred_var, color=c)
    ax.set_title('Homoscedasticity\nResidual vs. Predicted')
    plt.grid()
    plt.show()
    plt.clf()
    
#Importing Data Set
df = pd.read_csv('insurance.csv')

#Plotting Single Independant Variable Regressions
plot_single_val_reg('bmi', 'charges')
plot_single_val_reg('age', 'charges')

#Plotting Corrolation Matrix Of Continuous Variables
correlation_df= df.drop(['sex', 'smoker', 'region'], axis=1)
plot_correlation_matrix(correlation_df)

#Plotting Violin Plots For Sex, Smoking, and Region
plot_violin_plot('sex', 'charges')
plot_violin_plot('smoker', 'charges')
plot_violin_plot_w_hue('region', 'charges', 'smoker')
plot_violin_plot_w_hue('region', 'charges', 'sex')

#Preparing Data For Analysis
#Converting Boolean and Non-Quantitative Data
categorical_columns = ['sex', 'children', 'smoker', 'region']
df_encode = pd.get_dummies(data=df, prefix='DUM', prefix_sep='_', columns=categorical_columns, drop_first=True, dtype='int8')

display(df.columns.values)
display(df.shape)
display(df_encode.columns.values)
display(df_encode.shape)

#Transforming charges so it's normally distributed using a logorithmic transformation.
y_bc, lam, ci = boxcox(df_encode['charges'], alpha=.05)
df_encode['charges'] = np.log(df_encode['charges'])

#Splitting data for analysis
x = df_encode.drop('charges', axis=1)
y = df_encode['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=25)

#Setting intersect
x_train_0 = np.c_[np.ones((x_train.shape[0],1)), x_train]
x_test_0 = np.c_[np.ones((x_test.shape[0],1)), x_test]

#Setting Theta
theta = np.matmul(np.linalg.inv( np.matmul(x_train_0.T, x_train_0) ), np.matmul(x_train_0.T,y_train)) 

#Defining perameters of linear regresion model
parameter = ['theta_'+str(i) for i in range(x_train_0.shape[1])]
columns = ['intersect:x_0=1'] + list(x.columns.values)
parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})

#Scikit Learn Module
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

#Comparing NP Model To SKL Model
sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))

display(parameter_df)

#Calculating MSE and R-Square Value
y_preds =lin_reg.predict(x_test)
lin_reg_mse = mean_squared_error(y_preds, y_test)
lin_reg_r_square = lin_reg.score(x_test, y_test)

display(lin_reg_mse)
display(lin_reg_r_square)

#Checking for linearity of our model
plot_linear_reg_linearity(y_test, y_preds, 'b')

#Checking res vs preds
plot_linear_reg_residuals(y_test, y_preds, 'b', 'k')

#Checking multivar normality
plot_linear_reg_multivar_norm(y_test, y_preds)
                              
#Checking homoscedasticity
plot_linear_reg_homoscedasticity(y_test, y_preds, 'g')