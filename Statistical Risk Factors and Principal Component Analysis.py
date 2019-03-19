#
# =========================================================
# Computational Risk and Asset Management WS 17/18
# Problem Set 5, Week 5
# Statistical Risk Factors and Principal Component Analysis
# =========================================================
#
# Based on: 'Statistics and Data Analysis for Financial Engineering' (Ruppert)
#
# Prepared by Elmar Jakobs
#
# by Thi Ha Giang Vo and Lotta Rüter




#
# Description
# -----------
#
# This exercise uses yields on Treasury bonds at 11 maturities,
# T = 1, 3, and 6 months and 1, 2, 3, 5, 7, 10, 20, and 30 years.
#
# Daily yields were taken from a U.S. Treasury website for the
# time period January 2, 1990, to October 31, 2008.
#
# Goal: PCA to study how the curves change from day to day
#





# Setup
# -----

# Import packages for econometric analysis
#
# noinspection PyInterpreter
import numpy as np
import pandas as pd

import statsmodels.api as sm

import sklearn.decomposition as sck_dec


# Plotting library
#
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


# Import and configure the testing framework
#
from test_suite import Test
# For submission, please change the status to 'SOLN'
status = 'SOLN'

if status == 'SOLN':
    checker_results = pd.read_csv('results_ps5.csv').set_index('Task')






# Question 1a: Treasury Yield Curves
# ----------------------------------


# TODO: Read-in provided text-file with daily Treasury yields
#
yields_orig = pd.read_table('treasury_yields.txt', sep = '\s*')
yields_orig[0:5]



# TODO: Filter out 'NANs'
# E.g. 'Treasury discontinued the 20-year constant maturity series at the end of calendar year 1986
#       and reinstated that series on October 1, 1993.'
#   Hint: You can use the 'np.isnan' function
#
yields_df = yields_orig.dropna()
yields_df[0:5]



# Assign variables
#
dates  = np.asarray( yields_df[['Date']] )   # Select date column
yields = np.asarray( yields_df.ix[:, 1:12] ) # Take all yields
yields.shape


# TODO: Specify maturity of the yields
#   Hint: Use an 'np.array' to store the different maturities
#   Hint #2: Specify the time as a fraction of the year (e.g., 1 month -> 1/12)
#
# time = np.array([[1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30],] *len(yields))
time = np.array([1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])
time.shape


# TODO: Plot yield curves
#

fig = plt.gcf()
fig.set_size_inches(16, 10.5)

plt.subplot(1,1,1)
plt.plot(time, yields[0],  label = dates[0][0],   linestyle = '-',  marker = 'o' ) # x-axis: time, y-axis: yields
plt.plot(time, yields[485],  label = dates[485][0], linestyle = '--', marker = 'o' )
plt.plot(time, yields[-1],  label = dates[-1][0],  linestyle = ':',  marker = 'o' )
plt.legend(loc = 'lower right', frameon = True)
plt.xlabel('time')
plt.ylabel('yield')




# Check of intermediate result (1):
#
Test.assertEquals(np.round(yields_df.iloc[0][1],2), 3.67,  'incorrect result')
Test.assertEquals(np.round(time[0],3),              0.083, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(yields_df.iloc[0][2], 2), checker_results.loc[1][0], 'incorrect result')
    Test.assertEquals(np.round(time[1], 2),              checker_results.loc[2][0], 'incorrect result')







# Question 1b: Principal Component Analysis - Intuition
# -----------------------------------------------------


# Methodology and Goals
#   See lecture notes Ch. 7.5







# Question 1c+e: Class 'PCA'
# --------------------------



class PCA(object):


    def __init__(self, X):


        # X: Original data
        #
        self.X = np.matrix( X )



    def run_pca(self):

        #
        #
        # Transform X with P into Y such that Cov(Y) is a diagonal matrix:
        #     Y = X*P <=> Cov(Y) = 1/n*Y*Y' = 1/n*(X*P)(X*P)' = P'*Cov(X)*P
        #
        # Any symmetric matrix can be represented by Eigen-values (D) and vectors (E):
        #     Cov(X) = E*D*E^-1 = E*D*E'
        #
        # Choose Eigen-vectors to transform original data: P = E
        #     Cov(Y) = E'*Cov(X)*E = D
        #
        #


        # TODO: First step, demean X
        #
        self.X_dm = self.X - self.X.mean(0)
        # print(self.X_dm.shape)
        print('self.X_dm:', self.X_dm)


        # TODO: Second, obtain covariance matrix of demeaned matrix X
        #  Hint: You can use the 'np.cov' function
        #
        self.Cov_X = np.cov(self.X_dm.getT(), ddof = 1)   # 1/self.X.shape[0] * self.X_dm.getT() * self.X_dm


        # TODO: Third, calculate eigen-values and vectors
        #  Hint: Use the 'eig' function of the numpy.linalg package
        #
        eigen                 = np.linalg.eig(self.Cov_X)
        print('Eigen: ', eigen)

        eig_values_X          = np.matrix(eigen[0])     # Assign Eigenvalues as a 'np.matrix'
        print('Eigenvalues: ', eig_values_X)

        self.eig_values_X_mat = np.diagflat(np.array(eig_values_X))         # Lambda_X

        print('Eigenvalues: ', self.eig_values_X_mat)

        self.eig_vectors_X    = np.matrix(eigen[1])                   # Assign Eigenvectors as a 'np.matrix'
        print('Eigenvectors: ', self.eig_vectors_X)


        # TODO: Fourth, get covariance matrix of Y
        #
        self.Cov_Y  = self.eig_vectors_X.getT() * self.Cov_X * self.eig_vectors_X
        print('Cov_Y: ', self.Cov_Y)


        # TODO: Fifth, obtain transformed data Y
        #
        self.Y = np.dot(self.X_dm, self.eig_vectors_X)
        print('self.Y: ', self.Y)



        return self.Y, self.Cov_Y, self.eig_vectors_X, self.eig_values_X_mat, self.Cov_X, self.X_dm






    def explained_variance( self, summary = True, plot = True ):

        self.run_pca()

        # TODO: Calculate sum of Eigenvalues
        #
        eigen_values    = self.eig_values_X_mat.diagonal()
        print("Eigenvalues: ", eigen_values)

        eig_val_sum_all = np.sum(self.eig_values_X_mat)


        # TODO: Initialize arrays
        #
        var_explained     = np.empty(len(eigen_values))
        var_explained_agg = np.empty(len(eigen_values))


        # TODO: Calculate explained variance (ratio)
        #
        for i in range(0, len(eigen_values)):

            var_explained[i]     = self.Cov_Y[i,i]

            eig_val_sum          = np.sum( eigen_values[i] )
            var_explained_agg[i] = eig_val_sum / eig_val_sum_all


        if summary == True:

            print('Variance Explained:     ', np.round(var_explained, 3))

            print('Agg Variance Explained: ', np.round(var_explained_agg, 3))


        if plot == True:

            # 'Scree plot'
            #
            fig = plt.gcf()
            plt.bar(np.arange(len(var_explained_agg)), var_explained)
            plt.ylabel('Explained Variance Ratio')


        return var_explained, var_explained_agg









# Question 1d: Principal Component of Yields
# ------------------------------------------
#


# TODO: Instantiate object and run pca
#
yield_pca = PCA( yields )

yields_trans, yields_trans_cov, yields_eig_vec, yields_eig_val, yields_cov, yields_dm =  yield_pca.run_pca()    # Run PCA analysis


# Covariance matrix of transformed data
#
print(np.round(yields_trans_cov, 5))

# Eigenvalues
#
yields_eig_val.diagonal()



# Compare PCs
#
print(np.round(np.corrcoef(yields_trans, rowvar = False),2))


# TODO: Plot first principal component
#
fig = plt.gcf()
fig.set_size_inches(16, 10.5)

plt.plot(yields_trans[:, 1])
plt.legend(['PC 1'])


# TODO: Compare PC 1 with yields
#
pc1_yields                = pd.DataFrame(yields)
pc1_yields['Yields_PC1']  = yields_trans[:,0]*(-1)
pc1_yields.head()

fig = plt.gcf()
fig.set_size_inches(16, 10.5)

pc1_yields.plot()             # Plot the PC#1 along with the yields
plt.show()
pc1_yields.corr()             # Calculate the correlation between PC#1 and the yields





# Double check with 'scikit-learn' package
#

# Create PCA object (with number of components to keep)
yields_pca_sck = sck_dec.PCA(n_components = 11)

yields_pca_sck.fit( yields )

yields_trans_sck = yields_pca_sck.fit_transform(yields) # Shape: 819 x 11 ('Transformed data')



# Comparison
#
pca_comp = pd.DataFrame(yields_trans[:,0], columns = ['Yields_trans'])
pca_comp['Yields_trans_sck'] = yields_trans_sck[:, 0]
pca_comp.head()


# Correlation
pca_comp.corr()

# Plot
fig = plt.gcf()
fig.set_size_inches(16, 10.5)

pca_comp.plot()




# Recover 'original' covariance matrix
#
yields_cov_rec = yields_eig_vec * yields_eig_val * yields_eig_vec.getT()
print(np.round(yields_cov_rec, 3))

print(np.round(yields_cov, 3))




# Recover 'original' yields
#   Period 1 -> Index 0
#
print( 'Original 1m Yield : ', np.round(yields[0,0],4) )


# TODO: Invert Eigenvectors to recover X_dm
#   X_dm = Y * E^-1
#
X_dm_rec = yields_trans * yields_eig_vec.getT()
X_dm_rec


# TODO: Select 1m yield
#   Hint: You have to add the mean yield to get the original yield
#
X_rec         = X_dm_rec + yields.mean(0)
yields_1m_rec = X_rec[:,0]

print( 'Recovered 1m Yield: ', np.round(yields_1m_rec[0,0], 4) )



# Use a regression
#
X = sm.add_constant(yields_trans)

Y = yields[:,0]
Y = Y.reshape(821,1)

model  = sm.OLS(Y, X)
result = model.fit()
print(result.summary())


for i in range(0, 11):

    if i == 0:
        yields_1m_rec_reg = result.params[i]

    else:
        yields_1m_rec_reg = yields_1m_rec_reg + result.params[i]*yields_trans[0,i-1]

    print( 'Recovered 1m Yield (by Regression ', i, '): ', np.round(yields_1m_rec_reg, 4) )




# Compare regression coefficients and Eigenvectors
#
result.params[1:12]
yields_eig_vec[0,:]





print(yields_trans[0,0])
print(yields_trans[0,1])
print(yields_trans_cov[0,0])
print(yields_trans_cov[1,1])


# Check of intermediate result (2):
#
Test.assertEquals(np.round(yields_trans[0,0],3),     -0.356, 'incorrect result')
Test.assertEquals(np.round(yields_trans[0,1],3),      1.173, 'incorrect result')
Test.assertEquals(np.round(yields_trans_cov[0,0],2), 11.71,  'incorrect result')
Test.assertEquals(np.round(yields_trans_cov[1,1],3),  0.784, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(yields_trans[0,2],3),     checker_results.loc[3][0], 'incorrect result')
    Test.assertEquals(np.round(yields_trans[0,3],3),     checker_results.loc[4][0], 'incorrect result')
    Test.assertEquals(np.round(yields_trans_cov[2,2],2), checker_results.loc[5][0], 'incorrect result')
    Test.assertEquals(np.round(yields_trans_cov[3,3],3), checker_results.loc[6][0], 'incorrect result')









# Question 1e: Importance of PCs
# ------------------------------
#

yield_pca              = PCA( yields )


# TODO: Extract the importance of the PCs
#
expl_var, expl_var_agg = yield_pca.explained_variance()


print('Scikit-Learn: ', np.round(yields_pca_sck.explained_variance_ratio_, 3))




# Check of intermediate result (3):
#
Test.assertEquals(np.round(expl_var_agg[0],2), 0.93, 'incorrect result')
Test.assertEquals(np.round(expl_var_agg[1],2), 0.06, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(expl_var_agg[0],2), checker_results.loc[7][0], 'incorrect result')
    Test.assertEquals(np.round(expl_var_agg[1],2), checker_results.loc[8][0], 'incorrect result')








# Question 1f: Forecasting the Yield Curve
# ----------------------------------------


#
# Bond Portfolio Management
#
# A bond portfolio manager would be interested in the behavior of the yields over time.
#
# Time series analysis based on the 11 yields could be useful, but a better
# approach would be to use the first three principal components
#
# Next step: ARMA+GARCH Modeling of principal components
#                   -> Predictions of yield curve
#                   -> Adjust (duration) of bond portfolio
#


n = 11


# TODO: Step 1: Fit AR(1) to each PC 1 - 3 (via regression)
#

ar1_coef = np.empty(3)

for i in range(0,3):

    # PC #i
    #
    pc_i        = yields_trans[1:, i]
    pc_i_lag    = yields_trans[0:-1, i]

    Y           = pc_i
    X           = sm.add_constant(pc_i_lag)

    model       = sm.OLS(Y, X)
    result      = model.fit()

    ar1_coef[i] = result.params[1]              # beta_1 coefficient of the following regression: PC_t = beta_0 + beta_1 * PC_t-1 + eps_t

print(np.round(ar1_coef, 4))


# TODO: Step 2: Forecast PC 1 - 3
#   Note, set alpha (beta_0) to zero
#
pc1_fc = ar1_coef[0] *  yields_trans[-1, 0]                          # Take last value of pc -> index = -1
pc2_fc = ar1_coef[1] *  yields_trans[-1, 1]
pc3_fc = ar1_coef[2] *  yields_trans[-1, 2]

pc123_fc = np.array( [pc1_fc, pc2_fc, pc3_fc] )
pc123_fc

pc123_fc = pc123_fc.reshape(1,3)
pc123_fc.shape




# TODO: Step 3: Obtain yields from PC 1 - 3 forecast
#

# Invert Eigenvectors
#   X_dm = Y * E^-1


X_dm_rec    = yields_trans[0:(len(yields_trans) - 1), 0:3] * yields_eig_vec[0:3,
                                                          0:3].getI()

X_dm_fc_rec = pc123_fc * yields_eig_vec[0:3, 0:3].getI()
X_dm_fc_rec



# Add the mean (for each yield seperately)
#
yields_fc_rec = np.matrix(np.zeros((821,n)))

for i in range(n):
    yields_fc_rec[:,i] =  yields_trans[:,i] + np.mean(yields_trans[:,i])


yields_fc_rec[0,:]==yields_fc_rec[0]


# TODO: Step 4: Plot expected yield curve

# E.g. 1m forecast:
#
print( 'Original 1m Yield: ', np.round(yields[-1,0],     4) )
print( 'Forecast 1m Yield: ', np.round(yields_fc_rec[0], 4) )


# Plot yield curves
#
fig = plt.gcf()
fig.set_size_inches(16, 10.5)

plt.subplot(1,1,1)
plt.plot( time, yields[-1],                               label = dates[-1][0],            linestyle = '-',  marker = 'o' )
plt.plot( time, np.squeeze(np.array(yields_fc_rec[0])),   label = '11/01/08 (Forecast)',   linestyle = '--', marker = 'o' )
plt.legend(loc = 'lower right', frameon = True)
plt.xlabel('time')
plt.ylabel('(Expected) Yield')





# Check of intermediate result (4):
#
Test.assertEquals(np.round(ar1_coef[0],4), 0.9991, 'incorrect result')
Test.assertEquals(np.round(pc1_fc,3),      6.697, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(X_dm_fc_rec[0,0],2), checker_results.loc[9][0], 'incorrect result')
    Test.assertEquals(np.round(yields_fc_rec[0],4), checker_results.loc[10][0], 'incorrect result')




print(Test.passed, "tests passed of", Test.numTests,  "(", 100*Test.passed/Test.numTests, "%)")

