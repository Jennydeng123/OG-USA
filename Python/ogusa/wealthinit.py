import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from scipy.stats import kde
import os


def MVKDE(S, J ,proportion_matrix = None, filename = None, plot = False,  bandwidth = .25):
	'''
    Generates a Multivariate Kernel Density Estimator and returns a matrix
    representing a probability distribution according to given age categories,
    and ability type categories.

    Inputs:
        S   	 			  = scalar, the number of age groups in the model.

        J                     = scalar, the number of ability type groups in the model.

        proportion_matrix     = [S, J], array containing the proportion matrix created
                                by SCFExtract.py. This argument would be used if you are
                                passing in the proportion_matrix directly

        filename     		  = string, the file name of the .txt document that contains 
         						the original proportion matrix created by SCFExtract.py.
                                Use this argument if you have saved the proportion matrix 
                                in a .txt file

        plot                  = boolean, whether or not you want a plot of the probability
        						distribution generated by your given age and ability type 
        						groups.
       
    Functions called: 
    	kde.gaussian_kde      = scipy function that generates a Kernel 
    							Density Estimator from given data

    Objects in function:
        proportion_matrix     = [78, 7], array containing the proportion (0 < x < 1) of 
                                the total bequests that each age-income category receives. 
                                Derived in SCFExtract.py

    	age_probs             = [78,], array containing the frequency, or how many times, 
                                that random drawn numbers fell into the 78 different age bins

    	income_probs          = [7,], array containing the frequency, or how many times, 
                                that random drawn numbers fell into the 7 different ability 
                                type bins

    	age_frequency         = [70000,], array containing repeated age values for each 
                                age number at the frequency given by the age_probs vector 

    	xmesh                 = complex number, the number of age values that will be 
                                evaluated in the Kernel Density Estimator.

    	freq_mat              = [70000, 2], array containing age_frequency and 
                                income_frequency stacked

    	density               = object, class given by scipy.stats.gaussian_kde. 
                                The Multivariate Kernel Density Estimator for the given data set.

    	age_min, age_max      = scalars, the minimum and maximum age values and minimum 
        income_min, income_max  and maximum income values 
    	

    	agei  				  = [S, J], array containing the age values to be evaluated in 
                                the Kernel Estimator (ranging from 18-90)

    	incomei 			  = [S, J], array containing the income values to be evaluated 
                                in the Kernel Estimator (ranging from 1-7)

    	coords                = [2, S*J], array containing the raveled values of agei 
                                and incomei stacked

    	estimator             = [S, J], array containing the new proportion values for 
                                s age groups and e ability type groups that are evaluated 
                                using the Multivariate Kernel Density Estimator

    	estimator_scaled       = [S, J], array containing the new proportion values for 
                                s age groups and e ability type groups that are evaluated using 
                                the Multivariate Kernel Density Estimator, but scaled so that 
                                the sum of the array is equal to one.

    Returns: estimator_scaled
    '''
        if proportion_matrix is None:
	       proportion_matrix = np.loadtxt(filename, delimiter = ',')
	proportion_matrix_income = np.sum(proportion_matrix, axis = 0)
	proportion_matrix_age = np.sum(proportion_matrix, axis = 1)
	age_probs = np.random.multinomial(70000,proportion_matrix_age)
	income_probs = np.random.multinomial(70000, proportion_matrix_income)
	age_frequency = np.array([])
	income_frequency = np.array([])
	age_mesh = complex(str(S)+'j')
	income_mesh = complex(str(J)+'j')

	j = 18
	'''creating a distribution of age values'''
	for i in age_probs:
		listit = np.ones(i)
		listit *= j
		age_frequency = np.append(age_frequency, listit)
		j+=1

	k = 1
	'''creating a distribution of ability type values'''
	for i in income_probs:
		listit2 = np.ones(i)
		listit2 *= k
		income_frequency = np.append(income_frequency, listit2)
		k+=1

	freq_mat = np.vstack((age_frequency, income_frequency)).T
	density = kde.gaussian_kde(freq_mat.T, bw_method=bandwidth)
	age_min, income_min = freq_mat.min(axis=0)
	age_max, income_max = freq_mat.max(axis=0)
	agei, incomei = np.mgrid[age_min:age_max:age_mesh, income_min:income_max:income_mesh]
	coords = np.vstack([item.ravel() for item in [agei, incomei]])
	estimator = density(coords).reshape(agei.shape)
	estimator_scaled = estimator/float(np.sum(estimator))
	if plot == True:
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.plot_surface(agei,incomei, estimator_scaled, rstride=5)
		ax.set_xlabel("Age")
		ax.set_ylabel("Ability Types")
		ax.set_zlabel("Received proportion of total bequests")
		plt.show()
	return estimator_scaled

def wealth_dist(year_start, year_end, S, J, path):
    '''
    FROM THE SURVEY OF CONSUMER FINANCES
    Variables used:
    				

    X8022           FOR THE RESPONDENT, THIS VARIABLE CONTAINS THE DATE-OF-BIRTH (age)

    Net Worth		Already adjusted for inflation, respondent's perception of own
    				net worth, summary variable

    wgt             Summary variable, weights provided by the Federal Reserve 
                    to adjust for selection bias in the survey.

    networth / Ability Type categories:

    1.  Under $15000			
    2.  15,000-24,999			
    3.  25,000-49,999		
    4.  50,000-74,999			
    5.  75,000-99,999	
    6.  100,000-249,999	
    7.  250,000 or more	

    Age categories:

    (min age in SCF generally is 18)

    ages 18-96	

    (max in 2013 was 95 yrs old)

    Inputs:
    year_start   	 	  = scalar, the first year of the range of data years 
    						that you want from the survey.

    year_end              = scalar, the end year of the range of data years 
    						that you want from the survey	
    
    S   	 			  = scalar, the number of age groups in the model.

    J                     = scalar, the number of ability type groups in the model.	

    path                  = string, the location of your SCF data files. Instructions on 
                            how to download these files is located in HowToFindData.txt
                            You can use dir = os.path.dirname(__file__) + '/' which gives you
                            the path to the directory in which you currently are.

    Functions called: 

    initSCFvars	      	  = creates a dataframe with the relevant age, networth, and
    						weights variables

    genNetworthMatrix     = creates a matrix full of the networth values for each
    						age and income group

    MVKDE                 = generates a Multivariate Kernel Density Estimator according
                            to the data given
        
    Returns: kernel
    '''

    '''Loading in data sets into different panda dataframes'''


    filenames = ['p13i6','p10i6','p07i6','p04i6','p01i6','p98i6',
    'p95i6','p92i4','p89i6']

    summaryfiles = ['rscfp2013','rscfp2010','rscfp2007','rscfp2004','rscfp2001',
    'rscfp1998','rscfp1995','rscfp1992','rscfp1989']



    '''Making sure feasible inputs are given'''
    if (year_start > 2013) or (year_start<1989) or (year_start % 3 != 0) :
        raise ValueError("SCF data non-existent for start year provided")
    if (year_end > 2013) or (year_end<1989) or (year_end % 3 != 0):
        raise ValueError("SCF data non-existent for end year provided")
    if (S < 0) or (J<0) :
        raise ValueError("Values must be positive for age and ability types")

    '''initializing the year range to be used in SCF data extraction'''
    year_range = np.arange(year_end, year_start-3, -1)
    year_list = []
    for year in year_range:
        if year % 3 == 0:
            year_list.append(year)
        else:
            pass

    '''intializing dataframe lists with SCF and SCF summary variable filenames''' 
    df_list = []
    for i in xrange(len(year_list)):
        df_list.append(pd.read_stata(path +filenames[i]+'.dta'))
    dfs_list = []
    for i in xrange(len(year_list)):
        dfs_list.append(pd.read_stata(path+summaryfiles[i]+'.dta'))

    '''set different income levels for the different ability types'''
    income_levels = [-999999999, 15000,25000,50000,75000,100000,250000, 999999999]

    '''variable for the y axis income groups for graph'''
    income_y = np.arange(1,8)

    '''the minimum age in the survey'''
    min_age = 18

    '''the maximum age that we want to represent in the data'''
    max_age = 95

    '''creating an array full of the different ages'''
    age_groups = np.arange(min_age, max_age+1)
    income_groups = 7


    def initSCFvars(scf, scfSummary, year, year_range):
    	'''
        Generates variables from the SCF that will eventually be used
        to calculate total bequests from all respondents, total 
        bequests for age-income categories, and a proportion matrix.

        Inputs:
            scf   	 			  = dictionary with all SCF data, indexed with codebook
            		   			    variables.
            scfSummary            = dictionary with all SCF summary variables adjusted for inflation.

            year     			  = scalar, what year the SCF survey was taken

            year_range            = list, years to be adjusted for inflation
           
        Functions called: None

        Objects in function:
            age                  = panda dataframe, contains the ages of the respondents

            age_income_matrix     = [X = # of income groups, Y = # of age groups], 
            						Nested lists, this variable contains a list of Y lists
            						corresponding to each age group, and each of the Y
            						lists contains X amounts of income groups. These nested lists
            						will be filled with pandas and will be used to extract inheritance
            						data corresponding to the age and income conditions.

            income           	  = panda dataframe, contains total household income for each respondent


            age_income 			  = panda dataframe, combines the age, income, value_of_transfer,
            						and year_received pandas into one panda.


            total_networth  	  = scalar, total networth of all respondents

            weights               = dataframe, contains the weights corresponding to each respondent,
                                    these correct for selection bias in survey.


        Returns: age_income, total_bequests_received, age_income_matrix, total_bequest_matrix
        '''
        age = scf['X8022']
        weights = scfSummary['wgt']
        income = scfSummary['networth']* (weights/5.)
        age_income = pd.concat([age, income], axis =1)
        age_income.columns = ['age' , 'income']
        total_networth = income.sum()
        return (age_income, total_networth)


    def genNetworthMatrix(age_income, age_groups, 
    	income_groups, min_age, income_levels, year):
    	'''
        Generates a matrix for the total bequests of each age-income category, 
        after generating a Nested list containing the pandas corresponding to the age, 
        income and recent wealth transfer conditions.

        Inputs:

            age_groups            = [S, ], vector of integers corresponding to different ages
            						of respondents.

            income_groups         = scalar, total number of income groups/ ability types.

            min_age               = scalar, minimum age of respondents

            income_levels         = [J, ], vector of scalars, corresponding to different
            						income cut-offs.

            age_income            = panda dataframe, age, and income columns

            year                  = scalar, what year the SCF survey was taken

           
        Functions called: None

        Objects in function:
            age_income_matrix     = [J = # of income groups, S = # of age groups], 
            						Nested lists, this variable contains a list of Y lists
            						corresponding to each age group, and each of the Y
            						lists contains X amounts of income groups. Filled with pandas 
            						corresponding to the age, income and recent wealth transfer conditions.

            total_networth_matrix = [S = # of age groups, J = # of income_groups], array filled with 
            						the average networth of each age-income category

        Returns: total_bequest_matrix
        '''
        age_income_matrix = [[[] for z in xrange(income_groups)] for x in xrange(len(age_groups))]
        total_networth_mat = np.zeros((len(age_groups), income_groups))
        for i in age_groups:
		  for j in xrange(income_groups):
    		    age_income_matrix[i-min_age][j] = age_income[(age_income['age']==i) & (age_income['income'] < income_levels[j+1]) & (age_income['income'] >= income_levels[j])]
    		    total_networth_mat[(i-min_age),j] = (age_income_matrix[i-min_age][j]['income'].sum())/(age_income_matrix[i-min_age][j]['income']).count()
    	return total_networth_mat

    '''initialize lists to contain the matrices produced for the different years'''
    proportion_matrix_list = []
    total_networth_list = []
    total_networth_matrix_list = []
    counter = 0

    '''creating the different matrices for the different years and inserting them into their respective lists'''
    for year in (year_list):
        age_income, total_networth = initSCFvars(df_list[counter], dfs_list[counter], year, year_range)
        total_networth_mat = genNetworthMatrix(age_income, \
    		age_groups, income_groups, min_age, income_levels, year)
        total_networth_list.append(total_networth)
        total_networth_matrix_list.append(total_networth_mat)
        counter += 1

    '''creating the matrix of all years combined''' 
    summed_total_networth = np.zeros((len(age_groups), income_groups))
    for i in xrange(len(year_list)):
        summed_total_networth += total_networth_matrix_list[i]
    all_years_networth = (summed_total_networth/float(len(year_list)))
    all_years_proportion = all_years_networth / all_years_networth.sum()
    '''fitting a kernel to the data, then scaling it into our initial wealth matrix'''
    kernel = MVKDE(S, J, all_years_proportion, None, False)
    kernel = kernel * all_years_networth.sum()
    return kernel

        
def Kbar_Eq(scale, Kbar, omega0, lambda0, wealth_dist):
	'''
	Used in the fsolve as our equation for aggregate capital 
	stock.

	Inputs:

		scale       =  scalar, scaling parameter we are estimating
					  that will be used to scale data to the model's
					  aggregate capital stock

		Kbar        =  scalar, aggregate initial capital stock

		omega0      =  [S,] array, containing our population dynamics

		lambda0     =  [J,] array, conatining our income dynamics

		wealth_dist =  [J, S] array, contains our SCF distribution of 
						wealth among different age and networth categories
						calculated using our SCF_wealth_dist function

	Objects in Function:

		k_calc      =  scalar, the sum of each age-income category's 
					   average networth multiplied by their population 
					   and income dynamics
	Returns:

		Kbar - k_calc = should be equal to zero since Kbar = k_calc
	'''
	S = len(omega0)
	J = len(lambda0)
	k_calc = 0
	for s in xrange(S):
		for j in xrange(J):
			k_calc+= omega0[s]*lambda0[j]* scale*wealth_dist[j,s]
	return Kbar-k_calc

def init_wealth(Kbar, omega0, lambda0, wealth_dist, scale_tol):
	'''
	Used in the fsolve as our equation for aggregate capital 
	stock.

	Inputs:

		Kbar        =  scalar, aggregate initial capital stock

		omega0      =  [S,] array, containing our population dynamics

		lambda0     =  [J,] array, conatining our income dynamics

		wealth_dist =  [J, S] array, contains our SCF distribution of 
						wealth among different age and networth categories
						calculated using our SCF_wealth_dist function

		scale_tol   =  scalar, the tolerance to be used in the fsolve

	Objects in Function:

		scale       =  scalar, the factor that we multipliy our SCF
					   wealth distribution by to get the initial
					   wealth distribution in our model
	Returns:

		wealth_init = [J,S] array, our initial distribution of wealth_dist
					  to be used in our model
		scale       =  scalar, the factor that we multipliy our SCF
					   wealth distribution by to get the initial
					   wealth distribution in our model

	'''
	S = len(omega0)
	J = len(lambda0)
	scale_guess = 0.5
	scale = opt.fsolve(Kbar_Eq, scale_guess, args=(Kbar, omega0, lambda0, wealth_dist),
                      xtol=scale_tol)
	wealth_init = scale * wealth_dist
	k_calc = 0
	for s in xrange(S):
		for j in xrange(J):
			k_calc+= omega0[s]*lambda0[j]* scale*wealth_dist[j,s]
	return wealth_init, scale


def age_income_plot(networth_range, age_range, proportion_matrix, year):
    X,Y = np.meshgrid(networth_range, age_range)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y, proportion_matrix, rstride=5)
    ax.set_xlabel("Ability Types")
    ax.set_ylabel("Age")
    ax.set_zlabel("Total networth for years {}".format(str(year)))
    plt.show()

