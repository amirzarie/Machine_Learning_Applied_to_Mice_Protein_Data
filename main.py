import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.patches as mpatches
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
sns.set()

column_names = "DYRK1A_N	ITSN1_N	BDNF_N	NR1_N	NR2A_N	pAKT_N	pBRAF_N	pCAMKII_N	pCREB_N	pELK_N	pERK_N	pJNK_N	PKCA_N	pMEK_N	pNR1_N	pNR2A_N	pNR2B_N	pPKCAB_N	pRSK_N	AKT_N	BRAF_N	CAMKII_N	CREB_N	ELK_N	ERK_N	GSK3B_N	JNK_N	MEK_N	TRKA_N	RSK_N	APP_N	Bcatenin_N	SOD1_N	MTOR_N	P38_N	pMTOR_N	DSCR1_N	AMPKA_N	NR2B_N	pNUMB_N	RAPTOR_N	TIAM1_N	pP70S6_N	NUMB_N	P70S6_N	pGSK3B_N	pPKCG_N	CDK5_N	S6_N	ADARB1_N	AcetylH3K9_N	RRP1_N	BAX_N	ARC_N	ERBB4_N	nNOS_N	Tau_N	GFAP_N	GluR3_N	GluR4_N	IL1B_N	P3525_N	pCASP9_N	PSD95_N	SNCA_N	Ubiquitin_N	pGSK3B_Tyr216_N	SHH_N	BAD_N	BCL2_N	pS6_N	pCFOS_N	SYP_N	H3AcK18_N	EGR1_N	H3MeK4_N	CaNA_N"
column_names = column_names.split()
mice = fetch_openml(name='miceprotein', version=4)

parsing_targets = [0, 150, 300, 435, 570, 705, 840, 945, len(mice.target)]
parsing_groups = ['c-CS-m', 'c-SC-m', 'c-CS-s', 'c-SC-s', 't-CS-m', 't-SC-m', 't-CS-s', 't-SC-s']

def Analysis_Choice(parsing_targets, parsing_groups):
    group_name = input('Enter group code: ')
    i = parsing_groups.index(group_name)
    return i


def Combined_Data(parsing_targets, parsing_groups, mice, column_names):
    i = Analysis_Choice(parsing_targets, parsing_groups)
    ii = Analysis_Choice(parsing_targets, parsing_groups)
    target_i = pd.Series(mice.target[parsing_targets[i]:parsing_targets[i+1]])
    target_ii = pd.Series(mice.target[parsing_targets[ii]:parsing_targets[ii+1]])
    data_i = pd.DataFrame(mice.data[parsing_targets[i]:parsing_targets[i+1]], columns=column_names)
    data_ii = pd.DataFrame(mice.data[parsing_targets[ii]:parsing_targets[ii+1]], columns=column_names)
    target = pd.concat([target_i, target_ii], ignore_index=True)
    data = pd.concat([data_i, data_ii], ignore_index=True)
    return data, target, i, ii


def replace_nan_with_mean(data, parsing_targets, i, ii):
    for a in column_names:
        for aa in range(parsing_targets[i+1] - parsing_targets[i]):
            if np.isnan(data[a][aa]):
                data[a][aa] = np.nanmean(data[a][0:(parsing_targets[i+1] - parsing_targets[i])])
        for aa in range((parsing_targets[i+1] - parsing_targets[i]),len(data)):
            if np.isnan(data[a][aa]):
                data[a][aa] = np.nanmean(data[a][(parsing_targets[i+1] - parsing_targets[i]):])
    return data


def PCA_code(data):
    PCA_model = PCA(n_components=2)
    PCA_model.fit(data)
    PCA_data = pd.DataFrame(PCA_model.transform(data), columns=['PCA_1', 'PCA_2'])
    return PCA_data, PCA_model


def GaussianNA_Modeling_of_PCAs(PCA_data, target):
    GNB_model = GaussianNB()
    accuracy_list = []
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(PCA_data, target, test_size=0.10)
        GNB_model.fit(X_train, y_train)
        y_pred = GNB_model.predict(X_test)
        accuracy_list.append(np.round(accuracy_score(y_test, y_pred), decimals=2))
    mean_accuracy = np.round(np.mean(accuracy_list), decimals=2)
    return mean_accuracy, GNB_model


def Plot_Gaussian_Result(PCA_data, target, mean_accuracy, GNB_model, parsing_targets, parsing_groups, i, ii):
    x_new = pd.DataFrame([-5, -4] + [11, 8] * np.random.rand(2000, 2), columns=['test_1', 'test_2'])
    y_new = GNB_model.predict(x_new)

    PCA_data['target'] = target
    x_new['target'] = y_new

    PCA_color = (parsing_targets[i+1]-parsing_targets[i])*['light red'] + (parsing_targets[ii+1] - parsing_targets[ii])*['charcoal']
    x_color = []
    for a in range(2000):
        if y_new[a] == parsing_groups[i]:
            x_color.append('light red')
        elif y_new[a] == parsing_groups[ii]:
            x_color.append("charcoal")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set(title="Gaussian Naive Baye's Classifier over Principal Components of Mice Protein Data")
    ax.set(xlabel='PCA 1', ylabel='PCA 2')
    ax.scatter(PCA_data['PCA_1'], PCA_data['PCA_2'], c=sns.xkcd_palette(PCA_color))
    ax.scatter(x_new['test_1'], x_new['test_2'], c=sns.xkcd_palette(x_color), alpha=0.1)
    ax.text(-2, -2.5, 'Mean accuracy='+str(mean_accuracy), ha='center')
    amber_patch = mpatches.Patch(color=sns.xkcd_rgb["light red"], label=target.iloc[0])
    blue_patch = mpatches.Patch(color=sns.xkcd_rgb["charcoal"], label=target.iloc[-1])
    plt.legend(handles=[amber_patch, blue_patch], loc='lower right')
    plt.show()


def support_vector_classifier_validation(PCA_data, target):
    C_range = np.arange(0.01, 5, 0.1)
    train_score, val_score = validation_curve(SVC(kernel='rbf', gamma='auto'), PCA_data, target, param_name='C',
                                              param_range=C_range, cv=5)

    plt.figure(figsize=(9, 6))
    plt.plot(C_range, np.median(train_score, 1), color='b', label='Training score')
    plt.plot(C_range, np.median(val_score, 1), color='r', label='Validation score')
    plt.title('Training and Validation Scores of SVC with Different C Hyperparameter Values for c-CS-s & c-SC-s Groups')
    plt.xlabel('Range of C (softening) hyperparameter values for SVC')
    plt.ylabel('Accuracy score')
    plt.legend(loc='lower right')
    plt.show()

    print(f"The optimal C parameter based on validation score is: {np.round(np.amax(np.median(val_score, 1)), 3)*100}%")
    return


data, target, i, ii = Combined_Data(parsing_targets, parsing_groups, mice, column_names)
data = replace_nan_with_mean(data, parsing_targets, i, ii)
PCA_data, PCA_model = PCA_code(data)
mean_accuracy, GNB_model = GaussianNA_Modeling_of_PCAs(PCA_data, target)
support_vector_classifier_validation(PCA_data, target)
Plot_Gaussian_Result(PCA_data, target, mean_accuracy, GNB_model, parsing_targets, parsing_groups, i, ii)
loading_scores = pd.DataFrame(PCA_model.components_.T, index=column_names, columns=['PCA_1', 'PCA_2'])
print(loading_scores.sort_values('PCA_1', ascending=False)[:5])
print(loading_scores.sort_values('PCA_2', ascending=False)[:5])