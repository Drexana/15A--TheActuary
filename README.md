# [Health Insurance Claims](https://the-actuary-health-insurance-claims.streamlit.app/)
Developed an AI model that aids insurance agencies and brokers, during the underwriting process by classifying claimants into three risk-categories: Preferred, Standard, and High-Cost.  By employing a multitude of python techniques and data analysis methodologies all within AI4ALL's cutting-edge AI4ALL Ignite accelerator.

# Problem Statement
The underwriting process is a time-consuming process that can take a broker up to 6 to 8 weeks to complete. An inaccurate risk-classification can lead to inaccurate coverage and inconsistencies with premiums. The inability to properly assign or identify major classifications like Substandard or high-cost claimants can cost both the insurance agency and client money. With the help of The Actuary (Our AI model), we would also aid brokers greatly to identify outliers to further review the clients profile to justify their risk-classifications.

# Key Results
1. Sorted over 1330 claimaints into three risk classifications.
2. Accurately classified claimants with a 98.8% accuracy rate.
3. Identified only 5% of claimants as outliers within their sorted group.

# Methodologies
We achieved these results through the utilization of K-Means Clustering, Isolation Forest, and Random Forest. K-Means clustering was used to categorize our data. From there we analyzed how the data clustered and assigned risk-categories(preferred, standard, and high-cost). We then used an Isolation Forest to determine the anomalies(outliers) within the clusters and highlight them. Afterwards, a Random Forest was utilized during Training and Testing to help the model decide a clients classification based on their provided information. 

# Data Sources
* [Forbes Fraud Statistics  ](https://www.forbes.com/advisor/insurance/fraud-statistics/)  	
* [Healthcare Expenditures ](https://meps.ahrq.gov/data_files/publications/st533/stat533.shtml)   	
* [Linear Regression ](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/)  	
* [Insurance Risk Class](https://www.investopedia.com/terms/i/insurance-risk-class.asp)  	
* [AI in Healthcare Bias & Mitigations](https://www.nature.com/articles/s41746-023-00858-z)   	
* [Exploiting Machine Learning Bias to predict medical denials  ](https://ojs.aaai.org/index.php/AAAI-SS/article/download/31181/33341/35237)	
* [Medical Underwriting  ](https://www.investopedia.com/terms/m/medical-underwriting.asp)	
* [Random Forest  ](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/)	
* [Feature Importance  ](https://www.geeksforgeeks.org/machine-learning/understanding-feature-importance-and-visualization-of-tree-models/)	
* [Isolation Forest ](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)	
* [Kaggle Dataset ](https://www.kaggle.com/code/yash9439/health-insurance-claims-eda/notebook)	

# Technologies Used
* Kagglehub  
* Python  
* Pandas  
* Seaborn  
* Sklearn  
  * sklearn.preprocessing    
  * StandardScaler   
  * sklearn.cluster  
    * KMeans  
  * sklearn.metrics 
    * r2_score, silhouette_score, roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay   
  * sklearn.ensemble     
    * RandomForestClassifier, RandomForestRegressor, IsolationForest    
  * sklearn.model_selection  
    * train_test_split     
  * sklearn - tree  
    * export_graphviz
* Matplotlib   

# Project Repository
[Github Repository](https://github.com/Drexana/15A---Health-Insurance-Claims)

# Authors
Project Contributors:  
* Sebastian Davalos (sebas06lex@gmail.com) | [Github  ](https://github.com/chumboooo)		
* Drexana Rolle (drex.rolle909@gmail.com) | [Github](https://github.com/Drexana)		
