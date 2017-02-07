import argparse, os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

sys.path.append("../tools")
from ml.tools.common import printProgressBar


from preprocess_features import TitanicFeaturesCleaner
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def load_data(train_file, test_file):
	train = pd.read_csv(train_file)
	test = pd.read_csv(test_file)

	return train, test


def cleanup_features(features):
	fc = TitanicFeaturesCleaner(features)
	fc.run_all_cleaners()
	return fc.get_features()

def export_predictions(features_test, predictions, output_file):
	submission = pd.DataFrame({
    	"PassengerId": features_test["PassengerId"],
    	"Survived": predictions
    })

	submission.to_csv(output_file, index=False)

def drawScatter(features, x_label,y_label, colors):
	x = features[x_label]
	y = features[y_label]
	plt.scatter(x, y, c=colors, alpha=0.9)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()
	exit()



def main():
	#Change work dir to root of project
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	os.chdir("../")

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_file')
	parser.add_argument('--test_file')
	parser.add_argument('--output_file')
	args = parser.parse_args()


	train, test = load_data(args.train_file, args.test_file)
	test_copy = test.copy()

	main_predictors = [
		"Pclass", 
		"Sex", 
		"Age",  
		"Fare",
		"Ticket",
		"Embarked",
		"LName",
		"Title",
		"Reltvs"
	]


	labels_train = train["Survived"]
	train.drop("Survived", axis=1, inplace=True)
	train = cleanup_features(train)[main_predictors]
	test = cleanup_features(test)[main_predictors]

	###############################################
	from sklearn.feature_selection import SelectKBest
	kb = SelectKBest(k=8)
	train = kb.fit_transform(train, labels_train)
	test = kb.transform(test)
	# x_label = "Age"
	# y_label = "Reltvs"
	# drawScatter(train,x_label, y_label, labels_train)


 	###############################################
	from sklearn.model_selection import KFold
	from sklearn.model_selection import cross_val_score
	kf = KFold(n_splits=10, random_state=22)

	ensambles_list = []

	mlpc = MLPClassifier()
	lrc = LogisticRegression(C=13.)
	svcl = SVC(C=13., gamma=10)
	dtc = DecisionTreeClassifier(
		min_samples_split=10,
	)
	rfc = RandomForestClassifier(
			min_samples_split=5, 
			n_estimators=50,
			random_state=22
	)
	gnb = GaussianNB()


	ensambles_list.append(('lr', lrc))
	ensambles_list.append(('svc', svcl))
	ensambles_list.append(('dtc', dtc))
	# ensambles_list.append(('rf', rfc))
	# ensambles_list.append(('gnb', gnb))

	############ Chose final algorithm ########
	alg = VotingClassifier(estimators=ensambles_list, voting='hard')
	alg = rfc


	###########################################
	accuracies = cross_val_score(alg, train, labels_train)


	print "\n_____________________"	
	print "Max accuracy: %0.3f" % max(accuracies)
	print "Mean accuracy: %0.3f" % (sum(accuracies)/len(accuracies))
	print "Min accuracy: %0.3f" % min(accuracies)

	# names = train.columns.values

	# pd.Series(importances*100, index=names).plot(kind="bar")
	# plt.show()

	#Show all plots
	# plt.show()


	############### Predict and export #########################

	alg.fit(train, labels_train)
	predict_labels = alg.predict(test)	
	export_predictions(test_copy, predict_labels, args.output_file)
	############################################################


if __name__ == "__main__":
	main()