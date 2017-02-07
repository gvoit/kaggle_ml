import pandas as pd
from sklearn.preprocessing import LabelEncoder

class TitanicFeaturesCleaner(object):
	"""
		Initial DataFrame Columns:
		PassengerId    int64
		Survived       int64
		Pclass         int64
		Name           object
		Sex            object
		Age            float64
		SibSp          int64
		Parch          int64
		Ticket         object
		Fare           float64
		Cabin          object
		Embarked       object

	"""

	"""TitanicFeaturesCleaner tranform, encods and drops columns """
	def __init__(self, features):
		self._features = features.copy()

	def run_all_cleaners(self):
		self.cleanup_names()
		self.cleanup_parch_sibsp()
		self.cleanup_tickets()
		self.cleanup_ages()
		self.cleanup_sex()
		self.cleanup_embarked()
		self.cleanup_fare()
		
		self.drop_not_used()

	def cleanup_fare(self):
		self._features["Fare"].fillna(self._features["Fare"].mean(), inplace=True)

	def cleanup_embarked(self):
		self._features["Embarked"].fillna("C", inplace=True)
		self._features.loc[self._features["Embarked"] == "C", "Embarked"] = 0
		self._features.loc[self._features["Embarked"] == "Q", "Embarked"] = 1
		self._features.loc[self._features["Embarked"] == "S", "Embarked"] = 2
	
		
	def cleanup_sex(self):
		self._features.loc[self._features["Sex"] == "male", "Sex"] = 0
		self._features.loc[self._features["Sex"] == "female", "Sex"] = 1

	def cleanup_tickets(self):
		self._features["Ticket"] = LabelEncoder().fit_transform(self._features["Ticket"])

	def cleanup_ages(self):
		self._features["Age"].fillna(self._features["Age"].mean(), inplace=True)

	def cleanup_names(self):
		p = '(.*),\s+(.*)\.'
		names_cleaned = self._features["Name"].str.extract(p, expand=True)
		self._features["LName"] = LabelEncoder().fit_transform(names_cleaned[0])
		self._features["Title"] = LabelEncoder().fit_transform(names_cleaned[1])
		self._features.drop("Name", axis=1, inplace=True)


	def cleanup_parch_sibsp(self):
		self._features["Reltvs"] = self._features["Parch"] + self._features["SibSp"]
		self._features.drop("Parch", axis=1, inplace=True)
		self._features.drop("SibSp", axis=1, inplace=True)


	def drop_not_used(self):
		"""Remove extra features"""
		self._features.drop("PassengerId", axis=1, inplace=True)
		self._features.drop("Cabin", axis=1, inplace=True)
		# self._features.dropna(axis=1,inplace=True)
		# self._features.drop("Embarked", axis=1, inplace=True)
		####################################################


	def drop_NaNs(self):
		self._features.dropna(axis=1,inplace=True)
		


	def get_features(self):
		return self._features

