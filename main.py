import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron

data = pd.read_csv("titanic.csv")

data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis="columns")
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].median())
data["Embarked"] = data["Embarked"].fillna("S") 

encoder = LabelEncoder()
data["Sex"] = encoder.fit_transform(data["Sex"])
data["Embarked"] = encoder.fit_transform(data["Embarked"])

#For Embarked, 0=C(Cherbourg), 1=Q(Queenstown), 2=S(Southampton)
#For Sex, 0=Female, 1=Male 

min_fare = min(data["Fare"])
max_fare = max(data["Fare"])

X = data.drop(["Survived"], axis=1).values
y = data["Survived"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Welcome to the Titanic Survival Predictor. This app lets you choose between 7 different machine learning models and then the values of 7 features for a Titanic passenger and predicts the survival of that passenger.")
print()
print("The machine learning models you can choose from are: ")
print("1. Decision Tree Classifier")
print("2. Stochastic Gradient Descent Classifier")
print("3. Random Forest Classifier")
print("4. Logistic Regression")
print("5. K Nearest Neighbors Classifier")
print("6. Gaussian Naive Bayes")
print("7. Perceptron")
print()

while True:
    model = input("Please enter the number corresponding to the machine learning model you would like to use: ")
    if model == "1":
        model = tree.DecisionTreeClassifier()
        break
    elif model == "2":
        model = SGDClassifier()
        break
    elif model == "3":
        model = RandomForestClassifier()
        break
    elif model == "4":
        model = LogisticRegression(max_iter=1000)
        break
    elif model == "5":
        model = KNeighborsClassifier()
        break
    elif model == "6":
        model = GaussianNB()
        break
    elif model == "7":
        model = Perceptron(max_iter=100)
        break
    else:
        print("Invalid input! You must choose a number from 1 to 7.")
    

model.fit(X_train, y_train) 
accuracy = model.score(X_test, y_test)

print()
print("The 7 features you will be choosing are as follows:")
print("1. The class in which the passenger was traveling (1, 2, or 3)")
print("2. The sex of the passenger (male or female)")
print("3. The age of the passenger")
print("4. The number of siblings and spouses the person was traveling with")
print("5. The number of parents and children the person was traveling with")
print("6. The amount the person paid for his or her fare.")
print("7. The port from which the passenger Embarked (Cherbourg, Queenstown, or Southampton)")
print()
print("Ok, let's get started!")
print()

while True:
    try:
        pclass = int(input("In what class did your passenger travel? "))
        if pclass == 1 or pclass == 2 or pclass == 3:
            break
        else:
            print()
            print("Invalid input! You must enter 1, 2, or 3.")
            print()
    except ValueError:
        print()
        print("Invalid input! You must enter 1, 2, or 3.")
        print()
        
print()

while True:
    sex = input("Was your passenger male or female? Choose m or f: ").lower()
    if sex == "m":
        sex = 1
        break
    elif sex == "f":
        sex = 0
        break
    else:
        print()
        print("Invalid input! You must enter either 'm' or 'f'.")
        print()

print()

while True:
    try:
        age = int(input("How old was your passenger? "))
        if age <= 0:
            print()
            print("Invalid input! The age of your passenger must be an integer greater than zero.")
            print()
        else:
            break
    except ValueError:
        print()
        print("Invalid input! You must enter an integer.")
        print()

print()

while True:
    try:
        sibsp = int(input("How many siblings and spouses did your passenger travel with, including his or her own spouse and those of his or her siblings? "))
        if sibsp < 0:
            print()
            print("Invalid input! Your number must be greater than or equal to zero.")
            print()
        else:
            break
    except ValueError:
        print()
        print("Invalid input! You must enter an integer greater than or equal to zero.")
        print()

print()

while True:
    try:
        parch = int(input("How many parents did your passenger travel with? "))
        if parch < 0:
            print()
            print("Invalid input! Your number must be greater than or equal to zero.")
            print()
        else:
            break
    except ValueError:
        print()
        print("Invalid input! You must enter an integer greater than or equal to zero.")
        print()
        
print()

while True:
    try:
        fare = float(input(f"How much did your passenger pay for his or her ticket? Keep in mind that the range of prices for a ticket on the Titanic was £{min_fare} to £{max_fare}. "))
        if fare < 0:
            print()
            print("Invalid input! Your number must be equal to or greater than zero.")
            print()
        else:
            break
    except ValueError:
        print()
        print("Invalid input! You must enter either a positive integer or a positive decimal number.")
        print()
        
print()

while True:
    embarked = input("From which port did your passenger embark (Cherbourg, Queenstown, or Southampton; choose C, Q, or S? ").lower()
    if embarked == "c":
        embarked = 0
        break
    elif embarked == "q":
        embarked = 1
        break
    elif embarked == "s":
        embarked = 2
        break
    else:
        print()
        print("Invalid input! Choose C, Q, or S.")
        print()

prediction = model.predict([[pclass, sex, age, sibsp, parch, fare, embarked]])
accuracy = model.score(X, y)

print()
if prediction == [0]:
    print(f"I'm sorry, the model you chose predicted, with {accuracy:.1%} accuracy, that your passenger did not survive.")
else:
    print(f"Congratulations! The model you chose predicted, with {accuracy:.1%} accuracy, that your passenger survived!")






 