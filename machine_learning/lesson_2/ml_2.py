import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("train.csv")
print(df.info())

df.drop(['id', 'bdate', 'has_photo', 'langs', 'city', 'followers_count',
          'last_seen', 'relation', 'people_main', 'occupation_name', 'life_main'], axis=1, inplace=True)

def set_career(value):
    if value == "False":
        return 0
    return 1 
df["career_start"] = df["career_start"].apply(set_career)
df["career_end"] = df["career_end"].apply(set_career)

def set_sex(value):
    if value == 2:
        return 0
    return 1
df["sex"] = df["sex"].apply(set_sex)

def set_education_status(value):
    if value == "Undergraduate applicant":
        return 1
    if value == "Student (Bachelor's)":
        return 2
    if value == "Alumnus (Bachelor's)":
        return 3
    if value == "Student (Master's)":
        return 4
    if value == "Alumnus (Master's)":
        return 5
    if value == "Candidate of Sciences":
        return 6
    if value == "PhD":
        return 7
    return 0
df["education_status"] = df["education_status"].apply(set_education_status)

def set_graduation(value):
    if 1940 < value < 2030:
        return value
    return 2030
df["graduation"] = df["graduation"].apply(set_graduation)

# def set_life_main(value):
#     if value in ["0", "6", "7"]:
#         return "suitable"
#     if value in ["2", "3", "4", "8"]:
#         return "unsuitable"
#     return "same"
    
# df["life_main"] = df["life_main"].apply(set_life_main)

# temp = df.groupby(by="life_main")["result"].value_counts()
# print(temp)

# x = pd.get_dummies(df["life_main"])
# df[list(x.columns)] = x
# df.drop("life_main", axis=1, inplace=True)

df["education_form"].fillna("Full-time", inplace=True)
x = pd.get_dummies(df["education_form"])
df[list(x.columns)] = x
df.drop("education_form", axis=1, inplace=True)

df["occupation_type"].fillna("school", inplace=True)
x = pd.get_dummies(df["occupation_type"])
df[list(x.columns)] = x
df.drop("occupation_type", axis=1, inplace=True)

print(df.info())

x = df.drop("result", axis=1)
y = df["result"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test - sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
percent = accuracy_score(y_test, y_pred) * 100
print(percent)