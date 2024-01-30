import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df_train = pd.read_csv("train.csv")

df_test = pd.read_csv("test.csv")
ID = df_test["id"]

df_train.drop(['id', 'city', 'followers_count', 'langs', 'bdate', 'has_photo',
          'last_seen', 'relation', 'people_main', 'occupation_name', 'life_main'], axis=1, inplace=True)

df_test.drop(['id', 'city', 'followers_count', 'langs', 'bdate', 'has_photo',
          'last_seen', 'relation', 'people_main', 'occupation_name', 'life_main'], axis=1, inplace=True)

print(df_train.occupation_type.value_counts())

def set_career(value):
    if value == "False":
        return 0
    return 1 
df_train["career_start"] = df_train["career_start"].apply(set_career)
df_train["career_end"] = df_train["career_end"].apply(set_career)
df_test["career_start"] = df_test["career_start"].apply(set_career)
df_test["career_end"] = df_test["career_end"].apply(set_career)


def set_sex(value):
    if value == 2:
        return 0
    return 1
df_train["sex"] = df_train["sex"].apply(set_sex)
df_test["sex"] = df_test["sex"].apply(set_sex)


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
df_train["education_status"] = df_train["education_status"].apply(set_education_status)
df_test["education_status"] = df_test["education_status"].apply(set_education_status)


def set_graduation(value):
    if 1940 < value < 2030:
        return value
    return 2030
df_train["graduation"] = df_train["graduation"].apply(set_graduation)
df_test["graduation"] = df_test["graduation"].apply(set_graduation)


df_train["education_form"].fillna("Full-time", inplace=True)
x = pd.get_dummies(df_train["education_form"])
df_train[list(x.columns)] = x
df_train.drop("education_form", axis=1, inplace=True)

df_test["education_form"].fillna("Full-time", inplace=True)
x = pd.get_dummies(df_test["education_form"])
df_test[list(x.columns)] = x
df_test.drop("education_form", axis=1, inplace=True)

def set_occupation_type(value):
    if value == "school":
        return 0
    if value == "university":
        return 1
    return 2

df_train["occupation_type"].fillna("school", inplace=True)
df_test["occupation_type"].fillna("school", inplace=True)
df_train["occupation_type"] = df_train["occupation_type"].apply(set_occupation_type)
df_test["occupation_type"] = df_test["occupation_type"].apply(set_occupation_type)

print(df_train.info())
print(df_test.info())


'''Модель'''


#Для теста:
# x = df_train.drop("result", axis=1)
# y = df_train["result"]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# classifier = KNeighborsClassifier(n_neighbors=3)
# classifier.fit(x_train, y_train)

# y_pred = classifier.predict(x_test)
# percent = accuracy_score(y_test, y_pred) * 100
# print(percent)


x_train = df_train.drop("result", axis=1)
y_train = df_train["result"]


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(df_test)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

result = pd.DataFrame({"id" : ID, "result" : y_pred})
result.to_csv("result", index = False)