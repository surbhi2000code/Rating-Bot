import pandas
from sklearn.feature_extraction.text import DecisionTreeClassifier, CountVectorizer, TfidfTransformer
dt = pandas.read_csv('reviews.csv')
x = dt['Reviews']
vec = CountVectorizer()
vec.fit(x)
vec_X = vec.transform(x)
tfidf = TfidfTransformer()
tfidf.fit(vec_X)
rev = tfidf.transform(vec_X)
y = dt['Rating'].tolist()
Model = DecisionTreeClassifier()
Model.fit(rev, y)
txt = ["The product is not in good condition",]
txt_ex = vec.transform(txt)
txt_tf = tfidf.transform(txt_ex)
Model.predict(txt_tf)
def rate(*comment):
    f_ex = vec.transform(comment)
    tf = tfidf.transform(f_ex)
    pred = Model.predict(tf)
    for rev,ret in zip(comment,pred):
        print(rev,':\n','Rating:',ret)
rate('Not in good condition','It is satisfactory')