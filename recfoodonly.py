import pyspark
from pyspark.sql.functions import explode
from flask import Flask, request, render_template
from pyspark.ml.recommendation import ALS
import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline as PL
import findspark

findspark.init()

app = Flask(__name__)

foods_names = pd.read_csv('foods_names.csv')
food_list = foods_names[::40].food_name.to_list()

@app.route('/')
def home():
    return render_template('home.html', list=food_list)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            spark = pyspark.sql.SparkSession.builder.getOrCreate()
            df = spark.read.csv('rec_data_spark.csv', header=True, inferSchema=False)
            print(df)

            columnsToKeep = ['food_name', 'userid', 'stars']
            df = df.select(columnsToKeep)
            print('columnstokeep')

            df_try = df
            print('df_try')

            userid = '999999999'
            food1 = request.form['food1']
            rate1 = float(request.form['rate1'])
            food2 = request.form['food2']
            rate2 = float(request.form['rate2'])
            food3 = request.form['food3']
            rate3 = float(request.form['rate3'])
            food4 = request.form['food4']
            rate4 = float(request.form['rate4'])
            food5 = request.form['food5']
            rate5 = float(request.form['rate5'])
            print('rate: ', rate1)
            vals = [(food1, userid, rate1), (food2, userid, rate2), (food3, userid, rate3), (food4, userid, rate4),
                    (food5, userid, rate5)]
            print('values: ', vals)
            print(df_try.columns)

            newRows = spark.createDataFrame(vals, df_try.columns)
            print(newRows.show())
            df_try = df_try.union(newRows)
            print(df_try.show())
            a = df_try.toPandas()
            print(a.head(), a.userid.min(), a.userid.max())
            print(a[a.userid == '999999999']) ##benim oyladıklarım yeni eklenen

            df_try = df_try.dropna()
            indexer = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in
                       list(set(df_try.columns) - set(['stars', 'userid']))]
            pipeline = PL(stages=indexer)
            transformed_try = pipeline.fit(df_try).transform(df_try)

            for column in ['food_name_index', 'userid', 'stars']:
                transformed_try = transformed_try.withColumn(column, transformed_try[column].cast('int'))

            print(transformed_try)

            model = ALS(userCol='userid', itemCol='food_name_index', ratingCol='stars', seed=42,
                        rank=50, maxIter=10, regParam=0.03, nonnegative=True, implicitPrefs=False, coldStartStrategy='drop')
            print(model)

            als_new_user = model.fit(transformed_try.dropna())

            nrecommendations = als_new_user.recommendForAllUsers(10)
            print(nrecommendations.show(5))
            recommendationsDF = (nrecommendations.select("userid",explode("recommendations").alias("recommendation")).select("userid", "recommendation.*")).toPandas()
            print(recommendationsDF.head(5))

            collab_rec_999999999 = recommendationsDF[recommendationsDF['userid'] == 999999999]
            print(collab_rec_999999999)  # recommendationDF icindeki bana onerilenleri getirdi
            transformed_try_pd = transformed_try.toPandas()
            print(transformed_try_pd.head())
            print('pandasa cevrildi')
            rated_999999999 = transformed_try_pd[transformed_try_pd['userid'] == 999999999]['food_name_index'].tolist()
            fn = transformed_try_pd[transformed_try_pd['userid'] == 999999999]['food_name'].tolist()
            print('daha once ratelediklerim: ', rated_999999999)
            foods_rec = []
            for i in collab_rec_999999999['food_name_index']:
                if i not in rated_999999999:
                    foods_rec.append(i)

            print(foods_rec, 'bana tavsiye edilen yemeklerin index listesi')
            recommended_foods_name = transformed_try_pd[
                transformed_try_pd['food_name_index'].isin(foods_rec)].drop_duplicates(subset='food_name_index')
            print(recommended_foods_name)  # isimleri
            print(recommended_foods_name.food_name)

            all_data = pd.read_csv('analysis_dfyeni.csv', sep=";")
            #all_data = pd.read_excel("analysis_df.xlsx")
            print(all_data.head(1))
            print('yes')
            all_data = all_data[
                ['food_name', 'ingredients', 'recipe', 'total_time_new', 'nutrition']]  # diger bilgileri
            all_data = all_data.drop_duplicates(subset='food_name')
            print(all_data.head(1))
            print(all_data[all_data.food_name.isin(recommended_foods_name.food_name)])

            headings = ('Food Name', 'Ingredients', 'Recipe', 'Cooking Time', 'Nutrition')
            data = tuple(all_data[all_data.food_name.isin(recommended_foods_name.food_name)].iloc[:5, :].values)
            print(data)

            return render_template('output.html', headings=headings, data=data)

        else:
            render_template('output.html')
    except:
        return 'Error! Please go back'


if __name__ == '__main__':
    app.run()
