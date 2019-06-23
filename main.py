# Car analytics

from boab import Boab
bo = Boab()
bo.load_data('car_data.csv')
bo.fix_types()
bo.fix_missing()
bo.make_discrete_datetime_cols()
print(bo.df.head())

# Build normal regression
feature_list = ['cyclinders', 'displacement', 'hp', 'weight', 'acceleration', 'year', 'origin']
target = 'mpg'
bo.build_regression(feature_list=feature_list, target=target)