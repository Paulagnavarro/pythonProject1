from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def train_model(data, classifier_type, max_depth=None, n_neighbors=None):
    X = data[['sq_mt_built', 'sq_mt_useful', 'n_rooms', 'n_bathrooms', 'has_parking']]
    y = data['buy_price']

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    if classifier_type == 'LinearRegression':
        model = LinearRegression()
    elif classifier_type == 'DecisionTree':
        model = DecisionTreeRegressor(max_depth=int(max_depth) if max_depth else None)
    elif classifier_type == 'KNN':
        model = KNeighborsRegressor(n_neighbors=int(n_neighbors) if n_neighbors else 5)
    else:
        model = RandomForestRegressor()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    joblib.dump(model, 'model.pkl')

    return {'model': model, 'mae': mae, 'model_type': classifier_type}



def predict(features, model):
    prediction = model.predict([features])
    return prediction[0]  # Sem conversão, mantém em euros.

