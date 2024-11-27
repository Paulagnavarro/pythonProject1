from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, data, vectorizer=None):
        self.data = data
        self.model = None
        self.vectorizer = vectorizer

    def train_model(self):
        if 'description' in self.data.columns:
            if not self.vectorizer:
                self.vectorizer = TfidfVectorizer(stop_words='english')
                description_matrix = self.vectorizer.fit_transform(self.data['description'])
            else:
                description_matrix = self.vectorizer.transform(self.data['description'])

            description_df = pd.DataFrame(description_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
            self.data = pd.concat([self.data, description_df], axis=1)
            self.data = self.data.drop(columns=['description'])


        if 'buy_price' in self.data.columns:
            X = self.data.drop(columns=['buy_price'])
            y = self.data['buy_price']
        else:
            raise ValueError("A coluna 'buy_price' não foi encontrada nos dados.")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor()
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Erro médio quadrático (MSE): {mse}")

        return self.model

    def get_model(self):
        return self.model

    def save_model(self, filename):
        joblib.dump(self.model, filename)
        joblib.dump(self.vectorizer, f"{filename}_vectorizer.pkl")
