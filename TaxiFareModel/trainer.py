from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import clean_data, get_data
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import ml_flow_test
import joblib


MLFLOW_URI = "https://mlflow.lewagon.co/"

EXPERIMENT_NAME = "GER_MUC_phil_TaxiFareModel_1.0"


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())])
        return pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline = Trainer.set_pipeline(self)
        self.pipeline.fit(self.X,self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """returns the value of the RMSE"""

        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def train(self):

        for model in ["linear", "Randomforest"]:
            self.mlflow_run()
            self.mlflow_log_metric("rmse", 4.5)
            self.mlflow_log_param("model", model)



    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')

if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    result = trainer.evaluate(X_test, y_test)
    trainer.train()
    trainer.save_model()
    print('TODO')
