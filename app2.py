from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import mlflow
from prefect import task, flow
from sklearn.preprocessing import LabelEncoder

@task
def load_data(path: str, unwanted_cols: List) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.drop(unwanted_cols, axis=1, inplace=True)
    return data

@task
def get_classes(target_data: pd.Series) -> List[str]:
    return list(target_data.unique())

@task
def get_scaler(data: pd.DataFrame) -> Any:
    # scaling the numerical features
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler
@task
def get_label_encoder(data: pd.DataFrame) -> Any:
    label_cat = LabelEncoder()
    label_cat.fit(data)
    return label_cat

@task
def rescale_data_num(data: pd.DataFrame, scaler: Any) -> pd.DataFrame:    
    # scaling the numerical features
    # column names are (annoyingly) lost after Scaling
    # (i.e. the dataframe is converted to a numpy ndarray)
    data_rescaled_num = pd.DataFrame(scaler.transform(data), 
                                columns = data.columns, 
                                index = data.index)
    return data_rescaled_num

@task
def rescale_data_cat(data: pd.DataFrame, label_cat: Any) -> pd.DataFrame:    
    # scaling the numerical features
    # column names are (annoyingly) lost after Scaling
    # (i.e. the dataframe is converted to a numpy ndarray)
    data_rescaled_cat = pd.DataFrame(label_cat.transform(data)
                               )
    return data_rescaled_cat

@task
def split_data(input_: pd.DataFrame, output_: pd.Series, test_data_ratio: float) -> Dict[str, Any]:
    X_tr, X_te, y_tr, y_te = train_test_split(input_, output_, test_size=test_data_ratio, random_state=0)
    return {'X_TRAIN': X_tr, 'Y_TRAIN': y_tr, 'X_TEST': X_te, 'Y_TEST': y_te}

@task
def find_best_model(X_train_transformed: pd.DataFrame, y_train: pd.Series, estimator: Any, parameters: List) -> Any:
    # Enabling automatic MLflow logging for scikit-learn runs
    mlflow.sklearn.autolog(max_tuning_runs=None)
    with mlflow.start_run():        
        clf = GridSearchCV(
            estimator=estimator, 
            param_grid=parameters, 
            scoring='neg_mean_absolute_error',
            cv=5,
            return_train_score=True,
            verbose=1
        )
        clf.fit(X_train_transformed, y_train)

        # Disabling autologging
        mlflow.sklearn.autolog(disable=True)
        
        return clf

# Workflow
@flow

def main(path: str='./data/diamonds.csv',target: str='price', unwanted_cols: List[str]=[], test_size: float=0.2):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diamond price Prediction")
    # Define Parameters

    DATA_PATH = path
    
    TARGET_COL = target
    UNWANTED_COLS = unwanted_cols
    TEST_DATA_RATIO = test_size
    # Load the Data
    dataframe = load_data(path=DATA_PATH, unwanted_cols=UNWANTED_COLS)
    print("null values")
    print(dataframe.isnull().sum())
    # Identify Target Variable
    target_data = dataframe[TARGET_COL]
    input_data = dataframe.drop([TARGET_COL], axis=1)

    # Split the Data into Train and Test
    train_test_dict = split_data(input_=input_data, output_=target_data, test_data_ratio=TEST_DATA_RATIO)

    X_train_cat = train_test_dict['X_TRAIN'].select_dtypes(include=['object'])
    X_train_num = train_test_dict['X_TRAIN'].select_dtypes(include=['int64', 'float64'])
    X_train_cat=pd.DataFrame(X_train_cat)

    
    X_test_cat = train_test_dict['X_TEST'].select_dtypes(include=['object'])
    X_test_num = train_test_dict['X_TEST'].select_dtypes(include=['int64', 'float64'])
    X_test_cat =pd.DataFrame(X_test_cat)

    # Rescaling Train and Test Data
    scaler = get_scaler(X_train_num)
    X_train_num= pd.DataFrame(rescale_data_num(data=pd.DataFrame(X_train_num), scaler=scaler))
    X_test_num = pd.DataFrame(rescale_data_num(data=pd.DataFrame(X_test_num), scaler=scaler))


    label_cat_cut=get_label_encoder(X_train_cat['cut'])
    label_cat_color=get_label_encoder(X_train_cat['color'])
    label_cat_clarity=get_label_encoder(X_train_cat['clarity'])
    X_train_cat['cut']=rescale_data_cat(data=X_train_cat['cut'],label_cat=label_cat_cut)
    X_train_cat['color']=rescale_data_cat(data=X_train_cat['color'],label_cat=label_cat_color)
    X_train_cat['clarity']=rescale_data_cat(data=X_train_cat['clarity'],label_cat=label_cat_clarity)
    print(X_train_cat['color'])
    X_train_cat['color'].fillna(0.0, inplace=True)
    print(X_train_cat['color'].isnull().sum())
    X_train_cat['clarity'].fillna(0.0, inplace=True)
    X_train_cat['cut'].fillna(0.0, inplace=True)

    X_test_cat['cut']=rescale_data_cat(data=X_test_cat['cut'],label_cat=label_cat_cut)
    X_test_cat['color']=rescale_data_cat(data=X_test_cat['color'],label_cat=label_cat_color)
    X_test_cat['clarity']=rescale_data_cat(data=X_test_cat['clarity'],label_cat=label_cat_clarity)
    X_test_cat['color'].fillna(0.0, inplace=True)
    X_test_cat['clarity'].fillna(0.0, inplace=True)
    X_test_cat['cut'].fillna(0.0, inplace=True)

    print("null of num")
    print(X_train_num.isnull().sum())

    X_train_transformed = pd.concat([X_train_num, X_train_cat], axis=1)
    print(X_train_transformed.isnull().sum())
    X_test_transformed = pd.concat([X_test_num, X_test_cat], axis=1)
    
    
    print(X_train_transformed)

     # Model Training
    ESTIMATOR = KNeighborsRegressor()
    HYPERPARAMETERS = [{'n_neighbors':[i for i in range(1, 51)], 'p':[1, 2]}]
    
    #print(X_train_transformed)
    classifier = find_best_model(X_train_transformed, train_test_dict['Y_TRAIN'], ESTIMATOR, HYPERPARAMETERS)
    print(classifier.best_params_)
    print(classifier.score(X_test_transformed, train_test_dict['Y_TEST']))




# Deploy the main function
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta

deployment = Deployment.build_from_flow(
    flow=main,
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(days=7)),
    work_queue_name="ml"
)

deployment.apply()





