import pandas as pd
from sklearn.model_selection import train_test_split

class ProductData:
    def __init__(self, file_name):  
        self.file_folder = "Files"
        self.file_name = "recom_pivot.csv"
        self.path = f"{self.file_folder}//{self.file_name}"
        self.df_init_recommender = pd.read_csv(self.path, header=0)

        self.file_name = "recom.csv"
        self.path = f"{self.file_folder}//{self.file_name}"
        self.df_orig_recommender = pd.read_csv(self.path, header=0)

    def set_pivot_dataframe_data_types(self):
        self.df_customer_ids = self.df_init_recommender["Customer_ID"].astype("string").to_frame()
        self.df_products = self.df_init_recommender.drop(["Customer_ID"], axis=1)
        self.df_products = self.df_products.fillna(0).astype('Int64')
        self.df_recommender = self.df_products.copy()
        self.df_recommender.insert(0, "Customer_ID", self.df_customer_ids)
        self.df_recommender["Customer_ID"] = self.df_recommender["Customer_ID"].astype("string")

    def set_orig_dataframe_data_types(self):
        rename_columns = {"Unnamed: 0" : "ID", "Main_ID": "Customer_ID", "Amount": "Order_Amount"}
        self.df_orig_recommender.rename(columns=rename_columns, inplace=True)
        datatype_columns = {"Customer_ID" : "string",  "Order_Amount": int}
        self.df_orig_recommender = self.df_orig_recommender.astype(datatype_columns)
        drop_columns = ["ID", "Transaction_ID", "Date", "Price", "ItemKey"]
        self.df_orig_recommender = self.df_orig_recommender.drop(drop_columns, axis=1)
        self.df_orig_recommender = self.df_orig_recommender.fillna(0)

    def get_model_data(self):
        self.train_data, self.test_data = train_test_split(self.df_recommender, test_size = 0.2, random_state = 42)