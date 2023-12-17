# Importing all the required modules
import json
from pymongo import MongoClient

class Database:
    """ This is class dedicated to establish MongoDB Atlas Connection, feeding data into it as well
         accessing them. """

    def __init__(self):
        file = open("./mongodb/info.json")
        js = json.load(file)
        self.client = MongoClient(js['MongoDBContents'][0]['Connection String'])
        print("MongoDb Atlas Connection established...")

        self.db = self.client['bajaj-ai-assistant']
        # post = self.db.entities
        print("MongoDB Database Connection established...")

    def addData(self, entry, category):
        if category == 1:
            post = self.db["content-matrix"]
        elif category == 2:
            post = self.db["image-matrix"]
        else:
            return ValueError("Category out of scope.")

        id = post.insert_one(entry)
        print("Successfully added data to test.entities...")
        return id

    def getData(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __del__(self):
        # self.client.close()
        print("MongoDB database connection lost...")
