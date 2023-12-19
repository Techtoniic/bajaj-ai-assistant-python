# Importing all the required modules
import os

import numpy as np
from PIL import Image
from matching.similarity import SimilarityCalc
from mongodb import db

from pymongo import MongoClient


class TemplateMatching:
    def __init__(self):
        self.database = db.Database()
        self.sim = SimilarityCalc()

    def headerMatch(self, doc_type, feat1):
        doc = self.database.queryData(doc_type = doc_type, category = 2)
        headerMatchArray = []
        for ele in doc:
            temp = []
            temp = [ele['customer_id'], ele['filename']]
            score = self.sim.ssd(feat1, np.array(ele['header']).astype(np.float32))
            temp.append(score)
            headerMatchArray.append(temp)
            # temp.clear()

        max_item = []
        max_value = 0
        for k in headerMatchArray:
            if max_value < k[2]:
                max_item = k

        return max_item

    def footerMatch(self, doc_type, feat1):
        doc = self.database.queryData(doc_type = doc_type, category = 2)
        headerMatchArray = []
        for ele in doc:
            temp = []
            temp = [ele['customer_id'], ele['filename']]
            score = self.sim.ssd(feat1, np.array(ele['footer']).astype(np.float32))
            temp.append(score)
            headerMatchArray.append(temp)
            # temp.clear()

        max_item = None
        max_value = 0
        for k in headerMatchArray:
            if max_value < k[2]:
                max_item = k

        return max_item


    def bodyMatch(self, doc_type, feat1):
        doc = self.database.queryData(doc_type = doc_type, category = 2)
        headerMatchArray = []
        """for ele in doc:
            temp = []
            temp = [ele['customer_id'], ele['filename']]
            # Image.open(io.BytesIO(ele['body'])).show()
            Image.open(io.BytesIO(ele['body'])).save("./test/db/report/" + ele["filename"])
            feat2 = Image.open("./test/db/report/report1.jpg")
            score = self.sim.sift(np.array(feat1), np.array(feat2))
            temp.append(score)
            headerMatchArray.append(temp)
            # temp.clear()"""

        for file in os.listdir("./test/db/report/"):
            print(file)
            temp = []
            temp.append("./test/db/report/")
            feat2 = Image.open("./test/db/report/" + file)
            score = self.sim.sift(np.array(feat1), np.array(feat2))
            temp.append(score)
            headerMatchArray.append(temp)

        max_item = None
        max_value = 0
        for k in headerMatchArray:
            if max_value < k[1][1]:
                max_item = k

        return max_item

    def contentMatch(self, search_string):
        # Create a regular expression pattern for case-insensitive search
        # search string is a list

        for k in search_string:
            regex_pattern = {"$regex": k, "$options": "i"}

            # Query to find documents
            query = {
                "$or": [
                    {"company": regex_pattern},
                    {"customer_name": regex_pattern},
                    {"invoice_no": regex_pattern},
                    {"doctor_name": regex_pattern},
                    {"phone_no": regex_pattern},
                    {"total_cost": regex_pattern},
                    {"date": regex_pattern}
                ]
            }
        return self.database.getData(query)
        # # Use find to get the result
        # result = your_collection.find(query)
        #
        # # Print the results
        # for document in result:
        #     print(document)

