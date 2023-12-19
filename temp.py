from PIL import Image
import io
from mongodb import db

database = db.Database()
doc = database.queryData(doc_type = "prescription", category = 2)

for ele in doc:
    # print(ele)
    Image.open(io.BytesIO(ele['body'])).save("./test/db/prescription/" + ele["filename"])