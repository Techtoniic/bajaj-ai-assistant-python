
# Importing all the required modules
import io
import random
from mongodb import db
# from bson.binary import Binary
from ocr.tesseract import OCRTesseract
from llm.gpt_turbo import DataPromptGPT
from matching.segmentation import ImageSegment
# from matching.feature_extraction import FeatureExtractionUsingVGG16

customer_id = random.choice(range(1,1000))
doc_type = "invoice"
filename = "medic22.jpg"
#
ocr = OCRTesseract(filename, "./test/invoice/")
text = ocr()
print(text)
#
response, jsn = DataPromptGPT().getOutput(text, category = 1)
print(jsn)
#
entry1 = {"customer_id": customer_id, "doc_type": doc_type, "filename": filename, "invoice_no": jsn[1], "customer_name": jsn[0], "total_cost": jsn[2], "date": jsn[3], "phone_no": jsn[4], "company": jsn[5]}
id1 = db.Database().addData(entry1, category = 1)


# header, body, footer = ImageSegment.HBFCrop("./test/invoice/" + filename)
# model = FeatureExtractionUsingVGG16()
#
# header_feat = model.extract(header)
# # footer_feat = model.extract(footer)
#
# img_bytes = io.BytesIO()
# body.save(img_bytes, format = 'PNG')
# body_binary = Binary(img_bytes.getvalue())
# # print(body)
#
#
# entry2 = {"customer_id": customer_id, "doc_type": doc_type, "filename": filename,"header": header_feat.tolist(), "body": body_binary}
# id2 = db.Database().addData(entry2, category = 2)