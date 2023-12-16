# Importing all the required modules
from ocr.tesseract import OCRTesseract
# from llm.google_palm import DataPrompt
from llm.gpt_turbo import DataPromptGPT
from mongodb import db
from matching.similarity import SimilarityCalc
from matching.segmentation import ImageSegment
from matching.feature_extraction import FeatureExtractionUsingVGG16
import random
from PIL import Image

"""ocr = OCRTesseract("report.jpg", "./test/")
text = ocr()
print(text)

response, jsn = DataPromptGPT().getOutput(text, category = 1)
print(jsn)

entry = {"invoice_no": jsn[1], "customer_name": jsn[0], "date": jsn[3], "total_cost": jsn[2], "company": jsn[4]}
id = db.Database().addData(entry, category = 1)

print(id)"""


customer_id = random.choice(range(1,1000))
doc_type = "report"
header, body, footer = ImageSegment.HBFCrop("./test/report.jpg")
header_feat = FeatureExtractionUsingVGG16().extract(header)
body_feat = FeatureExtractionUsingVGG16().extract(body)
footer_feat = FeatureExtractionUsingVGG16().extract(footer)
# print(header_feat, body_feat, footer_feat)

# entry2 = {"customer_id": customer_id, "doc_type": doc_type, "header": header_feat.tolist(), "body": body_feat.tolist(), "footer": footer_feat.tolist()}
# id2 = db.Database().addData(entry2, category = 4)

# print(SimilarityCalc().sift('./test/invoice1.jpg', './test/invoice2.jpg'))
# print(SimilarityCalc().ssd('./test/invoice1.jpg', './test/invoice2.jpg'))
# print(SimilarityCalc().ncc('./test/invoice1.jpg', './test/invoice2.jpg'))