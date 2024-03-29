# Importing all the required modules
from ocr.tesseract import OCRTesseract
# from llm.google_palm import DataPrompt
from llm.gpt_turbo import DataPromptGPT
from mongodb import db
from matching.similarity import SimilarityCalc
from matching.segmentation import ImageSegment
from matching.feature_extraction import FeatureExtractionUsingVGG16
from template import TemplateMatching
import random
from PIL import Image
from bson.binary import Binary
import io


ocr = OCRTesseract("report.jpg", "./test/")
text = ocr()
print(text)

response, jsn = DataPromptGPT().getOutput(text, category = 1)
print(jsn)


"""
entry = {"invoice_no": jsn[1], "customer_name": jsn[0], "date": jsn[3], "total_cost": jsn[2], "company": jsn[4]}
id = db.Database().addData(entry, category = 1)

print(id)
"""

customer_id = random.choice(range(1,1000))
doc_type = "report"
filename = ""
header, body, footer = ImageSegment.HBFCrop("./test/report.jpg")

# header_feat = FeatureExtractionUsingVGG16().extract(header)
# img_bytes = io.BytesIO()
# body.save(img_bytes, format='PNG')
# img_binary = Binary(img_bytes.getvalue())
# print(body)

# body_feat = Binary(img_binary)
# footer_feat = FeatureExtractionUsingVGG16().extract(footer)
# print(header_feat, body_feat, footer_feat)

# entry2 = {"customer_id": customer_id, "doc_type": doc_type, "header": header_feat.tolist(), "body": body_feat.tolist(), "footer": footer_feat.tolist()}
# id2 = db.Database().addData(entry2, category = 4)



# print(SimilarityCalc().sift(body_feat, body_feat))
# print(SimilarityCalc().ssd(header_feat, header_feat))
# print(SimilarityCalc().ncc(body_feat, body_feat))

# header, body, footer = ImageSegment.HBFCrop("./test/report.jpg")

header_feat = FeatureExtractionUsingVGG16().extract(header)
footer_feat = FeatureExtractionUsingVGG16().extract(footer)

tm = TemplateMatching()
headerScore = tm.headerMatch("report", header_feat)
footerScore = tm.footerMatch("report", footer_feat)
bodyScore = tm.bodyMatch("report", body)
stringScore = tm.contentMatch(jsn)[0]

# print(headerScore, footerScore, bodyScore, stringScore)
returnUI = {"filename": headerScore[1], "customer_id": headerScore[0], "header_percen": headerScore[2], "body_percen": bodyScore[1][1], "footer_percen": footerScore[2], "content_percen": len(stringScore), "img": 0,"fraud_score": None}
print(returnUI)

import pika
import json

def send_json_data_to_queue(json_data):
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
        channel = connection.channel()

        channel.queue_declare(queue="decision-queue", durable=True)

        channel.basic_publish(
            exchange="",
            routing_key="decision-queue",
            body = json.dumps(json_data),
            properties=pika.BasicProperties(
                delivery_mode=2, # Make the message persistent
            ),
        )

        print(f" [x] Sent JSON data to the queue")

        connection.close()

    except Exception as e:
        raise e

send_json_data_to_queue(returnUI)