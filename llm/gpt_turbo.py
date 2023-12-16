from openai import OpenAI
import os
from dotenv import load_dotenv
import json

class DataPromptGPT:

    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key = os.environ['OPENAI_API_KEY'])

        self.invoice_prompt = "The below paragraph is extracted from a invoice image using OCR. " \
                              "Give me the name of the customer, the Invoice number, total cost, date and " \
                              "Company name from the paragraph below. Respond in JSON, if you fail to find " \
                              "any entity then skip that entity."
        self.report_prompt = "The below paragraph is extracted from a medical report image using OCR. " \
                              "Give me the name of the customer, the Invoice number, total cost, date and " \
                              "Company name from the paragraph below. Respond in JSON, if you fail to find " \
                              "any entity then skip that entity."
        self.prescription_prompt = "The below paragraph is extracted from a prescription image using OCR. " \
                              "Give me the name of the customer, the Invoice number, total cost, date and " \
                              "Company name from the paragraph below. Respond in JSON, if you fail to find " \
                              "any entity then skip that entity."

    def getOutput(self, text, category):

        """

        :param text:
        :param category: 1 - Invoice; 2 - Medical Report; 3 - Prescription
        :return:
        """
        if category == 1:
            response = self.client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages = [
                    {"role": "system", "content": self.invoice_prompt + text}
                ]
            )

        elif category == 2:
            response = self.client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages = [
                    {"role": "system", "content": self.report_prompt + text}
                ]
            )
        elif category == 3:
            response = self.client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages = [
                    {"role": "system", "content": self.prescription_prompt + text}
                ]
            )
        else:
            return ValueError("Category out of scope.")

        output = response.choices[0].message.content
        # print(output)
        return response, list(json.loads(output).values())
