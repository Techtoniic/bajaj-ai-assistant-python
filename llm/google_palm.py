# Importing all the required modules
import os
import google.generativeai as palm
from dotenv import load_dotenv

class DataPrompt:

    def __init__(self):
        load_dotenv()
        palm.configure(api_key = os.environ['GOOGLE_PALM_API_KEY'])

        self.defaults = {
            'model': 'models/text-bison-001',
            'temperature': 0.7,
            'candidate_count': 1,
            'top_k': 40,
            'top_p': 0.95,
            'max_output_tokens': 1024,
            'stop_sequences': [],
            'safety_settings': [{"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_LOW_AND_ABOVE"},
                                {"category": "HARM_CATEGORY_TOXICITY", "threshold": "BLOCK_LOW_AND_ABOVE"},
                                {"category": "HARM_CATEGORY_VIOLENCE", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                {"category": "HARM_CATEGORY_MEDICAL", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}],
        }

    def getOutput(self, input):
        prompt = f"""
        Give me the name of the customer, the doctor's name, the medicine names, the Invoice number, total cost from the paragraph below. Respond in JSON.
        Input: {input}
        Output:"""

        response = palm.generate_text(
          **self.defaults,
          prompt = prompt
        )

        return response

    def __del__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

# input = "I trust this message finds you well. We're reaching out to confirm your recent pharmacy transaction with us. Your order, under invoice number INV-2023-001, includes a selection of essential medicines prescribed by the renowned Dr. Emily Johnson. The medicines, namely PainRelief-X (Qty: 2), ColdCure Plus (Qty: 1), and AllergyAway Tablets (Qty: 3), aim to address your specific health needs. For your convenience, we've listed the total amount as $150.00, with a GST of $27.00 (at 18%), bringing the grand total to $177.00. Your continued trust in our services means a lot to us. Should you have any queries or require further assistance, please don't hesitate to reach out at 555-1234."
