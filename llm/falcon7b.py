# Import all the required modules
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import torch
import torchvision.models as models
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate import init_empty_weights, disk_offload
from transformers import AutoConfig, AutoModelForCausalLM

large_model = torch.nn.Linear(100000, 100000, device="meta")
config = AutoConfig.from_pretrained("tiiuae/falcon-7b-instruct")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
# device_map = infer_auto_device_map(model)
device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"], dtype="float16")

# Save the model weights
torch.save(model.state_dict(), 'model_weights.pth')

model.load_state_dict(torch.load('model_weights.pth'))

checkpoint = "tiiuae/falcon-7b-instruct" # tiiuae/falcon-40b-instruct
device_map["model.decoder.layers.37"] = "disk"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
"""pipeline = pipeline(
    "text-generation", # task
    model = model,
    tokenizer = tokenizer,
    torch_dtype = torch.bfloat16,
    device_map = device_map,
    offload_folder = "offload",
    offload_state_dict = True,
    max_length = 1024,
    do_sample = True,
    top_k = 10,
    num_return_sequences = 1,
    eos_token_id = tokenizer.eos_token_id
)
"""
# llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

model = AutoModelForCausalLM.from_pretrained(
    # "text-generation", # task
    checkpoint,
    # tokenizer = tokenizer,
    torch_dtype = torch.bfloat16,
    device_map = device_map,
    offload_folder = "offload",
    offload_state_dict = True,
    max_length = 1024,
    do_sample = True,
    top_k = 10,
    num_return_sequences = 1,
    eos_token_id = tokenizer.eos_token_id,
    temperature = 0
)
model = disk_offload(model, os.getcwd())

question = "I trust this message finds you well. We're reaching out to confirm your recent pharmacy transaction with us. Your order, under invoice number INV-2023-001, includes a selection of essential medicines prescribed by the renowned Dr. Emily Johnson. The medicines, namely PainRelief-X (Qty: 2), ColdCure Plus (Qty: 1), and AllergyAway Tablets (Qty: 3), aim to address your specific health needs. For your convenience, we've listed the total amount as $150.00, with a GST of $27.00 (at 18%), bringing the grand total to $177.00. Your continued trust in our services means a lot to us. Should you have any queries or require further assistance, please don't hesitate to reach out at 555-1234."
question2 = """Suraksha  DIAGNOSTICS  Pl,  Lab No.  NBP/24-08-2023/SR8068668  Lab Add.  je  Ly)  Th,  Spe  tb Sr  Patient Name  SUBHAM DAS  Ref Dr.  Dr.SUDIPTA MUKHERJEE  "0  22Y2M18D  Collection Date  me  i  ot  Age  aed  if  Gender  M  Report Date  24/Aug/2023 03:25PM  as  DEPARTMENT OF RADIOLOGY  X-RAY REPORT OF CHEST (PA)  FINDINGS :  Lung parenchyma shows no focal lesion. No gencral alteration of radiographic density. Apices  are clear. Bronchovascular lung markings  are within normal.  Both the hila are normal in size, density and position.  Mediastinum is central. Trachea is in midline.  Domes of diaphragm are smoothly outlined. Position is within normal limits  Lateral costo-phrenic angles are clear.  Cardiac size appears within normal limits.  Bony thorax reveals no definite abnormality.  IMPRESSION:  Normal study.  ADV: Clinical correlation & further relevant investigation.  Kindly note  Please Intimate us for an  2720  mistakes and send the r  port for correction within 7 days.  _  DR. SUBRATA SANYAL  CONSULTANT SONOLOGIST AND RADIOLOGIST  MBBS {CAL}, DMRD {CAL}.  o  Page 1 of 1  Suraksha Diagnostic Private Limited  E-mail: info@surakshanet.com | Website: www.surakshanet.com  a"""


template = """
Give me the name of the customer, the doctor's name, the medicine names, the Invoice number, total cost from the paragraph below. Respond in JSON.
Question: {question}
Answer:"""

template2 = f"""
The below paragraph is obtained using the OCR of a Lab Report. From the below paragraph give me the serial number, patient name, doctor name, issuer company of the report, report date, and what report it is. Respond in JSON.
Input: {question2}
Output:"""
"""prompt = PromptTemplate(template=template2, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=model)


print(llm_chain.run(question2))"""

print(model(template2))