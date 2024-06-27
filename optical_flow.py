import mdai
from dotenv import load_dotenv
from os import getenv

load_dotenv('dot.env')
ACCESS_TOKEN = getenv('MDAI_TOKEN')

DOMAIN = "ucsf.md.ai"
PROJECT_ID = "x9N2LJBZ"
DATASET_IDS = [
#   "D_JrYYMV", # Pelvic-1
  "D_V688LQ"    # PECARN Video
]

mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)