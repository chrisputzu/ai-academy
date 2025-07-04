import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('AZURE_API_KEY')

def saluta(): 
    api_key = os.getenv('AZURE_API_KEY')
    if api_key: 
        print (f'Ciao! La tua chiave è: {api_key}')
    else: 
        print('Ciao! Non ho trovato la chiave API.')


if __name__ == "__main__":
    saluta()