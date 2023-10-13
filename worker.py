import multi_bloom_naivy_beyes_cross_validation
import multi_bloom_bert_cross_validation
from client_python_m2p import M2P
import json
import sys
import os

# Load module specs
f = open('question-classifier-specs.json')
specs = json.load(f)

# Set enviroment configs up. Default values can be changed by altering
# the second argument in each "get" call
QUEUE_SERVER_HOST, QUEUE_SERVER_PORT = os.environ.get(
    "QUEUE_SERVER", "200.17.70.211:10163").split(":")

# (NB, B)
#Cria o objeto do M2P e passa o classificador
MyM2P = M2P(QUEUE_SERVER_HOST, QUEUE_SERVER_PORT, specs, multi_bloom_naivy_beyes_cross_validation.myCode)
# MyM2P = M2P(QUEUE_SERVER_HOST, QUEUE_SERVER_PORT, specs, multi_bloom_bert_cross_validation.myCode)

#Faz a conexão com o servidor do M2P
MyM2P.connect()

#Recebe a função e envia a informação que aparecerá no servidor M2P
MyM2P.run()