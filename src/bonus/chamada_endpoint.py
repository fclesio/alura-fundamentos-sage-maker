# Podemos realizar a chamada do nosso endpoint
# usando o "invoke_endpoint"

import sagemaker
import boto3
import json

sessao_boto = boto3.Session(region_name="us-east-1")
sessao_sagemaker = sagemaker.Session(boto_session=sessao_boto)

runtime = boto3.Session().client('sagemaker-runtime')
 
csv_text = '140000,2,2,1,37,0,0,0,0,0,0,58081,51013,54343,27537,9751,12569,5000,5000,5000,3000,3000,5000'

nome_endpoint='codigoCustomizadoEndpoint'

response = runtime.invoke_endpoint(EndpointName=nome_endpoint,
                                   ContentType='text/csv',
                                   Body=csv_text)

result = json.loads(response['Body'].read().decode())

print(result)
