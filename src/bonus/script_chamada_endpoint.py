import boto3

runtime \
    = boto3.Session().client(
        service_name='sagemaker-runtime',
        region_name='us-east-1')

payload \
    = '140000,2,2,1,37,0,0,0,0,0,0,58081,51013,54343,27537,9751,12569,5000,5000,5000,3000,3000,5000'

response \
    = runtime.invoke_endpoint(
        EndpointName='bytebankXGBEndpoint20210114125800',
        ContentType='text/csv',
        Body=payload)

predicao = int(float((response['Body'].read().decode('UTF-8'))))
print(f'Crédito será inadimplente?: {predicao}')
