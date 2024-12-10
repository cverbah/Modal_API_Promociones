from modal import Secret, App, asgi_app, Image as ModalImage
from fastapi import FastAPI, Response, Query, Request, File, UploadFile
from fastapi.responses import JSONResponse
from utils import *
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
import time
import datetime
from dotenv import load_dotenv
import os
import json
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'

app = App(name="fastapi-promos-v1")

image = (ModalImage.debian_slim(python_version="3.11.9")
         .pip_install("numpy==1.26.4", "pandas==1.5.3", "openpyxl==3.1.2",
                      "python-dotenv==1.0.0",
                      "google-cloud-aiplatform==1.60.0", "pillow==10.3.0",
                      "google-auth==2.29.0", "db-dtypes==1.2.0", "unicode==2.9", "Unidecode==1.3.6",
                      "langchain-google-genai==1.0.8", "langchain==0.2.14", "langchain-core==0.2.33",
                      "langchain-experimental==0.0.64", "streamlit-modal==0.1.2", "fastapi~=0.115.6",
                      "requests~=2.32.3")
         )


class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers['X-Process-Time'] = str(process_time)
        return response


promos_app = FastAPI(title='PromosRetailAPI',
                        summary="Promos Retail API", version="1.0",
                        contact={"name": "Cristian Vergara",
                                 "email": "cvergara@geti.cl"})

promos_app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'])
promos_app.add_middleware(MyMiddleware)


def analyze_promo_v2_wrapper(row):
    try:
        return analyze_promo_v2(image_path=row)
    except Exception as e:
        print(f"Error processing {row}: {e}")
        return {'error': str(e), 'image_path': row}


def parallel_process(df, func, workers=4):
    """
    Paraleliza la ejecución de una función sobre un DataFrame usando threads
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(func, df['url_img']))
    return results


@promos_app.get("/")
async def read_root():
    return {"Hello": "World"}


@promos_app.post("/upload-json-file/")
async def upload_json(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Parse JSON content
        data = json.loads(contents)
        loaded_file = {"filename": file.filename, "parsed_data": data}
        print(loaded_file)
        df = pd.DataFrame(data)
        #print(df)
        print('analyzing...')
        start = time.time()
        # analyze imgs with ai assistant
        df['promo_analysis'] = parallel_process(df, analyze_promo_v2_wrapper, workers=8)
        df['descripcion_promo'] = df['promo_analysis'].apply(lambda row: get_promo_data(row, key='promocion'))
        df['duracion_promo'] = df['promo_analysis'].apply(lambda row: get_promo_data(row, key='duracion_promo'))
        df['descuentos_promo'] = df['descripcion_promo'].apply(lambda row: extract_discount(str(row)))
        df['descuentos_promo'] = df['descuentos_promo'].apply(lambda row: format_as_percentage(row))
        df['categorias_en_promo'] = df['promo_analysis'].apply(
            lambda row: get_promo_data(row, key='categorias_en_promo'))
        df['marcas_en_promo'] = df['promo_analysis'].apply(lambda row: get_promo_data(row, key='marcas_en_promo'))
        df['publico_objetivo'] = df['promo_analysis'].apply(lambda row: get_promo_data(row, key='publico_objetivo'))
        df['productos_en_oferta'] = df['promo_analysis'].apply(
            lambda row: get_promo_data(row, key='productos_en_oferta'))
        df['cuotas_sin_interes'] = df['promo_analysis'].apply(lambda row: get_promo_data(row, key='cuotas_sin_interes'))
        df['cupon_app'] = df['promo_analysis'].apply(lambda row: get_promo_data(row, key='cupon_app'))
        df['promociones_envio'] = df['promo_analysis'].apply(lambda row: get_promo_data(row, key='promociones_envio'))
        df.drop(columns='promo_analysis', inplace=True)
        total = round(time.time() - start, 2)

        df_json = df.to_json(orient="records")
        print(f'time taken: {total} secs')
        return JSONResponse(content=json.loads(df_json))

    except json.JSONDecodeError:
        return {"error": "Invalid JSON file"}

    except ValueError as e:
        return {"error": f"Error processing data: {str(e)}"}


@app.function(image=image,
              secrets=[Secret.from_name("automatch-secret-keys")],)
@asgi_app(label='fastapi-promos-v1')
def retail_promos_app():
    import os
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no mostrar los warnings
    print('### Promos Retail - Falabella, Paris, Jumbo ###')

    return promos_app
