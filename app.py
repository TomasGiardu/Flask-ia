from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings   
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración
load_dotenv()
app.config['UPLOAD_FOLDER'] = 'upload'
openai_api_key = os.environ.get('OPENAI_API_KEY')
app.secret_key = 'secret'

# Extensiones de archivos permitidos
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Conexión con index.html
@app.route('/')
def index():
    return render_template('index.html')

# Manejo de la ruta para cargar archivos
@app.route('/upload', methods=['POST'])
def upload_file():
    #Guardar los archivos por session
    archivos_cargados = session.get('archivos_cargados', [])

    if 'file' not in request.files:
        return 'No se encontró ningún archivo'
    
    file = request.files['file']

    # Verificar si el archivo es válido y guardarlo
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f'Archivo {filename} subido con éxito')  
        
        # Verificar si el archivo se guardó correctamente
        if os.path.exists(file_path):
            print("El archivo se ha guardado correctamente.")
        else:
            print("Error al guardar el archivo.")

        # Procesar el archivo PDF
        text = ""
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Agregar el nombre del archivo a la lista de archivos cargados
            archivos_cargados.append(filename)  
            # Actualizar la lista en la sesión
            session['archivos_cargados'] = archivos_cargados  
        
        # Separar en chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Crear Embeddings -> Crear Index semántico -> 
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Renderizar en el archivo resultados.html
        return render_template('resultados.html', chunks=chunks)
    
    # Si no se ha enviado archivos o el archivo no es válido, renderizar cargar.html con un mensaje de error
    return render_template('cargar.html', error_message='No se ha enviado ningún archivo o el archivo no es válido')

# Iniciar la depuración con 'python app.py'
if __name__ == '__main__':
    app.run(debug=True)
