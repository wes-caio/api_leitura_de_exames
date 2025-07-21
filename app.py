# ==============================================================================
# --- BIBLIOTECAS (IMPORTS) ---
# ==============================================================================
import os
import base64  # Essencial para decodificar a imagem
from flask import Flask, request, jsonify  # Componentes da API Flask
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from openai import AzureOpenAI

# ==============================================================================
# --- 1. CONFIGURAÇÃO E CREDENCIAIS (A PARTIR DE VARIÁVEIS DE AMBIENTE) ---
# ==============================================================================
# O código agora lê as credenciais do ambiente. Isso é seguro para o GitHub.

# --- Credenciais do Document Intelligence (OCR) ---
doc_intel_endpoint = os.environ.get("DOC_INTEL_ENDPOINT")
doc_intel_key = os.environ.get("DOC_INTEL_KEY")

# --- Credenciais da Azure OpenAI (NLP) ---
openai_endpoint = os.environ.get("OPENAI_ENDPOINT")
openai_key = os.environ.get("OPENAI_KEY")
openai_api_version = os.environ.get("OPENAI_API_VERSION")
openai_deployment = "gpt-4o" # Este não é secreto, pode ficar aqui.

# --- Validação para garantir que todas as variáveis foram carregadas ---
# Isso ajuda a evitar erros se você esquecer de definir alguma variável.
required_vars = [doc_intel_endpoint, doc_intel_key, openai_endpoint, openai_key, openai_api_version]
if not all(required_vars):
    missing_vars = [
        var_name for var_name, var_value in {
            "DOC_INTEL_ENDPOINT": doc_intel_endpoint,
            "DOC_INTEL_KEY": doc_intel_key,
            "OPENAI_ENDPOINT": openai_endpoint,
            "OPENAI_KEY": openai_key,
            "OPENAI_API_VERSION": openai_api_version
        }.items() if not var_value
    ]
    # Este erro agora será mostrado se você tentar rodar a API sem definir as variáveis.
    raise ValueError(f"ERRO FATAL: As seguintes variáveis de ambiente obrigatórias não foram definidas: {', '.join(missing_vars)}")

# ==============================================================================
# --- 2. INICIALIZAÇÃO DA API FLASK ---
# ==============================================================================
app = Flask(__name__)

# ==============================================================================
# --- 3. LÓGICA DE NEGÓCIO (FUNÇÕES DE IA) ---
# ==============================================================================
# As funções de IA agora recebem os bytes da imagem diretamente.

def extrair_exames_do_documento(image_bytes, content_type):
    """
    Função do OCR: Analisa os bytes de uma imagem e extrai uma lista ordenada
    de todos os itens que foram selecionados.
    """
    print("--- ETAPA 1: INICIANDO OCR ---")
    try:
        client = DocumentIntelligenceClient(endpoint=doc_intel_endpoint, credential=AzureKeyCredential(doc_intel_key))
        poller = client.begin_analyze_document("prebuilt-layout", body=image_bytes, content_type=content_type)
        result: AnalyzeResult = poller.result()
    except Exception as e:
        print(f"❌ FALHA NA ANÁLISE OCR: {e}")
        raise  # Lança a exceção para ser tratada pela API

    if not result.pages:
        print("AVISO: OCR processado, mas nenhuma página foi encontrada.")
        return []

    page = result.pages[0]
    itens_selecionados = []
    if page.selection_marks and page.lines:
        for mark in page.selection_marks:
            if mark.state == 'selected':
                menor_distancia, texto_associado, linha_associada = float('inf'), None, None
                for line in page.lines:
                    dist_y = abs(mark.polygon[1] - line.polygon[1])
                    if dist_y < 20:
                        dist_x = abs(mark.polygon[0] - line.polygon[0])
                        dist_total = dist_y + (dist_x / 10)
                        if dist_total < menor_distancia:
                            menor_distancia, texto_associado, linha_associada = dist_total, line.content, line
                if texto_associado and linha_associada:
                    itens_selecionados.append({
                        "texto": texto_associado,
                        "pos_y": linha_associada.polygon[1],
                        "pos_x": linha_associada.polygon[0]
                    })

    ponto_medio_x = page.width / 2
    coluna_esquerda = sorted([item for item in itens_selecionados if item['pos_x'] < ponto_medio_x], key=lambda x: x['pos_y'])
    coluna_direita = sorted([item for item in itens_selecionados if item['pos_x'] >= ponto_medio_x], key=lambda x: x['pos_y'])
    exames_ordenados = [item['texto'] for item in coluna_esquerda + coluna_direita]
    
    print(f"✅ OCR concluído. {len(exames_ordenados)} exames selecionados encontrados.")
    return exames_ordenados


def obter_mnemonicos_com_nlp(lista_de_exames):
    """
    Função do NLP: Recebe uma lista de exames e usa a IA para encontrar
    os mnemônicos correspondentes para cada um.
    """
    print("--- ETAPA 2: INICIANDO NLP PARA OBTER MNEMÔNICOS ---")
    system_prompt = """
    Você é um assistente especialista em mnemônicos para exames médicos. Sua função é, para cada exame em uma lista que você receber, fornecer o mnemônico correspondente.
    Se um exame não estiver na sua base de conhecimento, retorne "Mnemônico não encontrado".
    Apresente o resultado no formato: "Nome do Exame: [Mnemônico]".

    Sua base de conhecimento é:
    - Exames Laboratoriais: "L.A.B." (Lembre-se de que "L.A.B." se refere a "Laboratórios Analisam Biomas")
    - Exames Anatomopatológicos: "A.P.P." (Análise Patológica de Pacientes)
    - Ecografica: "ECO" (Exame de Controle de Órgãos)
    - Tomografica: "TOMO" (Tomografia para Observação de Morfologia Óssea)
    - Cintilografia: "CINTI" (Cintilografia para Identificação de Novas Tomografias Internas)
    """
    exames_para_processar = "\n".join(f"- {exame}" for exame in lista_de_exames)
    user_prompt = f"Por favor, forneça os mnemônicos para a seguinte lista de exames:\n{exames_para_processar}"

    try:
        client = AzureOpenAI(api_version=openai_api_version, azure_endpoint=openai_endpoint, api_key=openai_key)
        response = client.chat.completions.create(
            model=openai_deployment,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=500
        )
        resultado_nlp = response.choices[0].message.content
        print("✅ NLP concluído. Resposta recebida.")
        return resultado_nlp
    except Exception as e:
        print(f"❌ FALHA NO PROCESSAMENTO NLP: {e}")
        raise # Lança a exceção para ser tratada pela API

# ==============================================================================
# --- 4. DEFINIÇÃO DO ENDPOINT DA API ---
# ==============================================================================
@app.route('/processar_documento', methods=['POST'])
def processar_documento_endpoint():
    """
    Este é o endpoint da API que recebe a imagem em base64.
    """
    print("\n\n--- Nova requisição recebida no endpoint /processar_documento ---")
    
    # 1. Obter e validar o JSON da requisição
    data = request.get_json()
    if not data:
        return jsonify({"error": "Requisição inválida. Corpo deve ser um JSON."}), 400

    attendant_id = data.get('attendant_id')
    pixeon_id = data.get('pixeon_id')
    document = data.get('document')

    if not all([attendant_id, pixeon_id, document]):
        return jsonify({"error": "Campos 'attendant_id', 'pixeon_id' e 'document' são obrigatórios."}), 400

    doc_type = document.get('type')
    doc_content_b64 = document.get('content')

    if not all([doc_type, doc_content_b64]):
        return jsonify({"error": "Campos 'type' e 'content' dentro de 'document' são obrigatórios."}), 400

    # 2. Decodificar a imagem base64
    try:
        image_bytes = base64.b64decode(doc_content_b64)
        content_type = f"image/{doc_type}" if doc_type != "pdf" else "application/pdf"
    except (base64.binascii.Error, TypeError) as e:
        return jsonify({"error": f"String base64 inválida. Detalhes: {e}"}), 400

    # 3. Executar o pipeline de IA (OCR -> NLP)
    try:
        lista_de_exames = extrair_exames_do_documento(image_bytes, content_type)
        
        if not lista_de_exames:
            return jsonify({
                "attendant_id": attendant_id,
                "pixeon_id": pixeon_id,
                "status": "Concluído",
                "message": "Nenhum exame selecionado foi encontrado no documento."
            }), 200

        resultado_final = obter_mnemonicos_com_nlp(lista_de_exames)

        # 4. Retornar a resposta final e bem-sucedida
        return jsonify({
            "attendant_id": attendant_id,
            "pixeon_id": pixeon_id,
            "status": "Sucesso",
            "resultado_mnemonicos": resultado_final.strip().split('\n')
        }), 200

    except Exception as e:
        # Captura qualquer erro que tenha ocorrido nas funções de IA
        return jsonify({"error": f"Ocorreu um erro interno durante o processamento de IA. Detalhes: {str(e)}"}), 500

# ==============================================================================
# --- 5. INICIAR O SERVIDOR DA API ---
# ==============================================================================
if __name__ == '__main__':
    # Validação inicial das credenciais antes de iniciar o servidor
    if any(val == "???" for val in [doc_intel_endpoint, doc_intel_key, openai_endpoint, openai_key, openai_api_version]):
        print("\n❌ ERRO FATAL: Preencha TODAS as variáveis de credenciais no topo do código antes de iniciar a API.")
    else:
        # app.run() é para desenvolvimento. Em produção, usa-se um servidor WSGI como Gunicorn.
        print("🚀 Servidor Flask iniciado. Aguardando requisições em http://127.0.0.1:5000/processar_documento" )
        app.run(debug=True, port=5000)
