import base64
import pandas as pd
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

llm = ChatOllama(
    model="gemma4:e4b",
    reasoning=True, # Obs: se o modelo usa o reasoning por padrao -> None: <think> label aparecera na resposta, True: removera.
    temperature=0.7,
    top_k=35,
    top_p=0.9
)

def send_to_gemma4(images_patient) -> dict:
    system_prompt = """
Você é um médico radiologista sênior. Sua tarefa é realizar uma análise sistemática e rigorosa de radiografias de tórax, garantindo a detecção de alterações vasculares, cardíacas e estruturais (incluindo achados degenerativos e crônicos).

DIRETRIZES DE ANÁLISE (Siga rigorosamente esta ordem):

0. Identificação: Determine qual imagem corresponde à incidência Frontal (PA/AP) e qual ao Perfil. Use-as de forma complementar.
1. Ossos e Partes Moles: Avalie integridade e morfologia (costelas, clavículas, coluna). Descreva explicitamente sinais degenerativos (osteófitos), alterações de textura óssea (osteopenia), escoliose ou sinais de esternorrafia. 
2. Vias Aéreas: Avalie a centralização e o calibre da traqueia; observe possíveis desvios por massas ou estruturas vasculares.
3. Pulmões e Pleura: Avalie transparência e seios costofrênicos. Mantenha o critério de exclusão para opacidades duvidosas, mas identifique proeminência hilar ou sinais de hipertensão venocapilar se o apagamento vascular for visível.
4. Coração e Mediastino: Meça o ICT (Cardiomegalia se > 0.5). Descreva obrigatoriamente o trajeto da aorta (ex: se é retificado, alongado ou sinuoso). Avalie a configuração do mediastino e dos hilos pulmonares. Caso o contorno cardíaco pareça limítrofe, mencione como 'área cardíaca no limite superior da normalidade' em vez de apenas 'normal'.
5. Dispositivos e Artefatos: Identifique marca-passos, fios de sutura, cateteres ou próteses.
6. Síntese das Alterações: Liste todos os achados anormais identificados anteriormente. Se algo foi notado nos pontos 1 a 5, DEVE ser sumarizado aqui com a devida referência (ex: 'Aorta ectasiada (Ponto 4)').

REGRAS DE FORMATO:
- Use exatamente: '- N. Título da Categoria: Descrição'.
- Responda APENAS com os bullet points.
- Idioma: Português do Brasil.
"""

    user_prompt = [
        {"type": "text", "text": "Realize a análise sistemática das radiografias de tórax (Incidências Frontal e Perfil) a seguir, seguindo rigorosamente o protocolo estabelecido:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_patient["image_0"]}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_patient["image_1"]}"}}
    ]

    messages = [
        SystemMessage(system_prompt),
        HumanMessage(user_prompt)
    ] 

    response = llm.invoke(messages, reasoning=True)
    
    response_content = response.content
    reasoning = response.additional_kwargs.get("reasoning_content")

    return {
        "reasoning": reasoning,
        "response": response_content
    }

def get_images_jpg() -> dict:
    path = Path.home() / "Downloads" / "HUAC-DICOM-TORAX-DCM" / "jpg-images" # obs: para usar path.home() tem que usar / para separa os diretorios
    images_encoded = {}

    # Verificação de segurança
    if not path.exists():
        print(f"Diretório não encontrado: {path}")
        return {}
    
    for image_path in path.rglob("*.jpg"):
        dict_father = image_path.parent.name
        
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

                if dict_father not in images_encoded:
                    images_encoded[dict_father] = {}
                
                index = len(images_encoded[dict_father])
                images_encoded[dict_father][f"image_{index}"] = encoded_string

        except Exception as e:
            print(f"Erro ao ler {image_path}: {e}")

    return images_encoded

def images_patient_responses():
    images_patients = get_images_jpg()
    dicts_responses = {key: send_to_gemma4(images_patients[key]) for key in images_patients.keys()}

    return dicts_responses


if __name__ == "__main__":
    dict_responses = images_patient_responses()
    df = pd.Series(dict_responses).to_frame(name="Gemma4-x-ray-findings")

    df.to_json('gemma4_x_ray_findings.json', orient='index', indent=4, force_ascii=False)
    