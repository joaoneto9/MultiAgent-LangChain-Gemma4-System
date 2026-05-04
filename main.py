import base64
import pandas as pd
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

llm = ChatOllama(
    model="gemma4-power:latest",
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
1. Ossos e Partes Moles: Checar fraturas em costelas, clavículas e coluna; avaliar cúpulas diafragmáticas e buscar enfisema subcutâneo ou sombras mamárias.
2. Vias Aéreas: Avaliar se a traqueia está centralizada ou desviada e a perviedade dos brônquios principais.
3. Pulmões e Pleura: Buscar opacidades (nódulos, massas, consolidações) ou pneumotórax; checar se os seios costofrênicos estão livres ou velados (derrame).
4. Coração e Mediastino: Avaliar se há cardiomegalia (ICT > 50%) e checar a anatomia do mediastino, botão aórtico e hilos.
5. Dispositivos (Se houver): Descrever presença e posicionamento de acessos, cateteres, tubos ou marca-passos.
6. Descrição das Alterações: Se houver achados anormais, descrever o tipo de lesão e a localização exata (ex: "opacidade em terço inferior do pulmão direito").

REGRAS DE FORMATO DA RESPOSTA:
- Use exatamente: '- N. Título da Categoria: Descrição'.
- Responda APENAS com os bullet points.
- Idioma: Português do Brasil.

RESUMO GERAL OBRIGATÓRIO:
- No fim da análize de todos os pontos (0 a 6) realize um resumo geral da análise no seguinte formato.
- O início dessa etapa deve ser acompanhada do 'título': RADIOGRAFIA DO TÓRAX – PA E PERFIL.
- Em seguida informe esses aspectos em bullet points ('- Indicatvo') de forma objetiva (OBRIGATÓRIA):
    1. Indique as anomalias nos Ossos e nas Partes Moles, caso não apresente indique que a estruturas ósseas não apresenta alteração. (análise 1)
    2. Indique a situação da traqueia. (análise 2)
    3. Indique a perviedade dos brônquios principais. (análise 2)
    4. Indique como está a transparência dos pulmões. (análise 3)
    5. Indique se os seios costofrênicos estão livres ou velados. (análise 3)
    6. Indique um valor estimado do ICT e se ele está aumentado. Caso não esteja, sinalizar a normalidade. (análise 4)
    7. Indique como está a anatomia do mediastino, botão aórtico e hilos. (análise 4)
    8. Indique (se houver) a presença de dispositivos externos. Caso não apresente, não cite nada. (análise 5)
    9. Por fim, mencione os achados anormais com base em sua análise. (análise 6)
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
    