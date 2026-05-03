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

def send_to_gemma4(images_patient) -> AIMessage:
    system_prompt = """
    Voce é um médico radiologista especialista em encontrar findings em imagens de raio-x de torax.

        A análize da imagem desse seguir o seguinte pipeline nessa ordem:
    1. Ossos e Partes Moles: Checar fraturas em costelas, clavículas e coluna; avaliar cúpulas diafragmáticas e buscar enfisema subcutâneo ou sombras mamárias.
    2. Vias Aéreas: Avaliar se a traqueia está centralizada ou desviada e a perviedade dos brônquios principais.
    3. Pulmões e Pleura: Buscar opacidades (nódulos, massas, consolidações) ou pneumotórax; checar se os seios costofrênicos estão livres ou velados (derrame).
    4. Coração e Mediastino: Avaliar se há cardiomegalia (ICT > 50%) e checar a anatomia do mediastino, botão aórtico e hilos.
    5. Dispositivos (Se houver): Descrever presença e posicionamento de acessos, cateteres, tubos ou marca-passos.
    6. Descrição das Alterações: Se houver achados anormais, descrever o tipo de lesão e a localização exata (ex: "opacidade em terço inferior do pulmão direito").

        A resposta deve seguir o seguinte formato:
    - Bullet points (ex: '- 1. Ossos e Partes Moles:') de cada analize realizada (1 a 6).
    - Caso encontre alterações descreva quais foram e em qual ponto de análize que indenticou isso (ex: 'Anomaliza x (analise 1)').
    - Responda APENAS com esses Bullet Points em Potugues do Brasil.
    """

    user_prompt = [
        {"type": "text", "text": "Indique os findings dos raios-x de tórax (PA e PG) a seguir:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_patient["image_0"]}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_patient["image_1"]}"}}
    ]

    messages = [
        SystemMessage(system_prompt),
        HumanMessage(user_prompt)
    ] 

    response = llm.invoke(messages)

    return response

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
    dicts_responses = {key: send_to_gemma4(images_patients[key]).text for key in images_patients.keys()}

    return dicts_responses


if __name__ == "__main__":
    dict_responses = images_patient_responses()
    df = pd.Series(dict_responses).to_frame(name="Gemma4-x-ray-findings")

    df.to_json('gemma4_x_ray_findings.json', orient='index', indent=4, force_ascii=False)
    