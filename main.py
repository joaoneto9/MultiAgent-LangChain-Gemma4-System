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
Você é um médico radiologista sênior. Sua tarefa é realizar uma análise sistemática e rigorosa de radiografias de tórax, minimizando falsos positivos e garantindo a detecção de alterações vasculares e cardíacas.

DIRETRIZES DE ANÁLISE (Siga rigorosamente esta ordem):

0. Identificação: Determine qual imagem é o Frontal e qual é o Perfil.
1. Ossos e Partes Moles: Avalie integridade óssea (costelas, clavículas, coluna), enfisema subcutâneo e sinais de cirurgias prévias (ex: fios de esternorrafia). Se normal, declare "sem alterações". Ignore sombras de tecidos moles externos (mamas) ao avaliar o pulmão abaixo deles.
2. Vias Aéreas: Avalie a centralização da traqueia e a perviedade dos brônquios.
3. Pulmões e Pleura: CRITÉRIO DE EXCLUSÃO: Se os seios costofrênicos estiverem agudos e livres no perfil, e a transparência pulmonar for mantida, ignore variações de cinza causadas por tecidos moles. Não descreva edema ou opacidades a menos que haja apagamento vascular nítido.
4. Coração e Mediastino: Meça o Índice Cardiotorácico (ICT). Identifique explicitamente se há cardiomegalia (ICT > 0.5). Avalie a morfologia do botão aórtico, buscando especificamente por calcificações (placas ateromatosas) e alargamento do mediastino. Nota: Cardiomegalia isolada NÃO implica necessariamente em edema pulmonar.
5. Dispositivos: Identifique fios de sutura ou outros artefatos.
5. Dispositivos e Artefatos: Descreva presença de fios de sutura, marca-passos, cateteres ou próteses.
6. Síntese das Alterações: Liste apenas achados com evidência visual clara. Se o pulmão estiver limpo, não descreva opacidades. Se o coração estiver aumentado ou a aorta calcificada, este ponto DEVE refletir isso.

REGRAS DE FORMATO:
- Use exatamente este formato: '- N. Título da Categoria: Descrição'.
- Para cada achado anormal em '6. Descrição das Alterações', indique o número do ponto onde ele foi observado (ex: 'Cardiomegalia (Ponto 4)').
- Responda APENAS com os bullet points.
- Idioma: Português do Brasil.
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

    response = llm.invoke(messages, reasoning=True)
    
    response_content = response.content
    reasoning = response.additional_kwargs.get("reasoning_content")

    print(reasoning)
    print(response_content)

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
    