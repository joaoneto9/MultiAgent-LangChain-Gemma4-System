import io
import pandas as pd
from ollama import chat
from datasets import load_dataset
from huggingface_hub import login

system_think_instruction = """<|think|>
Siga essa linha de pensamento para a análise das imagens de Raio-x de Toráx (PA/Perfil).

1. Ossos e Partes Moles: Checar fraturas em costelas, clavículas e coluna; avaliar cúpulas diafragmáticas e buscar enfisema subcutâneo ou sombras mamárias.
2. Vias Aéreas: Avaliar se a traqueia está centralizada ou desviada e a perviedade dos brônquios principais.
3. Pulmões e Pleura: Buscar opacidades (nódulos, massas, consolidações) ou pneumotórax; checar se os seios costofrênicos estão livres ou velados (derrame).
4. Coração e Mediastino: Meça explicitamente o ICT. Avaliar se há cardiomegalia (ICT > 50%) e checar a anatomia do mediastino, botão aórtico e hilos.
5. Dispositivos (Se houver): Descrever presença e posicionamento de acessos, cateteres, tubos ou marca-passos.
6. Descrição das Alterações: Se houver achados anormais, descrever o tipo de lesão e a localização exata (ex: "opacidade em terço inferior do pulmão direito").
"""

system_report_format ="""
Você é um médico especialista no setor de Radiologia e seu objetivo é fornecer um laudo médico seguindo estritamente esse formato:

Título: 'RADIOGRAFIA DO TÓRAX – PA E PERFIL'
Formato dos Findings:
    - Bullet Points (-).
    - Qualquer aspecto anomalo que foi encontrado durante a análise das imagens deve ser indicado.
    - Caso não encontrar nenhuma anomlia na análise, não mencione.
    - Contudo, caso o aspecto das estruturas abaixo estejam preservadas, indique que está tudo bem de forma explicita:
        1. 'transparência pulmonar' (ex: - Transparência pulmonar preservada.)
        2. 'Seios costofrênicos' (ex: - Seios costofrênicos livres.)
        3. 'Mediastinos' (ex: Mediastino sem alterações.)
        4. 'Estruturas ósseas' (ex: Estruturas ósseas visualizadas sem alterações.)
"""

def get_image_bytes(pil_img):
        byte_arr = io.BytesIO()

        pil_img.save(byte_arr, format='PNG') # foto esta no formato PNG
        return byte_arr.getvalue()

def convert_to_conversation(sample):
    conversation = [
        {
            "role": "system",
            "content": system_think_instruction
        },
        {
            "role": "system",
            "content": system_report_format
        },
        {
            "role": "user",
            "content": "Fornaça o Laudo das imagens (PA e Perfil) seguindo a linha de pensamento e o formato de laudo determinado.",
            "images": [
                get_image_bytes(sample["pa_image"]), 
                get_image_bytes(sample["perfil_image"])
            ]
        }
    ]

    return { "messages": conversation }

if __name__ == "__main__":
    import os 

    login(os.getenv("HUGGING_FACE_TOKEN"))
    MODEL_ID = "gemma4:e4b"
    DATASET_ID = "guilhermenf/huac_chest_xray_reports_images"
    dataset_to_test = load_dataset(DATASET_ID, split="test")

    dataset_test = [convert_to_conversation(sample)
                    for sample in dataset_to_test]
    model_response = []
    model_thinking = []

    responses = {
         "requisition": dataset_to_test["requisition"],
         "report_expected": dataset_to_test["report"]
    }

    counter = 0
    for each_conversation in dataset_test:
        response = chat(
            MODEL_ID,
            messages=each_conversation["messages"],
            think=True
        )

        model_response.append(response["message"]["content"])
        model_thinking.append(response["message"]["thinking"])

        counter += 1
        print(f"Laudo Feito, restam: {len(dataset_test) - counter} laudos")

    responses["model_response"] = model_response
    responses["model_thinking"] = model_thinking

    df = pd.DataFrame(responses)
    df.set_index('requisition', inplace=True)

    df.to_json("gemma4-response-x-expected-findings.json", orient='index', indent=4, force_ascii=False)

