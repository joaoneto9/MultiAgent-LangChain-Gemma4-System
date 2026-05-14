import io
import pandas as pd
from ollama import chat
from datasets import load_dataset
from huggingface_hub import login

system_think_instruction = """<|think|>
Execute um Protocolo de Busca Ativa por Patologias seguindo esta sequência rigorosa:

1. DESAFIO DA NORMALIDADE: Assuma que a imagem PODE conter alterações discretas. Sua missão é encontrá-las antes de declarar "preservado".

2. Inspeção Óssea e Degenerativa (Foco em Coluna): 
   - Não se limite a procurar fraturas agudas. 
   - Inspecione as bordas das vértebras em busca de osteófitos, redução de espaço discal ou desvios de eixo (espondilose/escoliose). 
   - Se houver qualquer "bico de papagaio" ou irregularidade, você deve relatar.

3. Varredura Pulmonar e Pleural (Foco em Pequenos Nódulos):
   - Analise os pulmões por zonas (ápices, terços médios e bases).
   - Procure por nódulos milimétricos, infiltrados reticulonodulares ou espessamentos pleurais.
   - Verifique se a redução volumétrica ou sinais de sequelas (como TB) estão presentes. Se o pulmão parecer "menor" de um lado, investigue a causa.

4. Análise Cardiovascular Quantitativa:
   - Não estime o ICT visualmente de forma genérica. Compare mentalmente o diâmetro horizontal máximo do coração com o diâmetro interno máximo do tórax.
   - Avalie o contorno da aorta: busque por alongamento, ectasia ou calcificações (ateromatose) no botão aórtico.

5. Vias Aéreas e Mediastino:
   - Cheque a posição da traqueia. Se houver desvio, correlacione com possíveis perdas volumétricas ou massas.

6. REGRA DE OURO DA REDAÇÃO:
   - Se encontrar uma alteração, por mais discreta que seja, você está PROIBIDO de usar as frases padrão "Estruturas ósseas sem alterações" ou "Transparência preservada" para aquele sistema.
   - Priorize descrever a anomalia. O laudo clínico deve ser útil para o médico residente identificar o que foge do padrão.
"""

system_report_format ="""
Você é um médico radiologista. Seu objetivo é gerar um laudo técnico e objetivo.

Título: 'RADIOGRAFIA DO TÓRAX – PA E PERFIL'

DIRETRIZES DE CONTEÚDO:
1. PRIORIDADE DE ANOMALIAS: Liste primeiro qualquer achado patológico ou variante anatômica encontrada. Seja descritivo (ex: "Opacidade focal em base pulmonar direita").
2. ESTRUTURAS OBRIGATÓRIAS (Mesmo que normais): Você DEVE obrigatoriamente declarar o estado das seguintes estruturas usando exatamente estes termos se estiverem normais:
    - Transparência pulmonar (ex: "- Transparência pulmonar preservada.")
    - Seios costofrênicos (ex: "- Seios costofrênicos livres.")
    - Mediastino e Silhueta Cardíaca (ex: "- Mediastino e silhueta cardíaca dentro da normalidade.")
    - Estruturas ósseas (ex: "- Estruturas ósseas visualizadas sem alterações.")

3. REGRA DE EXCLUSÃO: Não mencione outras estruturas (como traqueia, hilos ou partes moles) a menos que apresentem anomalias.
4. FORMATAÇÃO: Use apenas bullet points (-). Não inclua textos introdutórios, conclusões ou comentários fora do formato de laudo.
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

