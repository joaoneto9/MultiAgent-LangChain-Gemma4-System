import re
import pandas as pd
from pathlib import Path
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import Golden, SingleTurnParams

SYSTEM_INPUT_LLM = """
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

RESUMO GERAL (OBRIGATÓRIO):
- No fim da análize de todos os pontos (0 a 6) realize um resumo geral da análise no seguinte formato.
- O início dessa etapa deve ser acompanhada do 'título': RADIOGRAFIA DO TÓRAX – PA E PERFIL.
- Em seguida informe esses aspectos em bullet points ('-') de forma objetiva (OBRIGATÓRIA):
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
USER_INPUT_LLM = "Realize a análise sistemática das radiografias de tórax (Incidências Frontal e Perfil) a seguir, seguindo rigorosamente o protocólo estabelecido:"

path_csv = Path.home() / "Downloads" / "HUAC-DICOM-TORAX-DCM" / "model_x_reports.csv"

df = pd.read_csv(path_csv)

def regex_text(text: str):
    regex = r"\*\*RADIOGRAFIA DO TÓRAX – PA E PERFIL\*\*\n\n\*(.*)"
    resultado = re.search(regex, text, re.DOTALL)

    if resultado:
        return resultado.group(1).strip()

    return None

def regex_model_response(df: pd.DataFrame) -> pd.DataFrame:
    df_clone = df.copy()
    df_clone['model_response'] = [regex_text(response) for response in list(df_clone['model_response'])]

    return df_clone

def df_to_data_eval(df: pd.DataFrame):
    """
    A análize do modelo vai se basear no julgamento dos findigns no final, apenas.
    """
    goldens = []

    for index, row in df.iterrows():
        case = Golden(
            input=f"System: {SYSTEM_INPUT_LLM} \n\n User: {USER_INPUT_LLM}",
            actual_output=row["model_response"],
            retrieval_context= [
                    f"Reasoning do modelo: {row['model_reasoning']}",
                    f"Laudo de referência (Gabarito): {row['real_report']}"
                ],
            expected_output=row["real_report"]
        )

        goldens.append(case)

    return EvaluationDataset(goldens)

def generate_geval_FP_FN_clinic_findings():
    return GEval(
        name="Detecção de FP e FN Clínico",
        criteria="Avaliar a presença de achados clínicos no laudo do modelo e no laudo real (gabarito).",
        evaluation_steps=[
            "Identifique todos os achados/anomalias mencionados no 'Laudo Real' (Gabarito).",
            "Identifique todos os achados/anomalias mencionados no 'Laudo do Modelo'.",
            "FALSO POSITIVO: Liste anomalias que o modelo descreveu, mas que NÃO estão no laudo real. Se houver, a nota deve cair drasticamente.",
            "FALSO NEGATIVO: Liste anomalias que estão no laudo real, mas que o modelo omitiu totalmente. Este é o erro mais grave.",
            "Dê uma nota de 0 a 1, onde 1 significa que não houve nem FPs nem FNs.",
            "No campo de justificativa (reason), descreva explicitamente: 'FPs encontrados: ...' e 'FNs encontrados: ...'"
        ], # evaluation_steps -> indica os passos do modelo juiz a seguir para avaliar
        evaluation_params=[
            SingleTurnParams.INPUT, 
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT
        ]
    )



if __name__ == "__main__":
    df_model_reports = regex_model_response(df)
    dataset = df_to_data_eval(df_model_reports)    

    evaluate(
        test_cases=dataset.test_cases,
        metrics=[generate_geval_FP_FN_clinic_findings()]
    )




