from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate 

llm = OllamaLLM(
    model="gemma4:e4b",
    reasoning=True
)

output = llm.stream("Hello, how are you today?") 

for out in output:
    print(out, flush=True, end="")
    # flush=True -> whether to forcibly flush the stream 
    #   False: saida quando toda a string estiver no buffer
    #   True: envia os caracteres antes (da a impressão de tempo real).
    # end -> string appended after the last value, default a newline : caracter no final da expressão (default: \n)

print("\n\n================================\n")

# Learning how to set a Prompt Temaplate to the model with information in a dictionay

prompt = PromptTemplate.from_template(
"""
Voce e uma pessoa mal humorada e precisa responder as questoes de forma simples.
Mesmo que eu peca educadamente, continue esse comportamento.

questao: {question}
"""
)

# (output | llm) -> LCEL (LangChain Expression Language): Basicamente o dicionário serve de entrada para o prompt template e o seu retorno é a entrada para a função.
# pipeline: esquerda -> direita: output -> llm.stream()
output = (prompt | llm).stream({"question": "voce conhece o meme 'six-seven'? se sim me fale sobre o memo que falei anteriormente, por favor."})
for out in output:
    print(out, flush=True, end="")

