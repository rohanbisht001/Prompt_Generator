import numpy as np
import openai
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

def generate_prompts(user_input):
    # Initialize the OpenAI API
    openai.api_key = "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"
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Translate user input to English
    translator = Translator()
    english_input = translator.translate(user_input, dest="en").text

    # Define parameters for the OpenAI API request
    model_engine = "text-davinci-003"
    max_tokens = 1024
    n = 5

    # Send the prompt to the OpenAI API to generate new prompts
    prompt_text =f"User:\nTranslate in english \"Genetic Algorithm and Particle Swarm Optimization (PSO) are two powerful optimization algorithms that can help improve the performance of large language models\" and generate 5 prompts \n\nPrompts:\n How can \"Genetic Algorithm\" be used to optimize the hyperparameters of a large language model?\n What are the advantages of using \"Particle Swarm Optimization\" over other optimization algorithms in improving the performance of a language model?\n Can \"Genetic Algorithm\" and \"Particle Swarm Optimization\" be combined to improve the accuracy and efficiency of large language models?\n How can \"Particle Swarm Optimization\" be used to optimize the weights of a neural network used in a language model?\n In what ways can \"Genetic Algorithm\" and \"Particle Swarm Optimization\" help to reduce the computational complexity of training a large language model?\n##\n\nUser:\nTranslate in english \"{user_input}\" and generate 5 prompts \n\nPrompts:"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt_text,
        max_tokens=max_tokens,
        n=n,
        temperature=0.9,
        top_p=0.8
    )

    # Create a string to store the generated prompts and their scores
    generated_prompts = ""
    relevance_scores = []
    grammatical_scores = []
    prompt_embeddings = []
    for choice in response.choices: 
        generated_prompt = choice.text.strip()
        generated_prompts += f"{generated_prompt}\n"

        # Calculate the semantic similarity score for each prompt
        prompt_embedding = model.encode([generated_prompt])[0]
        input_embedding = model.encode([english_input])[0]
        similarity_score = cosine_similarity([prompt_embedding], [input_embedding])[0][0]
        relevance_scores.append(similarity_score)

        # Calculate the grammatical correctness score for each prompt
        with language_tool_python.LanguageTool('en-US') as tool:
            matches = tool.check(generated_prompt)
        grammatical_score = len(matches) / (len(generated_prompt.split()) + 1)
        grammatical_scores.append(grammatical_score)

    # Calculate the overall score for each prompt
    relevance_weight = 70
    grammatical_weight = 30
    overall_scores = (relevance_weight * np.array(relevance_scores)) + (grammatical_weight * np.array(grammatical_scores))

    # Combine the overall score and prompt text into tuples
    prompt_scores = list(zip(generated_prompts.split("\n"), overall_scores))

    # Sort the prompts based on their overall score
    prompt_scores_sorted = sorted(prompt_scores, key=lambda x: x[1], reverse=True)

    # Return the sorted prompts
    return prompt_scores_sorted


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate_prompts', methods=['POST'])
def get_prompts():
    user_input = request.form['user_input']
    prompt_scores_sorted = generate_prompts(user_input)
    return render_template('output.html', prompt_scores_sorted=prompt_scores_sorted)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
