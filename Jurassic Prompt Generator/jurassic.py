import requests
import numpy as np
import json
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Define a function to generate prompts
def generate_prompts(user_input):
    api_key = "BD4XigOSJWMj4M77shr0QC91OLpIkpmO"
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Translate user input to English
    translator = Translator()
    english_input = translator.translate(user_input, dest="en").text


    data = {
        "prompt": f"User:\nTranslate in english \"Genetic Algorithm and Particle Swarm Optimization (PSO) are two powerful optimization algorithms that can help improve the performance of large language models\" and generate 5 prompts \n\nPrompts:\n How can \"Genetic Algorithm\" be used to optimize the hyperparameters of a large language model?\n What are the advantages of using \"Particle Swarm Optimization\" over other optimization algorithms in improving the performance of a language model?\n Can \"Genetic Algorithm\" and \"Particle Swarm Optimization\" be combined to improve the accuracy and efficiency of large language models?\n How can \"Particle Swarm Optimization\" be used to optimize the weights of a neural network used in a language model?\n In what ways can \"Genetic Algorithm\" and \"Particle Swarm Optimization\" help to reduce the computational complexity of training a large language model?\n##\n\nUser:\nTranslate in english \"{user_input}\" and generate 5 prompts \n\nPrompts:",
        "numResults": 1,
        "maxTokens": 2048,
        "temperature": 0.9,
        "topKReturn": 0,
        "topP":1,
        "countPenalty": {
            "scale": 0,
            "applyToNumbers": False,
            "applyToPunctuations": False,
            "applyToStopwords": False,
            "applyToWhitespaces": False,
            "applyToEmojis": False
        },
        "frequencyPenalty": {
            "scale": 0,
            "applyToNumbers": False,
            "applyToPunctuations": False,
            "applyToStopwords": False,
            "applyToWhitespaces": False,
            "applyToEmojis": False
        },
        "presencePenalty": {
            "scale": 0,
            "applyToNumbers": False,
            "applyToPunctuations": False,
            "applyToStopwords": False,
            "applyToWhitespaces": False,
            "applyToEmojis": False
        },
        "stopSequences":["##"]
    }

    response = requests.post("https://api.ai21.com/studio/v1/j2-grande-instruct/complete", headers=headers, data=json.dumps(data))

 
    if response.status_code == 200:
        response_data = json.loads(response.text) 
        completions = response_data["completions"]
        generated_prompts = ""
        relevance_scores = []
        grammatical_scores = []
        for completion in completions:
            generated_prompt = completion["data"]["text"].strip()
            generated_prompts += f"{generated_prompt}\n"
            print(generated_prompt)
        
            # Stop generating prompts if the desired count is reached
            if len(generated_prompts.split("\n")) >= 5:
                break
        
        # Loop to generate scores for each prompt
        for generated_prompt in generated_prompts.split("\n"):
            # Calculate the semantic similarity score for each prompt
            prompt_embedding = model.encode([generated_prompt])[0]
            input_embedding = model.encode([english_input])[0]
            similarity_score = cosine_similarity([prompt_embedding], [input_embedding])[0][0]
            relevance_scores.append(similarity_score)
            print(similarity_score)

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
    return prompt_scores_sorted[:5]
            

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_prompts', methods=['POST'])
def get_prompts():
    user_input = request.form['user_input']
    prompt_scores_sorted = generate_prompts(user_input)
    return render_template('output.html', prompt_scores_sorted=prompt_scores_sorted)

if __name__ == '__main__':
    app.run(debug=True, port=8002)