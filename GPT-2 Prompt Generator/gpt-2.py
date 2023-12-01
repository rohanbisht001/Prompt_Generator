import numpy as np
import openai
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Set up OpenAI API key and model
openai.api_key = "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"
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Define a function to generate prompts
def generate_prompts(user_input):
    
    # Translate user input to English
    translator = Translator()
    english_input = translator.translate(user_input, dest="en").text
    
    # Set up the OpenAI GPT-2 API request parameters
    prompt_request = {
        "prompt": f'User:- Translate in English "Favor su ayuda, se requiere actualizar plan B2B, al marcar TSQ sistema arroja mensaje ""Cuenta de facturacion con subclase distinta a B2B no puede adquirir productos B2B"" por lo que solicito favor modificar subclase a B2B, se adjunta imagen." and generate 5 prompts .\n\nHere are 5 prompts generated based on the translated text:\n\nWhat is the reason for the B2B plan update request?\nHow can the billing account subclass be modified to B2B?\nCan you provide more information on the TSQ system message that appeared?\nWhat is the impact of having a billing account with a different subclass than B2B?\nAre there any other considerations that need to be taken into account when updating the B2B plan?\n##\n\nUser:- Translate in english "buenso dias solicito de su ayuda para que se le pueda brindar agenda a el cliente ya que el sistema arroja sin capacidad de agendamiento muchas gracias ya se puso la fecha mas gtardea pero no funciono gracias" and generate 5 prompts and display it in tabular format.\n\nHere are 5 prompts generated based on the translated text:\n\nPrompt\nWhy are you requesting this customer be placed on an agenda?\nWhat date and time works best for this customer?\nWhat is the current plan with regards to the customer?\nWhat adjustments will you need to make to the existing schedule in order to service this customer?\nHas having this customer been communicated to the sales force?\n##\n\nUser:\nTranslate in english "{user_input}" and generate 5 prompts \n\nPrompts:',
        "max_tokens": 1024,
        "temperature": 0.9,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": ["##"],
    }

    # Send the API request and get the response
    try:
        response = openai.Completion.create(engine="davinci", **prompt_request)
        
        if response.choices[0].finish_reason == "stop":
            generated_prompts = ""
            relevance_scores = []
            grammatical_scores = []
            
            for prompt in response.choices:
                generated_prompt = prompt.text.strip()
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
        else:
            print("Prompt generation did not complete successfully.")
            return None
        
    except Exception as e:
        print("An error occurred during prompt generation:")
        print(str(e))
        return None

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate_prompts", methods=["POST"])
def get_prompts():
    user_input = request.form["user_input"]
    prompt_scores_sorted = generate_prompts(user_input)
    return render_template("output.html", prompt_scores_sorted=prompt_scores_sorted)


if __name__ == "__main__":
    app.run(debug=True, port=8001)
