# Import necessary libraries
from flask import Flask, render_template, request
from calculate_similarity import load_glove_model,compute_similarity
import nltk
import string

# Create a Flask web application
app = Flask(__name__)


# # Example usage:
glove_model_file = 'glove-model/glove.twitter.27B.25d.pkl'
glove_model = load_glove_model(glove_model_file)

# Sample texts (you can replace this with texts from Wikipedia)

with open('corpus-dataset/HPBook5.txt', 'r') as hp_corpus:
    data = hp_corpus.read()
    sents = data.split('\n')
    sents = sents[1:-1]
    text =  '\n'.join(sents)

texts = nltk.sent_tokenize(text)
texts = [t for t in texts if t != '']
processed_sents = [sent.translate(str.maketrans('', '', string.punctuation)) for sent in sents]
processed_sents = [sent.lower() for sent in processed_sents]




# texts = [
#     "Wikipedia is a multilingual online encyclopedia with articles written by volunteers.",
#     "It was created in January 2001 by Jimmy Wales and Larry Sanger.",
#     "Wikipedia is the largest and most popular general reference work on the internet.",
#     "The name 'Wikipedia' is a portmanteau of two words: wiki and encyclopedia.",
#     # Add more texts as needed
# ]


@app.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        # Get the user input from the form
        query = request.form["query"]
        # Compute the dot product between the query and the texts
        similarities = compute_similarity(query, processed_sents, glove_model)

        # Get the corresponding passages
        results = [f"Text - {texts[0]} | Similarity - {texts[1]}" for texts in similarities]

        # Render the results page with the top 10 most similar passages
        return render_template("results.html", query=query, results=results)

    # Render the initial search page
    return render_template("search.html")

# Run the web application
if __name__ == "__main__":
    app.run(debug=True)
