from flask import Flask, render_template, request

app = Flask(__name__)
messages = []

def process_message(message):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print("thinking")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    inputs = tokenizer(message, return_tensors="pt")
    outputs = model.generate(**inputs,  max_new_tokens = 2048, min_length = 10, length_penalty = 2, num_beams = 8, no_repeat_ngram_size = 2, early_stopping = True)
    return(tokenizer.batch_decode(outputs, skip_special_tokens=True))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']
        processed_message = process_message(message)
        messages.append(("You", message))
        messages.append(("AI", processed_message))
    return render_template('index.html', messages=messages)

@app.route('/history')
def history():
    return render_template('history.html', messages=messages)

if __name__ == '__main__':
    app.run(debug=False, host = '0.0.0.0', port = 2000)
