from flask import Flask, request, jsonify


app = Flask(__name__)


generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    dialogs = data.get('dialogs')
    temperature = data.get('temperature', 0.6)
    top_p = data.get('top_p', 0.9)
    max_gen_len = data.get('max_gen_len', None)
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")

    
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
