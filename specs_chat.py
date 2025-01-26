import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from falkordb import FalkorDB
from urllib.parse import urljoin
import requests
from openai import OpenAI
client = OpenAI()

db = FalkorDB(host='localhost', port=6379)
graph = db.select_graph('specs')


app = Flask(__name__)

messages=[
        {"role": "system", "content": "Du bist ein hilfreicher Assistent der Fragen zu den Spezifikationen beantwortet. Dazu verwendest ausschliesslich die Informationen die du in Context: findest. Wenn diese leer ist erstellst du ein Query um eine RAG Datenbank abzufragen, beantwortest du die Frage nicht ! Sollten die Informationen im Context nicht geeignet sein, die Frage zu beantworten, so beantworte sie nicht. Erwähne den Context selber nicht sondern bitte die Frage zu präzisieren Nenne immer auch die Links die zu den verwendeten Informationen gehörem"}
    ]

def answer(text):
    global messages
    messages.append({"role": "user", "content": " Anweisung: Basierend auf den Fragen des Nutzers formuliere eine Query um eine RAG Datenbank abzufragen. Das Query sollte die Fragen des Nutzers zusammenfassen. Antworte nur mit dem Query. Frage des Nutzers: " + text})
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    query = completion.choices[0].message.content
    context = query_graph(query)
    messages = messages[0:-1]
    messages.append({"role": "user","content": text + context})
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    messages.append({"role":"assistant","content":completion.choices[0].message.content})
    print(messages)
    return str(completion.choices[0].message.content)

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_siblings(content_id):
    result = graph.ro_query(f"""
    MATCH (c:Content {{url: '{content_id}'}})<-[:CONTAINS]-(h:Heading)-[:CONTAINS]->(s:Content)
    RETURN h, collect(s.text) AS texts, collect(s.url) AS urls
    """)
    return result.result_set[0][1], result.result_set[0][2]
    


def query_graph(query):
    res = graph.ro_query(
        '''CALL db.idx.vector.queryNodes
    (
        'Content',
        'embedding',
        $fetch_k,
        vecf32($query_vector)
    )
    YIELD node AS closestNode, score
    RETURN closestNode.url, score
    ''', params={'fetch_k': 107, 'query_vector': list(get_embedding(query))})
    context = []
    links = []
    for l in res.result_set:
        if l[1] < 0.9:
            c,l = get_siblings(l[0])
            context += c 
            links += l
    result = ""
    for i,c in enumerate(context):
        result += c 
        if len(links) < i:
            results += links[i]
    return " Context:"+result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/specs/<path:filename>')
def serve_file(filename):
    return send_from_directory("specs", filename)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    response_text = answer(user_message)

    return jsonify({
        'answer': user_message #response_text
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=True)