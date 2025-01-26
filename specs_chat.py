import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from falkordb import FalkorDB
from urllib.parse import urljoin
import requests
import uuid
import threading
from openai import OpenAI
client = OpenAI()

db = FalkorDB(host='localhost', port=6379)


app = Flask(__name__)

tools = [{
    "type": "function",
    "function": {
        "name": "query_rag",
        "description": "Informationen zu den gematik Spezifikationen aus einer Datenbank für Retrieval-Augmented-Generation (RAG) zur Frage des Nutzers abrufen",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Abfrage um Informationen zu den Fragen des Nutzers zu bekommen. Die Abfrage muss alle relevanten Punkte der Anfrage des Nutzers umfassen. \
                        Beispiel: Frage Nutzer: Welche Anforderungen muss der TI-M Fachdienst hinsichtlich der Verschüsselung erfüllen ? Abfrage: TI-M Fachdienst welche Anforderungen zur Verschlüsselung."
                }
            },
            "required": [
                "query"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]


assistant_prompt = "Du bist eine deutschsprachiger Assistent, der dem Nutzer basierend auf Informationen aus einer Datenbank für Retrieval-Augmented-Generation (RAG) Fragen beantwortet. \
                    Sollten die Informationen aus der Datenbank nicht ausreichen um die Frage zu beantworten, so stelle dem Nutzer weitere Fragen um die Datenbank spezifischer abfragen zu \
                    können. Strukturiere deine Antwort gut und gib zu allen Informationen immer die URL auf die Quelle in den Spezifikationen an. Es muss alles nachvollziehbar sein. Verwende Markdown." 

messages={}
lock = threading.Lock()

def answer(session_id,user_message):
    global messages
    messages[session_id].append({"role": "user", "content": user_message})
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages[session_id],
        tools = tools
    )
    tool_call = completion.choices[0].message.tool_calls
    if tool_call is not None:
        messages[session_id].append(completion.choices[0].message)
        for tc in tool_call:
            result = query_graph(json.loads(tc.function.arguments)["query"])
            messages[session_id].append({                               # append result message
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
                })
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages[session_id],
            tools = tools
        )

    messages[session_id].append({"role":"assistant","content":completion.choices[0].message.content})
    return str(completion.choices[0].message.content)

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_siblings(graph,content_id):
    result = graph.ro_query(f"""
    MATCH (c:Content {{url: '{content_id}'}})<-[:CONTAINS]-(h:Heading)-[:CONTAINS]->(s:Content)
    RETURN h, collect(s.text) AS texts, collect(s.url) AS urls
    """)
    return result.result_set[0][1], result.result_set[0][2]
    


def query_graph(query):
    print(query)
    context = []
    urls = []
    top_scores = []
    for graph in ['TI-M_Basis']:
        g = db.select_graph(graph)
        res = g.ro_query(
            '''CALL db.idx.vector.queryNodes
        (
            'Content',
            'embedding',
            $fetch_k,
            vecf32($query_vector)
        )
        YIELD node AS closestNode, score
        RETURN closestNode.url, score
        ''', params={'fetch_k': 10, 'query_vector': list(get_embedding(query))})
    
        for l in res.result_set:
            if len(top_scores) < 5:
                top_scores.append(l[1])
            else:
                top_scores.sort() 
                if(l[1] < top_scores[-1]):
                    top_scores.pop()
                    top_scores.append(l[1])
            c,u = get_siblings(g,l[0])
            for i,u in enumerate(u):
                if not u in urls:
                    context.append(c[i])
                    urls.append((u,l[1]))
    results = ""
    for i,c in enumerate(context):
        if urls[i][1] < top_scores[-1]:
            results += c 
            results += "("+urls[i][0]+")"
    return results


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
    session_id = data.get('session_id')
    lock.acquire()
    if not session_id in messages:
        messages[session_id] = [{"role":"system", "content":assistant_prompt}]
    response_text = answer(session_id,user_message)
    lock.release()

    return jsonify({
        'answer': response_text 
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=True)