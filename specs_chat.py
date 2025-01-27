import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from falkordb import FalkorDB
from urllib.parse import urljoin
import requests
import numpy as np
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


assistant_prompt = "Du bist eine deutschsprachiger Assistent, der dem Nutzer hilft basierend auf Informationen aus einer Datenbank für Retrieval-Augmented-Generation (RAG) seine Fragen zu beantworten. \
                    Sollten die Informationen aus der Datenbank nicht ausreichen um die Frage zu beantworten, so stelle dem Nutzer weitere Fragen um die Datenbank spezifischer abfragen zu \
                    können. Beantworte die Frage des Nutzers ausführlich und vollständig. Füge zu jeder Aussage die zugehörige Quelle hinzu. Nutze Markdown. Bei Aufzählungen nutze nur eine Ebene. Die Quelle soll nicht als Aufzählungen aufgelistet werden."

messages={}
lock = threading.Lock()



def answer(session_id,user_message):
    global messages
    messages[session_id].append({"role": "user", "content": user_message })
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
            max_tokens=1000,    # Increase this to allow longer responses
            top_p=0.9,          # Nucleus sampling to ensure high-quality responses
            frequency_penalty=0, # Reduces repeated phrases
            presence_penalty=0,   # Encourages topic exploration
            messages=messages[session_id],
            tools = tools
        )
    if not tool_call is None:
        messages[session_id].append({"role":"assistant","content":completion.choices[0].message.content})
        messages[session_id].append({"role": "user", "content": "Überarbeite die Antwort. Strukturiere sie und achte darauf, dass meine Fragen ausführlich und vollumfänglich beantwortet werden. Falls dir noch Informationen fehlen, dann stelle eine angepasste Anfrage an die Datenbank. Gib nur das Ergebnis aus. Keine Einleitung und keine Zusammenfassung."})
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
                max_tokens=1000,    # Increase this to allow longer responses
                top_p=0.9,          # Nucleus sampling to ensure high-quality responses
                frequency_penalty=0, # Reduces repeated phrases
                presence_penalty=0,   # Encourages topic exploration
                messages=messages[session_id],
                tools = tools
            )
    messages[session_id].append({"role":"assistant","content":completion.choices[0].message.content})
    return str(completion.choices[0].message.content)

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_siblings(graph,content_id, embedding, top_k):
    result = graph.ro_query(f"""
    MATCH (c:Content {{url: '{content_id}'}})<-[:CONTAINS]-(h:Heading)-[:CONTAINS]->(s:Content)
    RETURN h, collect(s.text) AS texts, collect(s.url) AS urls, collect(s.embedding) as embeddings
    """)
    emb = np.array(embedding) / np.linalg.norm(np.array(embedding))
    text = []
    urls = []
    for i,e in enumerate(result.result_set[0][3]):
        e = np.array(e)
        e /= np.linalg.norm(e)
        s = np.sqrt(2*(1 - np.dot(e,emb)))
        text.append((result.result_set[0][1][i],float(s)))
        urls.append((result.result_set[0][2][i],float(s)))
    text.sort(key=lambda x: x[1])
    urls.sort(key=lambda x: x[1])
    return text[0:top_k], urls[0:top_k] 

def query_graph(query):
    print(query)
    context = []
    urls = []
    top_scores = []
    top_sieblings= []
    for graph in ['TI-M_Basis','TI-Messenger-Client','TI-M_ePA','TI-Messenger-FD','TI-M_Pro','TI-Messenger-Dienst']:
        embedding = get_embedding(query)
        try:
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
            ''', params={'fetch_k': 5,'query_vector': list(embedding)})
        
            for l in res.result_set:
                if len(top_scores) < 10:
                    top_scores.append(l[1])
                else:
                    top_scores.sort() 
                    if(l[1] < top_scores[-1]):
                        top_scores.pop()
                        top_scores.append(l[1])
                    else:
                        continue
                try:
                    t,u = get_siblings(g,l[0],embedding,3)
                    for i,uu in enumerate(u):
                        if not (uu[0],l[1]) in urls and not t[i][0] in context:
                            context.append(t[i][0])
                            urls.append((uu[0],l[1]))
                except:
                    continue
        except:
            continue
    results = ""
    for i,c in enumerate(context):
        if urls[i][1] < top_scores[-1]:
            results += c.replace("[<=]"," ").replace("[&lt;=]"," ") 
            results += " Quelle: "+urls[i][0] + ".\n\n\n"
    print(results)
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
    while len(messages[session_id]) > 10:
        delete_list = []
        for i in range(1,len(messages[session_id])):
            message = messages[session_id][i]
            if "role" in message:
                if message["role"] == "tool":
                    if len(delete_list) == 0:
                        delete_list.append(i-1)
                    delete_list.append(i)
                if len(delete_list) > 0 and message["role"] != "tool":
                    break
            else:
                if len(delete_list) > 0:
                    break
        delete_list.sort()
        c = 0
        for i in delete_list:
            messages[session_id].pop(i-c)
            c +=1
        if len(delete_list) == 0:
            break
    while len(messages[session_id]) > 20:
        messages[session_id].pop(1)
        messages[session_id].pop(1)

    try:
        response_text = answer(session_id,user_message)
    except Exception as e:
        print(e)
        response_text = e
    lock.release()

    return jsonify({
        'answer': response_text 
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))    
    app.run(host='127.0.0.1', port=port, debug=True)
