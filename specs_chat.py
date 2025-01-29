import os
import sys
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from falkordb import FalkorDB
from urllib.parse import urljoin
import requests
import copy
import traceback
import glob
import numpy as np
import uuid
import threading
from pydantic import BaseModel
from openai import OpenAI
client = OpenAI()

db = FalkorDB(host='localhost', port=6379)


OPTIONS = {}
DATABASES = ["TIM","Aktensystem"]

app = Flask(__name__)

class StructureElement(BaseModel):
    title: str
    description: str

class Structure(BaseModel):
    content: list[StructureElement] 

TOOLS = [{
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
                    können. Beantworte die Frage des Nutzers ausführlich und vollständig basierend auf den Informationen der Datenbankabfrage. Füge zu jeder Aussage die zugehörige Quelle hinzu. Füge hierzu die URL als Markdown Link bei. \
                    Verwende dazu folgendes Format [[Html Dateiname]#[Identifier]]([URL]). Nutze bei Aufzählungen nur eine Ebene. Die Quelle soll nicht als Aufzählungen aufgelistet werden."

MESSAGES={}
lock = threading.Lock()

def get_model_answer(session_id, messages, user_message, tools = None):
    messages = copy.deepcopy(messages)
    messages.append({"role": "user", "content": user_message })
    completion = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=5000 if tools is None else 1000,    # Increase this to allow longer responses
        messages=messages,
        tools = tools,
    )
    tool_call = completion.choices[0].message.tool_calls
    if tool_call is not None:
        messages.append(completion.choices[0].message)
        for tc in tool_call:
            result = query_graph(session_id,json.loads(tc.function.arguments)["query"])
            messages.append({                               # append result message
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
                })
        completion = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,    # Increase this to allow longer responses
            top_p=0.9,          # Nucleus sampling to ensure high-quality responses
            frequency_penalty=0, # Reduces repeated phrases
            presence_penalty=0,   # Encourages topic exploration
            messages=messages,
            tools = tools,
        )
        
    messages.append({"role":"assistant","content":completion.choices[0].message.content})
    return messages, False if tool_call is None else True

def answer(session_id,user_message):


    MESSAGES[session_id], tool_call = get_model_answer(session_id, MESSAGES[session_id],user_message, TOOLS)
    if tool_call:
        if OPTIONS[session_id]['deep'] == False:
            get_model_answer(session_id, "Überarbeite die Antwort. Strukturiere sie und achte darauf, dass meine Fragen ausführlich und vollumfänglich beantwortet werden. \
                                        Falls dir noch Informationen fehlen, dann stelle eine angepasste Anfrage an die Datenbank. Gib nur das Ergebnis aus. Keine Einleitung \
                                        und keine Zusammenfassung.", TOOLS)
        else:
            tmp_messages = copy.deepcopy(MESSAGES[session_id])
            tmp_messages.append({"role":"user","content":"Erstelle basierend auf den Informationen eine Struktur für ausführliche Antwort. Die Struktur soll aus mehreren Überschriften \
                                bestehen und eine kurze Beschreibung der Fragestellung beinhalten"})
            completion = client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=tmp_messages,
                response_format=Structure,
                max_tokens=1500
            )
            MESSAGES[session_id] = MESSAGES[session_id][0:-4]
            tmp_messages = copy.deepcopy(MESSAGES[session_id])
            structure = completion.choices[0].message.parsed
            content = []
            for s in structure.content:
                title = s.title
                description = s.description
                tmp_messages.append({"role":"user", "content":"Schreibe basierend auf Informationen aus einer Datenbank für Retrieval-Augmented-Generation (RAG) einen Absatz entsprechend einer vorgegebenen Beschreibung der Fragestellung.  \
                                    Stelle hierzu basierend auf der Beschreibung der Fragestellung geeignete Abfragen die Datenbank !\
                                    Füge zu jeder Aussage im Absatz die zugehörige Quelle hinzu. Füge hierzu die URL als Markdown Link bei. \
                                    Verwende dazu folgendes Format [[Html Dateiname]#[Identifier]]([URL]). Nutze bei Aufzählungen nur eine Ebene. Die Quelle soll nicht als Aufzählungen aufgelistet werden. Gib als Antwort nur den Titel und den zugehörigen Absatz zurück. Strukturiere den Text und verwende Markdown\n.\
                                    Titel:\"{}\"\n\nBeschreibung:\"{}\"".format(title,description)})
                res, _ = get_model_answer(session_id, tmp_messages,user_message, TOOLS)
                content.append(res[-1]["content"])
                tmp_messages.pop()
            content = "\n".join(content)
            MESSAGES[session_id], _ = get_model_answer(session_id, MESSAGES[session_id],"Überarbeite folgenden Text. Achte darauf, dass der Inhalt und alle Quellenangaben vollständig sind. Sei ausführlich und lasse inhaltlich nichts weg. Strukturiere den Text in geeigneter Form und verwende Markdown: \n" + content, None)

    return str(MESSAGES[session_id][-1]["content"])

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_siblings_topk(graph,content_id, embedding, top_k):
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


def query_graph(session_id,query):
    print(query)
    context = []
    urls = []
    top_scores = []
    graphs = glob.glob("specs/{}/*.html".format(OPTIONS[session_id]['database']))

    for graph in graphs: 
        graph_name = "/".join(graph.split("/")[-2:])
        print(graph_name)
        embedding = get_embedding(query)
        try:
            g = db.select_graph(graph_name)
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
            ''', params={'fetch_k': 10,'query_vector': list(embedding)})
        
            for l in res.result_set:
                if len(top_scores) < 20:
                    top_scores.append(l[1])
                else:
                    top_scores.sort() 
                    if(l[1] < top_scores[-1]):
                        top_scores.pop()
                        top_scores.append(l[1])
                    else:
                        continue
                try:
                    t,u = get_siblings_topk(g,l[0],embedding,5)
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
    if not session_id in MESSAGES:
        MESSAGES[session_id] = [{"role":"system", "content":assistant_prompt}]
        OPTIONS[session_id] = {"database":"Aktensystem","deep":False}
    if user_message.startswith("OPTION"):
        lock.release()
        return options(session_id,user_message)
    while len(MESSAGES[session_id]) > 10:
        delete_list = []
        for i in range(1,len(MESSAGES[session_id])):
            message = MESSAGES[session_id][i]
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
            MESSAGES[session_id].pop(i-c)
            c +=1
        if len(delete_list) == 0:
            break
    while len(MESSAGES[session_id]) > 20:
        MESSAGES[session_id].pop(1)
        MESSAGES[session_id].pop(1)

    try:
        response_text = answer(session_id,user_message)
    except Exception as e:
        exc_info = sys.exc_info()
        response_text = ''.join(traceback.format_exception(*exc_info))
    lock.release()

    return jsonify({
        'answer': response_text 
    })

def options(session_id,request):
    if "OPTION DATENBANK" in request:
        db = request.split(" ")[-1] 
        if db in DATABASES:
            OPTIONS[session_id]['database'] = db
            graphs = glob.glob("specs/{}/*.html".format(OPTIONS[session_id]["database"])) 
            return jsonify({
                'answer': "LADE SPEZIFIKATIONEN:" + "\n\t".join([ g.split("/")[-1].split(".")[-2] for g in graphs])
            })
    if "OPTION LANG" == request:
            OPTIONS[session_id]['deep'] = True
            return jsonify({
                'answer': 'LANG'
            })
    if "OPTION KURZ" == request:
            OPTIONS[session_id]['deep'] = False
            return jsonify({
                'answer': 'KURZ'
            })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))    
    app.run(host='127.0.0.1', port=port, debug=True)
