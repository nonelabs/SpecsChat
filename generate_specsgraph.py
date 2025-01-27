from bs4 import BeautifulSoup
from falkordb import FalkorDB
from urllib.parse import urljoin
import requests
from openai import OpenAI
client = OpenAI()

urls = ["https://chat.gemspecs.de/specs/gemSpec_TI-M_Basis_V1.1.1.html",
		"https://chat.gemspecs.de/specs/gemSpec_TI-Messenger-Client_V1.1.2.html",
		"https://chat.gemspecs.de/specs/gemSpec_TI-M_ePA_V1.1.1.html",
		"https://chat.gemspecs.de/specs/gemSpec_TI-Messenger-FD_V1.1.1.html",
		"https://chat.gemspecs.de/specs/gemSpec_TI-M_Pro_V1.0.1.html",
        "https://chat.gemspecs.de/specs/gemSpec_TI-Messenger-Dienst_V1.1.1.html"]

def traverse_element(element,url,cnt):
    parent_url = url + "#" + element.get('id')
    sibling = element.find_next_sibling()
    while sibling is not None:
        if sibling.get_text() != '':
            element_url = url + "#" + sibling.get('id')
            if sibling.name.startswith("h"):
                text = sibling.get_text(separator="\n", strip=True).replace("'","\"").replace("\xa0", " ")
                if int(sibling.name[1]) <= int(element.name[1]):
                    return sibling,cnt
                else:
                    graph.query(f"""
                            MATCH (p:Heading {{url:'{parent_url}'}})
                            CREATE (p)-[:CONTAINS]->(:Heading {{name:'{text}',url:'{element_url}',cnt:'{cnt}'}})
                            """)
                    cnt+=1
                    sibling,cnt = traverse_element(sibling,url,cnt)
                    continue
            if sibling.name.startswith('p') or sibling.name.startswith('ul'):
                text = ''
                links = []
                while sibling.name.startswith('p') or sibling.name.startswith('ul'):
                    text += sibling.get_text(separator="\n", strip=True).replace("'","\"").replace("\xa0", " ") + "\n"
                    links += [a['href'] for a in sibling.find_all('a', href=True)]
                    sibling = sibling.find_next_sibling()
                    if sibling is None:
                        break
                embedding = list(get_embedding(text)) 
                graph.query(f"""
                            MATCH (p:Heading {{url:'{parent_url}'}})
                            CREATE (p)-[:CONTAINS]->(:Content {{text:'{text}',url:'{element_url}',cnt:'{cnt}',embedding:vecf32({str(embedding)})}})
                        """)
                for l in links:
                    graph.query(f"""
                                MATCH (p:Content {{url:'{element_url}'}})
                                CREATE (p)-[:LINK]->(:Link {{href:'{l}'}})
                            """)

                cnt+=1
                continue
            if sibling.name.startswith('div'):
                text = sibling.get_text(separator="\n", strip=True).replace("'","\"").replace("\xa0", " ")
                links = [a['href'] for a in sibling.find_all('a', href=True)]
                embedding = list(get_embedding(text)) 
                graph.query(f"""
                            MATCH (p:Heading {{url:'{parent_url}'}})
                            CREATE (p)-[:CONTAINS]->(:Content {{text:'{text}',url:'{element_url}',cnt:'{cnt}',embedding:vecf32({str(embedding)})}})
                        """)
                for l in links:
                    graph.query(f"""
                                MATCH (p:Content {{url:'{element_url}'}})
                                CREATE (p)-[:LINK]->(:Link {{link:'{url}#{l}'}})
                            """)
                cnt+=1
        sibling = sibling.find_next_sibling()
    return None, cnt

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

db = FalkorDB(host='localhost', port=6379)
graph = db.select_graph('specs')

for url in urls:
    graph_name = "_".join(url.split("/")[-1].split("_")[1:-1])
    graph = db.select_graph(graph_name)
    try:
        graph.delete()
    except:
        pass
    graph.query("CREATE VECTOR INDEX FOR (p:Content) ON (p.embedding) OPTIONS {dimension:3072, similarityFunction:'cosine'}")
    cnt = 0
    print(url)
    html_content = requests.get(url).content
    graph.query(f"CREATE (:gemSpec {{id:'{url}'}})")
    soup = BeautifulSoup(html_content, "html.parser")
    heading = soup.find("h2",{'id': '1', 'class':'target-element'})

    while heading is not None:
        text = heading.get_text(separator="\n", strip=True).replace("'","\"").replace("\xa0", " ")
        id = heading.get('id')
        print(heading)
        graph.query(f"""
                MATCH (p:gemSpec {{id:'{url}'}})
                CREATE (p)-[:CONTAINS]->(:Heading {{name:'{text}',id:'{id}',url:'{url}#{id}',cnt:'{cnt}'}})
                """)
        cnt+=1
        _, cnt = traverse_element(heading,url,cnt)
        heading = heading.find_next("h2", {'class':'target-element'})





        
        
