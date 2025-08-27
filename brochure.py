import os
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
import json
import gradio as gr

load_dotenv(override=True)

HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

OLLAMA_API_ENDPOINT = os.getenv("OLLAMA_API")

llm = OpenAI(base_url=OLLAMA_API_ENDPOINT,api_key='llama')

def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try to extract only the JSON part between { ... }
        import re
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return {"links": []}
        return {"links": []}

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}


class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        self.url = url
        self.title = ''
        self.body = ''
        try:
            response = requests.get(url, headers=headers)
        except:
            return
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        if not (self.title and self.text): return ''
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

link_system_prompt = """
You are an assistant tasked with selecting only the most relevant links from a company's website 
for creating a brochure. 

Instructions:
1. Return ONLY a valid JSON object in the following format:
{
    "links": [
        {"type": "about page", "url": "https://..."},
        {"type": "careers page", "url": "https://..."}
    ]
}
2. Include only links relevant to a company brochure, such as About Us, Careers, Products/Services, Customers, or Culture pages.
3. Do NOT include Terms of Service, Privacy Policy, email links, login pages, or any unrelated links.
4. If a link is relative (starts with /), convert it to a full HTTPS URL using the website's base URL.
5. Do not include any explanations, text, or commentary outside the JSON.
6. Ensure the JSON is properly formatted and parseable.

Respond strictly according to these instructions.
"""


def get_links_user_prompt(website):
    user_prompt = (
        f"You are an assistant tasked with selecting only the relevant web pages for a brochure about {website.url}. "
        "Analyze the list of links and return a JSON array of full HTTPS URLs that are most useful for a company brochure. "
        "Do not include links to Terms of Service, Privacy Policy, email links, login pages, or external unrelated pages.\n\n"
        "If any link is relative (starts with /), convert it into a full URL using the website's base URL.\n\n"
        "Here is the list of links:\n"
        + "\n".join(website.links)
    )
    return user_prompt



def get_links(url):
    website = Website(url)
    messages = [
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)}
        ]
    response = llm.chat.completions.create(
        model="llama3.2",
        messages=messages,
        stream=False
    )
    print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()


def get_all_details(url):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = safe_json_loads(get_links(url))
    print("Found links:", links)
    for link in links["links"]:
        if not link["url"].startswith("http"):
            link["url"] = url + link["url"]
        result += f"\n\n{link['type']}\n"
        website = Website(link["url"])
        result += website.get_contents()
    return result

system_prompt = (
    "You are a professional marketing assistant tasked with creating a short, engaging brochure "
    "for a company based on the content of its website. Your audience includes prospective customers, "
    "investors, and potential employees. Respond strictly in markdown. \n\n"
    "Guidelines:\n"
    "1. Start with a brief overview of the company.\n"
    "2. Include sections on:\n"
    "   - Company Culture\n"
    "   - Products/Services\n"
    "   - Customers & Partners\n"
    "   - Careers/Job Opportunities\n"
    "3. Keep language concise, positive, and professional.\n"
    "4. Use headings, bullet points, or numbered lists where appropriate.\n"
    "5. Only include information available from the website; do not hallucinate facts.\n"
)


def get_brochure_user_prompt(company_name, url):
    """
    Generates a user prompt for the LLM to create a short brochure
    based on company website content.
    """
    user_prompt = (
        f"You are creating a professional brochure for the company: **{company_name}**.\n\n"
        "Instructions:\n"
        "1. Use the content provided from the landing page and other relevant pages.\n"
        "2. Write a concise, engaging brochure in **Markdown**.\n"
        "3. Include sections such as:\n"
        "   - Overview of the company\n"
        "   - Company culture\n"
        "   - Products or services\n"
        "   - Customers or partners\n"
        "   - Careers / job opportunities (if available)\n"
        "4. Use headings, bullet points, and short paragraphs for readability.\n"
        "5. Do not hallucinate information; only use the provided content.\n\n"
        "Website content:\n"
        f"{get_all_details(url)}"
    )

    return user_prompt[:8_000]


def create_brochure(company_name, url):
    yield "⏳ Generating brochure..."
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ]
    stream = llm.chat.completions.create(
        model="llama3.2",
        messages=messages,
        stream=True
    )
    reply = ""
    print(stream)
    for chunk in stream:
        reply += chunk.choices[0].delta.content or ''
        reply = reply.replace("```","").replace("markdown","")
        print(chunk.choices[0].delta.content or '', end="", flush=True)
        yield reply


with gr.Blocks() as demo:
    gr.Markdown("## 🌐 Website to Brochure Generator")

    with gr.Row():
        website_name = gr.Textbox(label="Enter Website Name", placeholder="Example")
        url_input = gr.Textbox(label="Enter Website URL", placeholder="https://example.com")
    
    generate_btn = gr.Button("Generate Brochure")
    brochure_output = gr.Markdown()

    generate_btn.click(fn=create_brochure, inputs=[website_name,url_input], outputs=brochure_output)

demo.launch()