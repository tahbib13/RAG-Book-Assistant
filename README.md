At first create virtual environment ->

uv venv

Then activate ->

.venv\Scripts\activate

Then install requirements.txt ->

uv pip install -r requirements.txt

.env file ->

OPENAI_API_KEY = ""

GROQ_API_KEY = ""

GOOGLE_API_KEY = ""

MISTRAL_API_KEY = ""

HUGGINGFACEHUB_API_TOKEN=

chorma_db folder will automatically create when you upload the book. so if you are going download it then before running the code remove chorma_db folder.

create_database.py is for how database created shows and main02.py shows how use that database and create rag in terminal.

app.py shows the full way how database creating and how implemented rag system. chroma_db always created when you run app.py and upload the the bokk.

run command ->

streamlit run app.py

updated file app02.py -> streamlit run app02.py

<img width="1847" height="801" alt="image" src="https://github.com/user-attachments/assets/ff4d60bb-317c-4bac-8a01-c84848cde81f" />

<img width="1862" height="822" alt="image" src="https://github.com/user-attachments/assets/75eb45d9-dcea-4275-abd1-b9f48b2fe8cb" />

<img width="1780" height="755" alt="image" src="https://github.com/user-attachments/assets/d1ec72d9-168f-4c00-bd67-aef794b7a739" />

<img width="1867" height="795" alt="image" src="https://github.com/user-attachments/assets/120c2ec3-9d3b-497c-8461-e1a1fe5811c9" />

<img width="1797" height="837" alt="image" src="https://github.com/user-attachments/assets/efae976b-b1e7-4852-931d-5aa989733568" />

<img width="1732" height="822" alt="image" src="https://github.com/user-attachments/assets/88a3caed-b81f-46c8-b893-ec2b9fe5f392" />








mistral is free that's why i used this.

If you have money you can use openai/geminni/groq etc.
