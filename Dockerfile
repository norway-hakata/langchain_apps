FROM python:3
USER root

WORKDIR /langchain_and_streamlit

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "initial_page.py"]
