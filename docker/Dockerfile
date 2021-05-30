FROM python:3.7
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY app/app.py app.py
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run app.py
