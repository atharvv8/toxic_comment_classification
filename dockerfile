FROM python:3.9
WORKDIR E:\jigsaw-toxic-comment-classification-challenge
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python3","./Comments_Toxicity.py"]