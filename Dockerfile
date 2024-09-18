# Use a imagem base oficial do Python
FROM python:3.10-slim

# Defina o diretório de trabalho no container
WORKDIR /app

# Copie o arquivo requirements.txt para o container
COPY requirements.txt .

# Instale as dependências da aplicação
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código da aplicação para o container
COPY . .

# Exponha a porta que o Dash usará (geralmente 8050)
EXPOSE 8050

# Defina o comando para iniciar a aplicação (usando gunicorn para produção)
CMD ["gunicorn", "-b", "0.0.0.0:8050", "app:server"]
